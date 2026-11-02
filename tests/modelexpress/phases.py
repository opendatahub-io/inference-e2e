"""ModelExpress conformance phases.

``ModelExpressPhases`` is mixed into ``TestConformance`` (tests/test_conformance.py)
so these phases run for each testcase after the llm-d phases have validated
the model and before cleanup. They are deselected unless the testcase manifest
declares a ``ModelExpressServer``; the server is applied with the model in
phase 02, so a ModelExpress testcase is a testcase config plus its manifest.

The probes run from inside the cluster: gRPC calls from an engine pod with the
engine's ``modelexpress`` client, and TCP checks from a pod in a namespace of
its own. To add a check, add a ``test_40<letter>_modelexpress_<name>`` method here.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

from conformance.config import TestCase
from conformance.deployer import Deployer
from conformance.modelexpress import (
    CA_FILE,
    ENGINE_CONDITION,
    ENGINE_CONTAINER,
    P2P_STRATEGY,
    SCC_CLUSTER_ROLE,
    TOKEN_PATH,
    EngineWiring,
    ProbeResult,
    ServerRef,
    conditions,
    create_token,
    engine_capabilities,
    engine_service_account,
    engine_wiring_problems,
    loaded_strategies,
    manifest_servers,
    network_policy_problems,
    outsider_pod,
    pod_completion,
    run_probe,
    scc_binding_problems,
    tcp_reachable,
    wait_for_pod_ready,
    wait_for_server_ready,
)

# A ServiceAccount the server's allowlist does not name, for the PERMISSION_DENIED check.
INTRUDER_SERVICE_ACCOUNT = "modelexpress-e2e-intruder"
# Time for a replica that has loaded to publish itself as a READY source.
SOURCE_PUBLISH_TIMEOUT = 120

_MANIFEST_DIR = Path(__file__).resolve().parents[2] / "deploy" / "manifests"


def _log(msg: str):
    print(f"  → {msg}")


def _declares_server(tc: TestCase) -> bool:
    return bool(manifest_servers(_MANIFEST_DIR / tc.deployment.manifest_path, ""))


# Applied per method: a class-level mark would be inherited by every TestConformance phase.
# with_args: a mark called with a lone callable would treat it as the decorated function.
_for_modelexpress_manifests = pytest.mark.applies_when.with_args(_declares_server)


@dataclass
class ModelExpressTarget:
    """The deployed ModelExpressServer and the engine pod the probes run in."""

    server: ServerRef
    service: str
    namespace: str
    wiring: EngineWiring
    pod: str

    def probe(self, deployer: Deployer, **overrides) -> ProbeResult:
        """ListSources from the engine pod; with no overrides, with the engine's own address, CA and token."""
        settings = {
            "address": self.wiring.address,
            "ca_file": CA_FILE if self.wiring.tls and self.wiring.ca_configmap else None,
            "token_path": TOKEN_PATH if self.wiring.token_audience else None,
        }
        settings.update(overrides)
        return run_probe(deployer.kubectl, self.pod, self.namespace, **settings)


def _get_json(deployer: Deployer, kind: str, name: str, namespace: str) -> dict:
    return json.loads(deployer.kubectl("get", kind, name, "-n", namespace, "-o", "json"))


def _wait_for_new_pod(deployer: Deployer, service: str, existing: set[str], timeout: float) -> str:
    deadline = time.monotonic() + timeout
    while True:
        new = sorted(set(deployer.list_workload_pods(service)) - existing)
        if new:
            return new[0]
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out after {timeout:g}s waiting for a new pod of '{service}'")
        time.sleep(5)


def _wait_for_sources(target: ModelExpressTarget, deployer: Deployer, count: int, timeout: float) -> ProbeResult:
    deadline = time.monotonic() + timeout
    while True:
        result = target.probe(deployer)
        if not result.ok or len(set(result.worker_ids)) >= count or time.monotonic() >= deadline:
            return result
        time.sleep(10)


class ModelExpressPhases:
    """ModelExpress server, engine wiring, P2P and security checks for testcases whose manifest declares a server."""

    @pytest.fixture(scope="class")
    def mx_target(self, deployer: Deployer, tc: TestCase, test_mode: str) -> ModelExpressTarget:
        if test_mode != "discover" and not deployer.is_deployed(tc.name):
            pytest.skip(f"skipped — deploy failed or was skipped for '{tc.name}'")
        servers = manifest_servers(deployer.manifest_dir / tc.deployment.manifest_path, deployer.namespace)
        llmisvc = _get_json(deployer, "llminferenceservice", tc.name, deployer.namespace)
        pods = deployer.list_workload_pods(tc.name)
        if not pods:
            pytest.skip(f"'{tc.name}' has no engine pod")
        return ModelExpressTarget(
            server=servers[0],
            service=tc.name,
            namespace=deployer.namespace,
            wiring=EngineWiring.from_service(llmisvc),
            pod=pods[0],
        )

    @_for_modelexpress_manifests
    def test_40a_modelexpress_server_ready(self, deployer: Deployer, tc: TestCase, mx_target: ModelExpressTarget):
        """ModelExpressServer Ready and NetworkPolicyReady, at the address the engines use, with the policy it asks for."""
        server = wait_for_server_ready(
            deployer.kubectl, mx_target.server, timeout=tc.deployment.ready_timeout.total_seconds()
        )
        endpoint = (server.get("status") or {}).get("endpoint")
        _log(f"{mx_target.server.name} at {endpoint}")
        assert endpoint == mx_target.wiring.address, (
            f"server endpoint {endpoint!r} is not the engines' address {mx_target.wiring.address!r}"
        )
        policy = _get_json(deployer, "networkpolicy", mx_target.server.name, mx_target.server.namespace)
        problems = network_policy_problems(policy, server)
        assert not problems, "; ".join(problems)

    @_for_modelexpress_manifests
    def test_40b_modelexpress_engine_wiring(self, deployer: Deployer, tc: TestCase, mx_target: ModelExpressTarget):
        """KServe reports ModelExpressReady and renders the engine for the server: load format, address, token, CA, SCC."""
        llmisvc = _get_json(deployer, "llminferenceservice", tc.name, deployer.namespace)
        condition = conditions(llmisvc).get(ENGINE_CONDITION, {})
        assert condition.get("status") == "True", (
            f"{ENGINE_CONDITION}={condition.get('status')} {condition.get('reason', '')} {condition.get('message', '')}"
        )

        pod = _get_json(deployer, "pod", mx_target.pod, deployer.namespace)
        service_account = engine_service_account(tc.name)
        problems = engine_wiring_problems(pod, mx_target.wiring)
        actual_account = (pod.get("spec") or {}).get("serviceAccountName")
        if actual_account != service_account:
            problems.append(f"engine runs as ServiceAccount {actual_account!r}, want {service_account!r}")

        capabilities = engine_capabilities(pod)
        if capabilities and deployer.kubectl(
            "get", "clusterrole", SCC_CLUSTER_ROLE, "--ignore-not-found", "-o", "name"
        ):
            _log(f"engine adds {capabilities}; checking its SCC RoleBinding")
            rolebinding = _get_json(deployer, "rolebinding", f"{service_account}-scc", deployer.namespace)
            problems += scc_binding_problems(rolebinding, service_account)
        assert not problems, "; ".join(problems)

    @_for_modelexpress_manifests
    def test_40c_modelexpress_p2p_scale_out(
        self, deployer: Deployer, tc: TestCase, mx_target: ModelExpressTarget, test_mode: str
    ):
        """A replica added to the running service loads its weights from it over P2P and serves."""
        if test_mode == "discover":
            pytest.skip("discover mode — not scaling a service the suite did not deploy")
        before = _wait_for_sources(mx_target, deployer, 1, SOURCE_PUBLISH_TIMEOUT)
        assert before.ok and before.worker_ids, f"no READY source before scale-out: {before.describe()}"
        existing = set(deployer.list_workload_pods(tc.name))
        cold = loaded_strategies(
            deployer.kubectl("logs", mx_target.pod, "-n", deployer.namespace, "-c", ENGINE_CONTAINER)
        )
        _log(f"{mx_target.pod} loaded with {cold}; {len(before.worker_ids)} READY source(s)")

        replicas = len(existing) + 1
        _log(f"Scaling '{tc.name}' to {replicas} replicas")
        deployer.kubectl(
            "patch",
            "llminferenceservice",
            tc.name,
            "-n",
            deployer.namespace,
            "--type=merge",
            "-p",
            json.dumps({"spec": {"replicas": replicas}}),
        )
        timeout = tc.deployment.ready_timeout.total_seconds()
        pod = _wait_for_new_pod(deployer, tc.name, existing, timeout)
        wait_for_pod_ready(deployer.kubectl, pod, deployer.namespace, timeout)

        strategies = loaded_strategies(deployer.kubectl("logs", pod, "-n", deployer.namespace, "-c", ENGINE_CONTAINER))
        _log(f"{pod} loaded with {strategies}")
        assert strategies and set(strategies.values()) == {P2P_STRATEGY}, (
            f"{pod} loaded with {strategies or 'no strategy'}, want {P2P_STRATEGY!r} on every worker"
        )

        after = _wait_for_sources(mx_target, deployer, len(set(before.worker_ids)) + 1, SOURCE_PUBLISH_TIMEOUT)
        assert after.ok, f"ListSources after scale-out: {after.describe()}"
        assert len(set(after.worker_ids)) > len(set(before.worker_ids)), (
            f"the new replica did not publish itself as a source: {before.worker_ids} → {after.worker_ids}"
        )

        base_url = f"{tc.validation.health_scheme.lower()}://localhost:{tc.validation.health_port}"
        response = pod_completion(deployer.kubectl, pod, deployer.namespace, base_url, tc.model.name, "What is 2+2?")
        assert response.get("choices"), f"{pod} returned no completion: {response}"
        assert response.get("usage", {}).get("completion_tokens", 0) > 0, f"{pod} generated no tokens: {response}"

    @_for_modelexpress_manifests
    def test_40d_modelexpress_tls(self, deployer: Deployer, mx_target: ModelExpressTarget):
        """The server only answers TLS clients that trust its CA."""
        if not mx_target.wiring.tls:
            pytest.skip(f"server address {mx_target.wiring.address!r} is not https://")
        trusted = mx_target.probe(deployer)
        _log(f"https with the CA bundle: {trusted.describe()}")
        assert trusted.ok, f"a client trusting the CA bundle was refused: {trusted.describe()}"

        untrusted = mx_target.probe(deployer, ca_file=None)
        _log(f"https with system roots only: {untrusted.describe()}")
        assert not untrusted.ok, "a client without the CA bundle completed the TLS handshake"

        plaintext = mx_target.probe(deployer, address=mx_target.wiring.address.replace("https://", "http://", 1))
        _log(f"plaintext: {plaintext.describe()}")
        assert not plaintext.ok, "the server answered a plaintext client"

    @_for_modelexpress_manifests
    def test_40e_modelexpress_auth(self, deployer: Deployer, mx_target: ModelExpressTarget):
        """Only the allowlisted engine ServiceAccount's token for the server's audience is accepted."""
        audience = mx_target.wiring.token_audience
        if not audience:
            pytest.skip("the service sets no token audience")
        namespace = deployer.namespace
        engine_account = engine_service_account(mx_target.service)
        deployer.kubectl("create", "serviceaccount", INTRUDER_SERVICE_ACCOUNT, "-n", namespace, check=False)
        try:
            cases = [
                ("engine's projected token", {}, None),
                ("no token", {"token_path": None}, "UNAUTHENTICATED"),
                (
                    "engine ServiceAccount token for the API server audience",
                    {"token_path": None, "token": create_token(deployer.kubectl, engine_account, namespace)},
                    "UNAUTHENTICATED",
                ),
                (
                    f"ServiceAccount {INTRUDER_SERVICE_ACCOUNT} outside the allowlist, audience {audience}",
                    {
                        "token_path": None,
                        "token": create_token(deployer.kubectl, INTRUDER_SERVICE_ACCOUNT, namespace, audience),
                    },
                    "PERMISSION_DENIED",
                ),
            ]
            failures = []
            for label, overrides, want in cases:
                result = mx_target.probe(deployer, **overrides)
                _log(f"{label}: {result.describe()}")
                if want is None and not result.ok:
                    failures.append(f"{label} was refused: {result.describe()}")
                elif want is not None and (result.ok or result.code != want):
                    failures.append(f"{label}: got {result.describe()}, want {want}")
            assert not failures, "; ".join(failures)
        finally:
            deployer.kubectl(
                "delete", "serviceaccount", INTRUDER_SERVICE_ACCOUNT, "-n", namespace, "--ignore-not-found"
            )

    @_for_modelexpress_manifests
    def test_40f_modelexpress_network_policy(self, deployer: Deployer, mx_target: ModelExpressTarget):
        """The restrictive profile admits the gRPC port from the test namespace and blocks other namespaces."""
        server = _get_json(deployer, "modelexpressserver", mx_target.server.name, mx_target.server.namespace)
        if not ((server.get("spec") or {}).get("networkPolicy") or {}).get("allowFrom"):
            pytest.skip("compatibility profile — the server admits every source")
        namespace = mx_target.server.namespace
        host = deployer.kubectl(
            "get", "service", mx_target.server.name, "-n", namespace, "-o", "jsonpath={.spec.clusterIP}"
        )
        port = mx_target.wiring.grpc_port
        assert tcp_reachable(deployer.kubectl, mx_target.pod, deployer.namespace, host, port, ENGINE_CONTAINER), (
            f"engine pod {mx_target.pod} cannot reach the server at {host}:{port}"
        )
        _log(f"{mx_target.pod} reaches {host}:{port}")

        api_server = deployer.kubectl(
            "get", "service", "kubernetes", "-n", "default", "-o", "jsonpath={.spec.clusterIP}"
        )
        image = deployer.kubectl(
            "get",
            "deployment",
            mx_target.server.name,
            "-n",
            namespace,
            "-o",
            "jsonpath={.spec.template.spec.containers[0].image}",
        )
        outsider_namespace = f"{deployer.namespace}-mx-outsider"
        with outsider_pod(deployer.kubectl, outsider_namespace, image) as pod:
            assert tcp_reachable(deployer.kubectl, pod, outsider_namespace, api_server, 443), (
                f"{pod} cannot reach the API server at {api_server}:443, so its blocked result would mean nothing"
            )
            reachable = tcp_reachable(deployer.kubectl, pod, outsider_namespace, host, port)
            _log(f"{outsider_namespace}/{pod} → {host}:{port}: {'reachable' if reachable else 'blocked'}")
            assert not reachable, f"a pod in {outsider_namespace} reached the server's gRPC port {host}:{port}"
