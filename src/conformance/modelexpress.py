"""ModelExpress helpers for testcases whose manifest declares a ``ModelExpressServer``.

A ModelExpress testcase manifest deploys a ``ModelExpressServer`` next to an
LLMInferenceService that loads its weights through it, configured by the
``serving.kserve.io/exp-modelexpress-*`` annotations. The conformance phases
check the server and the engine wiring KServe renders, scale the service out
so the new replica loads its weights from a running one over P2P, and probe
the server's TLS, token auth and NetworkPolicy from inside the cluster.

The shipped server image holds only the server binary, so gRPC probes run in
an engine pod with the engine's own ``modelexpress`` client
(``modelexpress_probe.py``, through ``kubectl exec``).
"""

from __future__ import annotations

import json
import posixpath
import re
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from conformance.config import load_manifest_documents

SERVER_KIND = "ModelExpressServer"
_ANNOTATION = "serving.kserve.io/exp-modelexpress-"
ADDRESS_ANNOTATION = _ANNOTATION + "address"
TOKEN_AUDIENCE_ANNOTATION = _ANNOTATION + "token-audience"
CA_CONFIGMAP_ANNOTATION = _ANNOTATION + "ca-bundle-configmap"
CA_KEY_ANNOTATION = _ANNOTATION + "ca-bundle-key"
ENGINE_CONDITION = "ModelExpressReady"

# Where KServe mounts the projected token and the CA bundle in the engine's main container.
ENGINE_CONTAINER = "main"
TOKEN_PATH = "/var/run/secrets/modelexpress/token"
CA_FILE = "/etc/modelexpress/ca/ca.crt"
DEFAULT_GRPC_PORT = 8001

# OpenShift ClusterRole for the SCC that allows IPC_LOCK; KServe binds it for engines that add a capability.
SCC_CLUSTER_ROLE = "openshift-ai-llminferenceservice-scc"
# The load strategy that pulls weights from a running replica over NIXL.
P2P_STRATEGY = "rdma"

OUTSIDER_POD = "modelexpress-outsider"
_PROBE_SCRIPT = Path(__file__).with_name("modelexpress_probe.py")
_TRYING_STRATEGY = re.compile(r"\[Worker (\d+)\] Trying strategy: (\S+)")
_FAILED_STRATEGY = re.compile(r"\[Worker (\d+)\] Strategy (\S+) (?:failed|raised unexpected error)")
_P2P_TRANSFER = re.compile(r"\[Worker (\d+)\] \[TIMING\] RDMA transfer complete: (.+)$")
_UCX_TABLE_ROW = re.compile(r"^\[[\d.]+\] \[[^\]]+\]\s+(\|.*\|)\s*$")
_UCX_LANE = re.compile(r"([a-z][a-z0-9_]*)/[^\s,]+")

Kubectl = Callable[..., str]


@dataclass(frozen=True)
class ServerRef:
    name: str
    namespace: str


@dataclass(frozen=True)
class EngineWiring:
    """How an LLMInferenceService's ModelExpress annotations ask KServe to connect its engines."""

    address: str
    token_audience: str = ""
    ca_configmap: str = ""
    ca_key: str = ""

    @classmethod
    def from_service(cls, llmisvc: dict) -> EngineWiring:
        annotations = (llmisvc.get("metadata") or {}).get("annotations") or {}
        return cls(
            address=annotations.get(ADDRESS_ANNOTATION, "").strip(),
            token_audience=annotations.get(TOKEN_AUDIENCE_ANNOTATION, "").strip(),
            ca_configmap=annotations.get(CA_CONFIGMAP_ANNOTATION, "").strip(),
            ca_key=annotations.get(CA_KEY_ANNOTATION, "cabundle.crt").strip(),
        )

    @property
    def tls(self) -> bool:
        return self.address.startswith("https://")

    @property
    def grpc_port(self) -> int:
        return _split_address(self.address).port or DEFAULT_GRPC_PORT


@dataclass(frozen=True)
class ProbeResult:
    """Outcome of one ListSources call made from an engine pod."""

    ok: bool
    code: str = ""
    detail: str = ""
    worker_ids: tuple[str, ...] = ()

    def describe(self) -> str:
        if self.ok:
            return f"OK, {len(self.worker_ids)} READY source(s)"
        return f"{self.code} {self.detail}".strip()


def engine_service_account(service_name: str) -> str:
    """The ServiceAccount KServe creates for a single-node LLMInferenceService's engine pods."""
    return f"{service_name}-kserve"


def manifest_servers(manifest_path: str | Path, default_namespace: str) -> list[ServerRef]:
    """ModelExpressServers a manifest declares, in manifest order."""
    path = Path(manifest_path)
    if not path.is_file():
        return []
    return [ServerRef(s.name, s.namespace or default_namespace) for s in _servers_in(load_manifest_documents(path))]


def patch_modelexpress_refs(documents: list, primary_name: str | None, service_name: str, namespace: str) -> None:
    """Keep the manifest's ModelExpress wiring pointing at the deployed objects.

    The deployer renames the primary LLMInferenceService to the testcase name and
    deploys into an overridable test namespace. An address annotation that names
    a server the manifest deploys into the test namespace follows the namespace,
    and the servers' allowlists name the test namespace and the renamed
    service's engine ServiceAccount.
    """
    local_servers = {server.name for server in _servers_in(documents) if not server.namespace}

    for document in documents:
        if _kind(document) == SERVER_KIND:
            security = (document.get("spec") or {}).get("security") or {}
            for entry in security.get("allowedServiceAccounts") or []:
                if not isinstance(entry, dict):
                    continue
                entry["namespace"] = namespace
                if primary_name and entry.get("serviceAccount") == engine_service_account(primary_name):
                    entry["serviceAccount"] = engine_service_account(service_name)
        elif _kind(document) == "LLMInferenceService":
            annotations = (document.get("metadata") or {}).get("annotations")
            if isinstance(annotations, dict) and annotations.get(ADDRESS_ANNOTATION):
                annotations[ADDRESS_ANNOTATION] = _address_in_namespace(
                    annotations[ADDRESS_ANNOTATION], local_servers, namespace
                )


def _kind(document: object) -> str | None:
    return document.get("kind") if isinstance(document, dict) else None


def _servers_in(documents: list) -> list[ServerRef]:
    """ModelExpressServer documents, with the namespace they declare ("" when none)."""
    servers = []
    for document in documents:
        metadata = document.get("metadata") if _kind(document) == SERVER_KIND else None
        if isinstance(metadata, dict) and metadata.get("name"):
            servers.append(ServerRef(metadata["name"], metadata.get("namespace") or ""))
    return servers


def _split_address(address: str):
    return urlsplit(address if "://" in address else f"//{address}")


def _address_in_namespace(address: str, servers: set[str], namespace: str) -> str:
    """``address`` with a ``<server>.<namespace>.svc`` host moved to ``namespace`` when ``server`` is in ``servers``."""
    parsed = _split_address(address)
    labels = (parsed.hostname or "").split(".")
    if len(labels) < 3 or labels[0] not in servers or labels[2] != "svc":
        return address
    labels[1] = namespace
    netloc = ".".join(labels) + (f":{parsed.port}" if parsed.port else "")
    moved = parsed._replace(netloc=netloc).geturl()
    return moved if "://" in address else moved.removeprefix("//")


def conditions(resource: dict) -> dict[str, dict]:
    """A resource's status conditions by type."""
    found = (resource.get("status") or {}).get("conditions") or []
    return {c["type"]: c for c in found if isinstance(c, dict) and isinstance(c.get("type"), str)}


def wait_for_server_ready(kubectl: Kubectl, server: ServerRef, timeout: float, poll_interval: float = 5) -> dict:
    """Wait until the ModelExpressServer reports Ready and NetworkPolicyReady; return it."""
    wanted = ("Ready", "NetworkPolicyReady")
    deadline = time.monotonic() + timeout
    while True:
        resource = json.loads(kubectl("get", "modelexpressserver", server.name, "-n", server.namespace, "-o", "json"))
        current = conditions(resource)
        if all(current.get(t, {}).get("status") == "True" for t in wanted):
            return resource
        if time.monotonic() >= deadline:
            detail = "; ".join(
                f"{t}={current.get(t, {}).get('status')} {current.get(t, {}).get('reason', '')} "
                f"{current.get(t, {}).get('message', '')}".strip()
                for t in wanted
            )
            raise TimeoutError(f"Timed out after {timeout:g}s waiting for {SERVER_KIND}/{server.name}: {detail}")
        time.sleep(poll_interval)


def network_policy_problems(policy: dict, server: dict) -> list[str]:
    """Ways the server's NetworkPolicy departs from what its ``spec.networkPolicy`` asks for.

    The policy must be owned by the server. With ``allowFrom`` set (the
    restrictive profile), no ingress rule may admit every source and the gRPC
    port is admitted from exactly ``allowFrom``.
    """
    name = (server.get("metadata") or {}).get("name")
    policy_name = (policy.get("metadata") or {}).get("name")
    problems = []
    owners = [(o.get("kind"), o.get("name")) for o in (policy.get("metadata") or {}).get("ownerReferences") or []]
    if (SERVER_KIND, name) not in owners:
        problems.append(f"NetworkPolicy/{policy_name} is not owned by {SERVER_KIND}/{name} (owners: {owners})")

    spec = server.get("spec") or {}
    allow_from = (spec.get("networkPolicy") or {}).get("allowFrom") or []
    if not allow_from:
        return problems

    grpc_port = spec.get("port") or DEFAULT_GRPC_PORT
    ingress = (policy.get("spec") or {}).get("ingress") or []
    for rule in ingress:
        if not rule.get("from"):
            ports = [p.get("port") for p in rule.get("ports") or []]
            problems.append(f"ingress rule for ports {ports} admits any source")
    grpc_rules = [rule for rule in ingress if any(p.get("port") == grpc_port for p in rule.get("ports") or [])]
    if not grpc_rules:
        problems.append(f"no ingress rule admits the gRPC port {grpc_port}")
    for rule in grpc_rules:
        if rule.get("from") != allow_from:
            problems.append(f"gRPC port {grpc_port} is admitted from {rule.get('from')}, not allowFrom {allow_from}")
    return problems


def _main_container(pod: dict) -> dict | None:
    containers = (pod.get("spec") or {}).get("containers") or []
    return next((c for c in containers if c.get("name") == ENGINE_CONTAINER), None)


def _load_format(args: list[str]) -> str | None:
    for index, arg in enumerate(args):
        if arg == "--load-format" and index + 1 < len(args):
            return args[index + 1]
        if arg.startswith("--load-format="):
            return arg.split("=", 1)[1]
    return None


def engine_capabilities(pod: dict) -> list[str]:
    """Capabilities the engine's main container adds."""
    container = _main_container(pod) or {}
    return list(((container.get("securityContext") or {}).get("capabilities") or {}).get("add") or [])


def engine_wiring_problems(pod: dict, wiring: EngineWiring) -> list[str]:
    """Ways an engine pod's main container departs from the wiring its annotations ask for."""
    container = _main_container(pod)
    if container is None:
        return [f"pod has no {ENGINE_CONTAINER!r} container"]
    problems = []
    load_format = _load_format(container.get("args") or [])
    if load_format != "modelexpress":
        problems.append(f"--load-format is {load_format!r}, want 'modelexpress'")

    env = {e["name"]: e.get("value") for e in container.get("env") or [] if isinstance(e, dict) and "name" in e}
    for name in ("MX_SERVER_ADDRESS", "MODEL_EXPRESS_URL"):
        if env.get(name) != wiring.address:
            problems.append(f"env {name}={env.get(name)!r}, want {wiring.address!r}")
    if not env.get("MX_MODEL_REVISION"):
        problems.append("env MX_MODEL_REVISION is empty")

    volumes = {v.get("name"): v for v in (pod.get("spec") or {}).get("volumes") or []}
    mounts = {m.get("mountPath"): m.get("name") for m in container.get("volumeMounts") or []}

    if wiring.token_audience:
        if env.get("MX_AUTH_TOKEN_PATH") != TOKEN_PATH:
            problems.append(f"env MX_AUTH_TOKEN_PATH={env.get('MX_AUTH_TOKEN_PATH')!r}, want {TOKEN_PATH!r}")
        volume = volumes.get(mounts.get(posixpath.dirname(TOKEN_PATH)), {})
        audiences = [
            source["serviceAccountToken"].get("audience")
            for source in (volume.get("projected") or {}).get("sources") or []
            if "serviceAccountToken" in source
        ]
        if audiences != [wiring.token_audience]:
            problems.append(
                f"{posixpath.dirname(TOKEN_PATH)} holds projected tokens for {audiences}, "
                f"want [{wiring.token_audience!r}]"
            )

    if wiring.tls and wiring.ca_configmap:
        if env.get("MODEL_EXPRESS_TLS_CA_FILE") != CA_FILE:
            problems.append(f"env MODEL_EXPRESS_TLS_CA_FILE={env.get('MODEL_EXPRESS_TLS_CA_FILE')!r}, want {CA_FILE!r}")
        config_map = volumes.get(mounts.get(posixpath.dirname(CA_FILE)), {}).get("configMap") or {}
        items = [(item.get("key"), item.get("path")) for item in config_map.get("items") or []]
        want = (wiring.ca_configmap, [(wiring.ca_key, posixpath.basename(CA_FILE))])
        if (config_map.get("name"), items) != want:
            problems.append(
                f"{posixpath.dirname(CA_FILE)} mounts ConfigMap {config_map.get('name')!r} items {items}, want {want}"
            )
    return problems


def scc_binding_problems(rolebinding: dict, service_account: str) -> list[str]:
    """Ways the engine's SCC RoleBinding fails to grant the SCC to ``service_account``."""
    problems = []
    role = rolebinding.get("roleRef") or {}
    if (role.get("kind"), role.get("name")) != ("ClusterRole", SCC_CLUSTER_ROLE):
        problems.append(
            f"RoleBinding grants {role.get('kind')}/{role.get('name')}, want ClusterRole/{SCC_CLUSTER_ROLE}"
        )
    subjects = [(s.get("kind"), s.get("name")) for s in rolebinding.get("subjects") or []]
    if ("ServiceAccount", service_account) not in subjects:
        problems.append(f"RoleBinding subjects {subjects} miss ServiceAccount/{service_account}")
    return problems


def loaded_strategies(log: str) -> dict[int, str]:
    """The strategy each worker loaded its weights with, read from an engine log.

    The strategy chain logs every attempt and every failure, so a worker's last
    attempt with no failure after it is the one that loaded.
    """
    current: dict[int, str | None] = {}
    for line in log.splitlines():
        if match := _TRYING_STRATEGY.search(line):
            current[int(match[1])] = match[2]
        elif (match := _FAILED_STRATEGY.search(line)) and current.get(int(match[1])) == match[2]:
            current[int(match[1])] = None
    return {worker: strategy for worker, strategy in current.items() if strategy}


def p2p_transfers(log: str) -> dict[int, str]:
    """Each worker's P2P transfer summary (tensors, bytes, seconds, rate), read from an engine log."""
    return {int(match[1]): match[2] for line in log.splitlines() if (match := _P2P_TRANSFER.search(line))}


def ucx_transports(log: str) -> set[str]:
    """UCX transports the protocol tables printed under ``UCX_PROTO_INFO=y`` select, e.g. ``{"tcp", "cuda_copy"}``.

    Each protocol row has three cells (size range, protocol, lanes); a table's
    title row has two and names memory types, not transports.
    """
    transports: set[str] = set()
    for line in log.splitlines():
        if row := _UCX_TABLE_ROW.match(line):
            cells = row[1].strip("|").split("|")
            if len(cells) == 3:
                transports.update(_UCX_LANE.findall(cells[2]))
    return transports


def parse_probe_output(output: str) -> ProbeResult:
    """The result line ``modelexpress_probe.py`` prints, past any client logging."""
    for line in reversed(output.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            result = json.loads(line)
            return ProbeResult(
                ok=bool(result.get("ok")),
                code=result.get("code", ""),
                detail=result.get("detail", ""),
                worker_ids=tuple(source.get("workerId", "") for source in result.get("sources") or []),
            )
    raise RuntimeError(f"ModelExpress probe printed no result: {output[-500:]}")


def run_probe(
    kubectl: Kubectl,
    pod: str,
    namespace: str,
    *,
    address: str,
    ca_file: str | None = None,
    token_path: str | None = None,
    token: str | None = None,
) -> ProbeResult:
    """Call ListSources from ``pod``'s engine container with the given client settings."""
    request = {"address": address, "caFile": ca_file, "tokenPath": token_path, "token": token}
    output = kubectl(
        "exec",
        "-i",
        pod,
        "-n",
        namespace,
        "-c",
        ENGINE_CONTAINER,
        "--",
        "python3",
        "-c",
        _PROBE_SCRIPT.read_text(),
        input=json.dumps(request),
    )
    return parse_probe_output(output)


def create_token(kubectl: Kubectl, service_account: str, namespace: str, audience: str = "") -> str:
    """A short-lived token for ``service_account``; the API server's audience unless ``audience`` is given."""
    args = ["create", "token", service_account, "-n", namespace, "--duration=10m"]
    if audience:
        args.append(f"--audience={audience}")
    return kubectl(*args).strip()


def tcp_reachable(
    kubectl: Kubectl, pod: str, namespace: str, host: str, port: int, container: str = "", timeout: int = 5
) -> bool:
    """Whether ``pod`` can open a TCP connection to ``host:port`` within ``timeout`` seconds."""
    script = 'if timeout "$2" bash -c "exec 3<>/dev/tcp/$0/$1" 2>/dev/null; then echo reachable; else echo blocked; fi'
    args = ["exec", pod, "-n", namespace]
    if container:
        args += ["-c", container]
    output = kubectl(*args, "--", "bash", "-c", script, host, str(port), str(timeout))
    return output.strip() == "reachable"


def wait_for_pod_ready(kubectl: Kubectl, pod: str, namespace: str, timeout: float, poll_interval: float = 15) -> None:
    """Wait for ``pod`` to become Ready; fail as soon as a container restarts."""
    deadline = time.monotonic() + timeout
    while True:
        resource = json.loads(kubectl("get", "pod", pod, "-n", namespace, "-o", "json"))
        if conditions(resource).get("Ready", {}).get("status") == "True":
            return
        for status in (resource.get("status") or {}).get("containerStatuses") or []:
            if status.get("restartCount", 0) > 0:
                last = (status.get("lastState") or {}).get("terminated") or {}
                raise RuntimeError(
                    f"pod {pod} container {status.get('name')} restarted "
                    f"(exit {last.get('exitCode')}, {last.get('reason')})"
                )
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out after {timeout:g}s waiting for pod {pod} to become Ready")
        time.sleep(poll_interval)


def pod_completion(kubectl: Kubectl, pod: str, namespace: str, base_url: str, model: str, prompt: str) -> dict:
    """A /v1/completions response served by ``pod`` itself, bypassing the gateway."""
    body = json.dumps({"model": model, "prompt": prompt, "max_tokens": 16, "temperature": 0})
    output = kubectl(
        "exec",
        pod,
        "-n",
        namespace,
        "-c",
        ENGINE_CONTAINER,
        "--",
        "curl",
        "-sfk",
        "--max-time",
        "120",
        f"{base_url}/v1/completions",
        "-H",
        "Content-Type: application/json",
        "-d",
        body,
    )
    return json.loads(output)


@contextmanager
def outsider_pod(kubectl: Kubectl, namespace: str, image: str, timeout: int = 240) -> Generator[str, None, None]:
    """A pod running ``image`` in its own namespace, for checks from outside the test namespace.

    The namespace is deleted on exit.
    """
    if not kubectl("get", "namespace", namespace, "--ignore-not-found", "-o", "name"):
        kubectl("create", "namespace", namespace)
    try:
        kubectl("delete", "pod", OUTSIDER_POD, "-n", namespace, "--ignore-not-found")
        container = {
            "name": OUTSIDER_POD,
            "image": image,
            "command": ["sleep", "3600"],
            "securityContext": {"allowPrivilegeEscalation": False, "capabilities": {"drop": ["ALL"]}},
        }
        overrides = {
            "spec": {
                "containers": [container],
                "securityContext": {"runAsNonRoot": True, "seccompProfile": {"type": "RuntimeDefault"}},
            }
        }
        kubectl(
            "run",
            OUTSIDER_POD,
            "-n",
            namespace,
            f"--image={image}",
            "--restart=Never",
            f"--overrides={json.dumps(overrides)}",
        )
        kubectl("wait", f"pod/{OUTSIDER_POD}", "-n", namespace, "--for=condition=Ready", f"--timeout={timeout}s")
        yield OUTSIDER_POD
    finally:
        kubectl("delete", "namespace", namespace, "--ignore-not-found", "--wait=false")
