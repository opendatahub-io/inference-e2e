"""Unit tests for resolving and reaching the Gateway an LLMInferenceService is routed through.

Objects follow the shapes KServe, the Gateway API, and Istio produce: the
``<name>-kserve-route`` HTTPRoute with its ``parentRefs``, Gateways with
``Programmed`` conditions and listeners, and the gateway Service labeled
``gateway.networking.k8s.io/gateway-name``.

Run with: ``uv run pytest tests/test_gateway.py -v``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from conformance.deployer import (
    GATEWAY_NAME_LABEL,
    Deployer,
    GatewayRef,
    gateway_address,
    gateway_http_port,
    route_gateway,
)

NAMESPACE = "llm-conformance-test"
SERVICE = "single-gpu-smoke"
HTTP_LISTENER = {"name": "http", "port": 80, "protocol": "HTTP", "allowedRoutes": {"namespaces": {"from": "All"}}}
HTTPS_LISTENER = {"name": "https", "port": 443, "protocol": "HTTPS", "tls": {"mode": "Terminate"}}
OCP_SERVICE = "openshift-ai-inference-openshift-default"
OCP_ADDRESS = f"{OCP_SERVICE}.openshift-ingress.svc.cluster.local"


def _route(parent_refs: list[dict], namespace: str = NAMESPACE) -> dict:
    return {
        "apiVersion": "gateway.networking.k8s.io/v1",
        "kind": "HTTPRoute",
        "metadata": {"name": f"{SERVICE}-kserve-route", "namespace": namespace},
        "spec": {
            "parentRefs": parent_refs,
            "rules": [
                {
                    "backendRefs": [
                        {"group": "inference.networking.k8s.io", "kind": "InferencePool", "name": f"{SERVICE}-pool"}
                    ],
                    "matches": [{"path": {"type": "PathPrefix", "value": f"/{namespace}/{SERVICE}/v1/completions"}}],
                }
            ],
        },
    }


def _gateway_parent(name: str, namespace: str, **extra: str) -> dict:
    return {"group": "gateway.networking.k8s.io", "kind": "Gateway", "name": name, "namespace": namespace, **extra}


def _gateway(
    namespace: str, name: str, *, programmed: bool, service: str = "", listeners: list[dict] | None = None
) -> dict:
    hostname = f"{service or f'{name}-istio'}.{namespace}.svc.cluster.local"
    if programmed:
        programmed_condition = {
            "type": "Programmed",
            "status": "True",
            "reason": "Programmed",
            "message": f"Resource programmed, assigned to service(s) {hostname}:80",
        }
        addresses = [{"type": "Hostname", "value": hostname}]
    else:
        programmed_condition = {
            "type": "Programmed",
            "status": "False",
            "reason": "AddressNotAssigned",
            "message": f"Assigned to service(s) {hostname}:80, but failed to assign to all requested addresses: "
            f'address pending for hostname "{hostname}"',
        }
        addresses = []
    return {
        "apiVersion": "gateway.networking.k8s.io/v1",
        "kind": "Gateway",
        "metadata": {"name": name, "namespace": namespace},
        "spec": {"gatewayClassName": "istio", "listeners": listeners or [HTTP_LISTENER]},
        "status": {
            "addresses": addresses,
            "conditions": [
                {"type": "Accepted", "status": "True", "reason": "Accepted", "message": "Resource accepted"},
                programmed_condition,
            ],
        },
    }


def _service(namespace: str, gateway: str, name: str) -> dict:
    return {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": name, "namespace": namespace, "labels": {GATEWAY_NAME_LABEL: gateway}},
    }


class FakeCluster:
    """Answers the ``kubectl get`` calls gateway discovery makes; any other call fails the test."""

    def __init__(self, *objects: dict):
        self.objects: dict[tuple[str, str, str], dict] = {}
        for obj in objects:
            self.add(obj)

    def add(self, obj: dict) -> None:
        metadata = obj["metadata"]
        self.objects[(obj["kind"].lower(), metadata["namespace"], metadata["name"])] = obj

    def kubectl(self, *args: str, check: bool = True) -> str:
        match args:
            case ("get", kind, name, "-n", namespace, "-o", "json"):
                obj = self.objects.get((kind, namespace, name))
                if obj is None:
                    raise RuntimeError(f'Error from server (NotFound): {kind}s "{name}" not found')
                return json.dumps(obj)
            case ("get", "svc", "-n", namespace, "-l", selector, "-o", "jsonpath={.items[*].metadata.name}"):
                key, value = selector.split("=", 1)
                return " ".join(
                    name
                    for (kind, ns, name), obj in self.objects.items()
                    if kind == "service" and ns == namespace and obj["metadata"]["labels"].get(key) == value
                )
            case ("get", "llminferenceservice", name, "-n", namespace, "-o", "jsonpath={.status.url}"):
                obj = self.objects.get(("llminferenceservice", namespace, name))
                return obj["status"]["url"] if obj else ""
        raise AssertionError(f"unexpected kubectl call: {args}")


def _llmisvc(url: str) -> dict:
    return {
        "apiVersion": "serving.kserve.io/v1alpha2",
        "kind": "LLMInferenceService",
        "metadata": {"name": SERVICE, "namespace": NAMESPACE},
        "status": {"url": url},
    }


def _openshift_cluster() -> FakeCluster:
    """OpenShift RHOAI: default gateway in openshift-ingress, plus another team's unprogrammed Gateway."""
    return FakeCluster(
        _gateway("aaa-team", "team-gateway", programmed=False, service="team-gateway-openshift-default"),
        _service("aaa-team", "team-gateway", "team-gateway-openshift-default"),
        _gateway("openshift-ingress", "openshift-ai-inference", programmed=True, service=OCP_SERVICE),
        _service("openshift-ingress", "openshift-ai-inference", OCP_SERVICE),
        _route([_gateway_parent("openshift-ai-inference", "openshift-ingress")]),
        _llmisvc(f"http://{OCP_ADDRESS}/{NAMESPACE}/{SERVICE}"),
    )


def _deployer(monkeypatch, cluster: FakeCluster) -> Deployer:
    deployer = Deployer(namespace=NAMESPACE)
    monkeypatch.setattr(deployer, "kubectl", cluster.kubectl)
    return deployer


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = 0

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds
        self.sleeps += 1


@pytest.fixture
def clock(monkeypatch) -> FakeClock:
    c = FakeClock()
    monkeypatch.setattr("conformance.deployer.time.time", c.time)
    monkeypatch.setattr("conformance.deployer.time.sleep", c.sleep)
    return c


class FakeProcess:
    def __init__(self, cmd: list[str], exits_immediately: bool = False):
        self.cmd = cmd
        self.returncode = 1 if exits_immediately else None
        self.terminated = False

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    def wait(self, timeout: float | None = None) -> int | None:
        return self.returncode


@pytest.fixture
def port_forwards(monkeypatch) -> list[FakeProcess]:
    started: list[FakeProcess] = []

    def popen(cmd, **kwargs):
        process = FakeProcess(cmd)
        started.append(process)
        return process

    monkeypatch.setattr("conformance.deployer.subprocess.Popen", popen)
    monkeypatch.setattr("conformance.deployer.time.sleep", lambda _: None)
    return started


# --- route_gateway -----------------------------------------------------------


def test_route_gateway_reads_explicit_namespace_and_section():
    route = _route([_gateway_parent("openshift-ai-inference", "openshift-ingress", sectionName="http")])
    assert route_gateway(route) == GatewayRef("openshift-ingress", "openshift-ai-inference", "http")


def test_route_gateway_defaults_namespace_group_and_kind():
    """Gateway API defaults an omitted namespace to the route's, group and kind to Gateway."""
    assert route_gateway(_route([{"name": "local-gateway"}])) == GatewayRef(NAMESPACE, "local-gateway")


def test_route_gateway_takes_first_gateway_in_parent_ref_order():
    """MaaS services list inference-gateway before maas-default-gateway; inference uses the first."""
    route = _route(
        [
            _gateway_parent("inference-gateway", "redhat-ods-applications"),
            _gateway_parent("maas-default-gateway", "redhat-ods-applications"),
        ]
    )
    assert route_gateway(route) == GatewayRef("redhat-ods-applications", "inference-gateway")


def test_route_gateway_skips_parent_refs_that_are_not_gateways():
    route = _route(
        [
            {"group": "", "kind": "Service", "name": "mesh-service"},
            {"group": "example.com", "kind": "Gateway", "name": "foreign-gateway"},
            _gateway_parent("inference-gateway", "redhat-ods-applications"),
        ]
    )
    assert route_gateway(route) == GatewayRef("redhat-ods-applications", "inference-gateway")


@pytest.mark.parametrize("spec", [{}, {"parentRefs": []}, {"parentRefs": [{"kind": "Service", "name": "svc"}]}])
def test_route_gateway_none_without_gateway_parent(spec):
    route = {"metadata": {"name": "r", "namespace": NAMESPACE}, "spec": spec}
    assert route_gateway(route) is None


# --- gateway_address ---------------------------------------------------------


def test_gateway_address_of_programmed_gateway():
    gateway = _gateway("openshift-ingress", "openshift-ai-inference", programmed=True, service=OCP_SERVICE)
    assert gateway_address(gateway) == OCP_ADDRESS


def test_gateway_address_empty_while_not_programmed():
    assert gateway_address(_gateway("aaa-team", "team-gateway", programmed=False)) == ""


def test_gateway_address_ignores_stale_address_on_unprogrammed_gateway():
    gateway = _gateway("aaa-team", "team-gateway", programmed=False)
    gateway["status"]["addresses"] = [{"type": "IPAddress", "value": "10.0.0.1"}]
    assert gateway_address(gateway) == ""


def test_gateway_address_empty_when_programmed_without_address():
    gateway = _gateway("openshift-ingress", "openshift-ai-inference", programmed=True)
    gateway["status"]["addresses"] = []
    assert gateway_address(gateway) == ""


@pytest.mark.parametrize("gateway", [{}, {"status": {}}, {"status": {"conditions": [], "addresses": None}}])
def test_gateway_address_empty_without_status(gateway):
    assert gateway_address(gateway) == ""


# --- gateway_http_port -------------------------------------------------------


def test_gateway_http_port_picks_http_listener_over_https():
    """The xKS inference-gateway lists http:80 and, with TLS, https:443; the client speaks HTTP."""
    gateway = _gateway("redhat-ods-applications", "inference-gateway", programmed=True)
    gateway["spec"]["listeners"] = [HTTPS_LISTENER, HTTP_LISTENER]
    assert gateway_http_port(gateway) == 80


def test_gateway_http_port_honors_section_name():
    gateway = _gateway("ns", "gw", programmed=True)
    gateway["spec"]["listeners"] = [HTTP_LISTENER, {"name": "alt", "port": 8080, "protocol": "HTTP"}]
    assert gateway_http_port(gateway, "alt") == 8080


def test_gateway_http_port_rejects_section_on_https_listener():
    gateway = _gateway("ns", "gw", programmed=True)
    gateway["spec"]["listeners"] = [HTTP_LISTENER, HTTPS_LISTENER]
    with pytest.raises(RuntimeError, match="Gateway gw has no HTTP listener 'https'"):
        gateway_http_port(gateway, "https")


def test_gateway_http_port_rejects_gateway_without_http_listener():
    gateway = _gateway("openshift-ingress", "data-science-gateway", programmed=True)
    gateway["spec"]["listeners"] = [HTTPS_LISTENER]
    with pytest.raises(RuntimeError, match="Gateway data-science-gateway has no HTTP listener$"):
        gateway_http_port(gateway)


# --- Deployer.wait_for_gateway -----------------------------------------------


def test_wait_for_gateway_uses_the_services_gateway_not_the_first_listed(monkeypatch, clock):
    """Another namespace's unprogrammed Gateway must not block or answer for this service."""
    deployer = _deployer(monkeypatch, _openshift_cluster())
    assert deployer.wait_for_gateway(SERVICE) == OCP_ADDRESS
    assert clock.sleeps == 0


def test_wait_for_gateway_resolves_explicit_gateway_refs(monkeypatch, clock):
    cluster = FakeCluster(
        _gateway("redhat-ods-applications", "inference-gateway", programmed=True),
        _gateway("redhat-ods-applications", "maas-default-gateway", programmed=False),
        _route(
            [
                _gateway_parent("inference-gateway", "redhat-ods-applications"),
                _gateway_parent("maas-default-gateway", "redhat-ods-applications"),
            ]
        ),
    )
    deployer = _deployer(monkeypatch, cluster)
    assert deployer.wait_for_gateway(SERVICE) == "inference-gateway-istio.redhat-ods-applications.svc.cluster.local"


def test_wait_for_gateway_polls_until_route_exists_and_gateway_is_programmed(monkeypatch, clock):
    cluster = FakeCluster(
        _gateway("openshift-ingress", "openshift-ai-inference", programmed=False, service=OCP_SERVICE)
    )
    deployer = _deployer(monkeypatch, cluster)
    polls = 0
    kubectl = cluster.kubectl

    def kubectl_as_controllers_progress(*args, **kwargs):
        nonlocal polls
        if args[:2] == ("get", "httproute"):
            polls += 1
            if polls == 2:
                cluster.add(_route([_gateway_parent("openshift-ai-inference", "openshift-ingress")]))
            if polls == 3:
                cluster.add(
                    _gateway("openshift-ingress", "openshift-ai-inference", programmed=True, service=OCP_SERVICE)
                )
        return kubectl(*args, **kwargs)

    monkeypatch.setattr(deployer, "kubectl", kubectl_as_controllers_progress)
    assert deployer.wait_for_gateway(SERVICE) == OCP_ADDRESS
    assert polls == 3
    assert clock.sleeps == 2


def test_wait_for_gateway_timeout_reports_unprogrammed_gateway(monkeypatch, clock):
    cluster = FakeCluster(
        _gateway("openshift-ingress", "openshift-ai-inference", programmed=False, service=OCP_SERVICE),
        _route([_gateway_parent("openshift-ai-inference", "openshift-ingress")]),
    )
    deployer = _deployer(monkeypatch, cluster)
    with pytest.raises(TimeoutError) as excinfo:
        deployer.wait_for_gateway(SERVICE, timeout=30)
    message = str(excinfo.value)
    assert message.startswith(f"Gateway for {SERVICE} not programmed after 30s")
    assert "Gateway openshift-ingress/openshift-ai-inference Programmed=False" in message
    assert "address pending" in message
    assert clock.sleeps == 3


def test_wait_for_gateway_timeout_reports_missing_route(monkeypatch, clock):
    deployer = _deployer(monkeypatch, FakeCluster())
    with pytest.raises(TimeoutError, match=f'httproutes "{SERVICE}-kserve-route" not found'):
        deployer.wait_for_gateway(SERVICE, timeout=10)


def test_wait_for_gateway_timeout_reports_route_without_gateway(monkeypatch, clock):
    deployer = _deployer(monkeypatch, FakeCluster(_route([])))
    with pytest.raises(TimeoutError, match=f"HTTPRoute {SERVICE}-kserve-route has no Gateway parentRef"):
        deployer.wait_for_gateway(SERVICE, timeout=10)


def test_wait_for_gateway_timeout_reports_missing_gateway(monkeypatch, clock):
    deployer = _deployer(monkeypatch, FakeCluster(_route([_gateway_parent("gone", "openshift-ingress")])))
    with pytest.raises(TimeoutError, match='gateways "gone" not found'):
        deployer.wait_for_gateway(SERVICE, timeout=10)


# --- Deployer.get_endpoint / port-forward ------------------------------------


def test_get_endpoint_port_forwards_to_the_services_gateway_on_openshift(monkeypatch, port_forwards):
    deployer = _deployer(monkeypatch, _openshift_cluster())
    deployer.kubeconfig = "/tmp/kubeconfig"

    url = deployer.get_endpoint(SERVICE)

    [forward] = port_forwards
    local_port = deployer._port_forward_port
    assert forward.cmd == [
        "kubectl",
        "--kubeconfig",
        "/tmp/kubeconfig",
        "port-forward",
        "-n",
        "openshift-ingress",
        f"svc/{OCP_SERVICE}",
        f"{local_port}:80",
    ]
    assert url == f"http://localhost:{local_port}/{NAMESPACE}/{SERVICE}"


def test_get_endpoint_port_forwards_to_inference_gateway_on_xks(monkeypatch, port_forwards):
    cluster = FakeCluster(
        _gateway(
            "redhat-ods-applications", "inference-gateway", programmed=True, listeners=[HTTP_LISTENER, HTTPS_LISTENER]
        ),
        _service("redhat-ods-applications", "inference-gateway", "inference-gateway-istio"),
        _route([_gateway_parent("inference-gateway", "redhat-ods-applications")]),
        _llmisvc(f"http://inference-gateway-istio.redhat-ods-applications.svc.cluster.local/{NAMESPACE}/{SERVICE}"),
    )
    deployer = _deployer(monkeypatch, cluster)

    url = deployer.get_endpoint(SERVICE)

    [forward] = port_forwards
    assert forward.cmd[-3:] == [
        "redhat-ods-applications",
        "svc/inference-gateway-istio",
        f"{deployer._port_forward_port}:80",
    ]
    assert url == f"http://localhost:{deployer._port_forward_port}/{NAMESPACE}/{SERVICE}"


def test_get_endpoint_forwards_to_the_listener_named_by_section(monkeypatch, port_forwards):
    alt = {"name": "alt", "port": 8080, "protocol": "HTTP"}
    cluster = FakeCluster(
        _gateway("gw-ns", "gw", programmed=True, listeners=[HTTP_LISTENER, alt]),
        _service("gw-ns", "gw", "gw-istio"),
        _route([_gateway_parent("gw", "gw-ns", sectionName="alt")]),
        _llmisvc(f"http://gw-istio.gw-ns.svc.cluster.local:8080/{NAMESPACE}/{SERVICE}"),
    )
    deployer = _deployer(monkeypatch, cluster)

    deployer.get_endpoint(SERVICE)

    assert port_forwards[0].cmd[-1] == f"{deployer._port_forward_port}:8080"


def test_get_endpoint_defaults_path_when_status_url_has_none(monkeypatch, port_forwards):
    cluster = _openshift_cluster()
    cluster.add(_llmisvc(f"http://{OCP_ADDRESS}"))
    deployer = _deployer(monkeypatch, cluster)

    assert deployer.get_endpoint(SERVICE) == f"http://localhost:{deployer._port_forward_port}/{NAMESPACE}/{SERVICE}"


def test_get_endpoint_reuses_port_forward_for_the_same_gateway(monkeypatch, port_forwards):
    deployer = _deployer(monkeypatch, _openshift_cluster())

    first = deployer.get_endpoint(SERVICE)
    second = deployer.get_endpoint(SERVICE)

    assert first == second
    assert len(port_forwards) == 1
    assert not port_forwards[0].terminated


def test_get_endpoint_restarts_port_forward_when_the_gateway_changes(monkeypatch, port_forwards):
    cluster = _openshift_cluster()
    cluster.add(_gateway("other-ns", "other-gateway", programmed=True))
    cluster.add(_service("other-ns", "other-gateway", "other-gateway-istio"))
    deployer = _deployer(monkeypatch, cluster)

    deployer.get_endpoint(SERVICE)
    cluster.add(_route([_gateway_parent("other-gateway", "other-ns")]))
    deployer.get_endpoint(SERVICE)

    first, second = port_forwards
    assert first.terminated
    assert second.cmd[-3:-1] == ["other-ns", "svc/other-gateway-istio"]
    assert not second.terminated


def test_get_endpoint_falls_back_to_status_url_when_port_forward_exits(monkeypatch):
    deployer = _deployer(monkeypatch, _openshift_cluster())
    monkeypatch.setattr(
        "conformance.deployer.subprocess.Popen", lambda cmd, **kwargs: FakeProcess(cmd, exits_immediately=True)
    )
    monkeypatch.setattr("conformance.deployer.time.sleep", lambda _: None)

    url = deployer.get_endpoint(SERVICE)

    assert url == f"http://{OCP_ADDRESS}/{NAMESPACE}/{SERVICE}"
    assert deployer._port_forward_proc is None
    assert deployer._port_forward_target is None


def test_get_endpoint_fails_when_gateway_has_no_service(monkeypatch, port_forwards):
    cluster = _openshift_cluster()
    del cluster.objects[("service", "openshift-ingress", OCP_SERVICE)]
    deployer = _deployer(monkeypatch, cluster)

    with pytest.raises(RuntimeError, match=f"No Service labeled {GATEWAY_NAME_LABEL}=openshift-ai-inference"):
        deployer.get_endpoint(SERVICE)
    assert port_forwards == []


def test_stop_port_forward_terminates_the_gateway_forward(monkeypatch, port_forwards):
    deployer = _deployer(monkeypatch, _openshift_cluster())
    deployer.get_endpoint(SERVICE)

    deployer.stop_port_forward()

    assert port_forwards[0].terminated
    assert deployer._port_forward_proc is None
    assert deployer._port_forward_target is None
