"""Unit tests for the LLMBatchGateway async dispatch rewrite — no cluster required.

Run with: ``uv run pytest tests/batch/test_batch_dispatch.py -v``

The deploy time rewrite exists because the dispatch URL is the one manifest field
that does not port across platforms: Istio's gateway serves plain HTTP on 80,
RHOAI 3.6's AIGateway only TLS on 443, and an async processor pointed at the
wrong host or port leaves batches ``in_progress`` forever without an obvious
error. These tests drive the pure helpers (``dispatch_target`` /
``patch_batch_dispatch``) and ``Deployer._patch_manifest`` with a fake gateway
Service, so the derived URL, the TLS override, multiple SQL queues and the
``validation.batchGateway.dispatchUrl`` override are all checked without kubectl.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from conformance.batch import (
    DISPATCH_QUEUE_URL_PATH,
    DISPATCH_URL_PATH,
    dispatch_target,
    patch_batch_dispatch,
)
from conformance.config import load_testcase
from conformance.deployer import Deployer

_NAMESPACE = "llm-conformance-test"
_NAME = "batch-gateway-async-sql"
_HTTP_URL = "http://gw.ns.svc.cluster.local:80/ns/model"
_HTTPS_URL = "https://gw.ns.svc.cluster.local:443/ns/model"

_LLMISVC = {
    "apiVersion": "serving.kserve.io/v1alpha2",
    "kind": "LLMInferenceService",
    "metadata": {"name": "manifest-name"},
    "spec": {"model": {"name": "Qwen/Qwen3-0.6B", "uri": "hf://Qwen/Qwen3-0.6B"}},
}


def _batch_gateway(url: bool = True, queue_urls: tuple[bool, ...] = (), tls: bool = False) -> dict:
    """Minimal ``LLMBatchGateway`` document shaped like the manifests-repo skeleton."""
    async_config: dict = {"concurrency": 8}
    if url:
        async_config["inferenceGateway"] = {"url": "http://inference-gateway-istio.redhat-ods-applications:80/x/y"}
    if queue_urls:
        queues = []
        for index, with_url in enumerate(queue_urls):
            queue: dict = {"name": f"llm-d-async:requests:pool-{index}", "workerPoolID": "default"}
            if with_url:
                queue["igwBaseURL"] = "http://stale.example.com:80/x/y"
            queues.append(queue)
        async_config["sql"] = {"queuesConfig": queues}
    if tls:
        async_config["tls"] = {"caCertKey": "ca.crt"}
    return {
        "apiVersion": "batch.llm-d.ai/v1alpha1",
        "kind": "LLMBatchGateway",
        "metadata": {"name": _NAME},
        "spec": {"processor": {"asyncConfig": async_config}},
    }


def _async_config(document: dict) -> dict:
    return document["spec"]["processor"]["asyncConfig"]


def _batch_case():
    tc = load_testcase("configs/testcases/batch-gateway-async-sql.yaml")
    tc.name = _NAME
    return tc


def _patch(tmp_path: Path, tc, *documents: dict) -> Path:
    """Write an inline manifest and return its path (no cluster, no kubectl)."""
    path = tmp_path / tc.deployment.manifest_path
    path.write_text("---\n".join(yaml.dump(document) for document in documents))
    return path


def test_dispatch_target_uses_http_on_port_80():
    url, insecure_tls = dispatch_target("svc/inference-gateway-istio", "redhat-ods-applications", 80, _NAMESPACE, _NAME)

    assert url == f"http://inference-gateway-istio.redhat-ods-applications.svc.cluster.local:80/{_NAMESPACE}/{_NAME}"
    assert insecure_tls is False, "a plain HTTP gateway needs no TLS override"


def test_dispatch_target_uses_https_and_skips_tls_verification_on_443():
    """RHOAI 3.6's AIGateway is TLS-only and its certificate is signed by the OpenShift service CA."""
    url, insecure_tls = dispatch_target("svc/data-science-gateway", "openshift-ingress", 443, _NAMESPACE, _NAME)

    assert url == f"https://data-science-gateway.openshift-ingress.svc.cluster.local:443/{_NAMESPACE}/{_NAME}"
    assert insecure_tls is True


def test_dispatch_target_accepts_a_service_name_without_the_svc_prefix():
    url, _ = dispatch_target("inference-gateway-istio", "redhat-ods-applications", 80, _NAMESPACE, _NAME)

    assert url.startswith("http://inference-gateway-istio.redhat-ods-applications.svc.cluster.local:80/")


def test_patch_batch_dispatch_rewrites_the_gateway_and_every_sql_queue():
    document = _batch_gateway(queue_urls=(True, False, True))

    rewritten = patch_batch_dispatch([document], url=_HTTP_URL)

    async_config = _async_config(document)
    assert async_config["inferenceGateway"]["url"] == _HTTP_URL
    queues = async_config["sql"]["queuesConfig"]
    assert [queue.get("igwBaseURL") for queue in queues] == [_HTTP_URL, None, _HTTP_URL]
    assert rewritten == [DISPATCH_URL_PATH, DISPATCH_QUEUE_URL_PATH % 0, DISPATCH_QUEUE_URL_PATH % 2]


def test_patch_batch_dispatch_does_not_invent_a_queue_url():
    """A queue without ``igwBaseURL`` inherits the gateway URL; adding the key would change semantics."""
    document = _batch_gateway(queue_urls=(False,))

    patch_batch_dispatch([document], url=_HTTP_URL)

    assert "igwBaseURL" not in _async_config(document)["sql"]["queuesConfig"][0]


def test_patch_batch_dispatch_forces_insecure_tls_only_for_a_tls_gateway():
    secure = _batch_gateway(tls=True)
    patch_batch_dispatch([secure], url=_HTTP_URL)
    assert _async_config(secure)["tls"] == {"caCertKey": "ca.crt"}, "HTTP must not touch the manifest's TLS block"

    insecure = _batch_gateway(tls=True)
    patch_batch_dispatch([insecure], url=_HTTPS_URL, insecure_tls=True)
    assert _async_config(insecure)["tls"] == {"caCertKey": "ca.crt", "insecureSkipVerify": True}

    created = _batch_gateway()
    patch_batch_dispatch([created], url=_HTTPS_URL, insecure_tls=True)
    assert _async_config(created)["tls"] == {"insecureSkipVerify": True}


def test_patch_batch_dispatch_ignores_manifests_without_an_async_config():
    other_document = {"kind": "ConfigMap", "metadata": {"name": "unrelated"}, "data": {"url": "http://keep:80"}}
    sync = {"kind": "LLMBatchGateway", "spec": {"processor": {"modelGateways": {}}}}

    rewritten = patch_batch_dispatch([other_document, sync], url=_HTTP_URL)

    assert rewritten == [], "nothing to dispatch through means nothing to rewrite"
    assert other_document["data"]["url"] == "http://keep:80"
    assert "asyncConfig" not in sync["spec"]["processor"]


def test_deployer_derives_the_dispatch_url_from_the_discovered_gateway(tmp_path, monkeypatch):
    tc = _batch_case()
    deployer = Deployer(manifest_dir=str(tmp_path))
    monkeypatch.setattr(deployer, "_gateway_service", lambda: ("svc/data-science-gateway", "openshift-ingress", 443))

    documents = deployer._patch_manifest(_patch(tmp_path, tc, _LLMISVC, _batch_gateway(tls=True)), tc)

    async_config = _async_config(documents[1])
    assert async_config["inferenceGateway"]["url"] == (
        f"https://data-science-gateway.openshift-ingress.svc.cluster.local:443/{_NAMESPACE}/{_NAME}"
    )
    assert async_config["tls"]["insecureSkipVerify"] is True


def test_deployer_uses_the_http_form_when_the_gateway_serves_port_80(tmp_path, monkeypatch):
    tc = _batch_case()
    deployer = Deployer(manifest_dir=str(tmp_path))
    monkeypatch.setattr(
        deployer, "_gateway_service", lambda: ("svc/inference-gateway-istio", "redhat-ods-applications", 80)
    )

    documents = deployer._patch_manifest(_patch(tmp_path, tc, _LLMISVC, _batch_gateway()), tc)

    async_config = _async_config(documents[1])
    assert async_config["inferenceGateway"]["url"] == (
        f"http://inference-gateway-istio.redhat-ods-applications.svc.cluster.local:80/{_NAMESPACE}/{_NAME}"
    )
    assert "tls" not in async_config


def test_deployer_dispatch_url_override_is_verbatim_and_skips_discovery(tmp_path, monkeypatch):
    tc = _batch_case()
    tc.validation.batch_gateway.dispatch_url = "https://edge.example.com/team/model"

    def no_discovery():
        raise AssertionError("an explicit dispatchUrl must not trigger gateway discovery")

    deployer = Deployer(manifest_dir=str(tmp_path))
    monkeypatch.setattr(deployer, "_gateway_service", no_discovery)

    documents = deployer._patch_manifest(
        _patch(tmp_path, tc, _LLMISVC, _batch_gateway(queue_urls=(True,), tls=True)),
        tc,
    )

    async_config = _async_config(documents[1])
    assert async_config["inferenceGateway"]["url"] == "https://edge.example.com/team/model"
    assert async_config["sql"]["queuesConfig"][0]["igwBaseURL"] == "https://edge.example.com/team/model"
    assert async_config["tls"] == {"caCertKey": "ca.crt"}, "the manifest's own TLS block is left untouched"


def test_deployer_does_not_look_up_a_gateway_for_a_non_batch_manifest(tmp_path, monkeypatch):
    tc = load_testcase("configs/testcases/single-gpu-smoke.yaml")

    def no_discovery():
        raise AssertionError("a manifest without an LLMBatchGateway must not discover the gateway")

    deployer = Deployer(manifest_dir=str(tmp_path))
    monkeypatch.setattr(deployer, "_gateway_service", no_discovery)

    documents = deployer._patch_manifest(_patch(tmp_path, tc, _LLMISVC), tc)

    assert [doc["kind"] for doc in documents] == ["LLMInferenceService"]


def test_dispatch_prefers_the_gateway_kserve_routes_to(tmp_path, monkeypatch):
    """RHOAI 3.6-EA2 runs two Gateways; the first labelled Service is the OAuth-protected one."""

    def kubectl(self, *args, check=True):
        if args[:2] == ("get", "cm"):
            return '{"kserveIngressGateway": "openshift-ingress/openshift-ai-inference"}'
        if args[:3] == ("get", "svc", "-A"):
            return "data-science-gateway-data-science-gateway-class/openshift-ingress\n"
        if args[:2] == ("get", "svc") and "-l" in args:
            return "openshift-ai-inference-data-science-gateway-class"
        if args[:2] == ("get", "svc") and "ports" in args[-1]:
            return "15021 80 "
        raise AssertionError(f"unexpected kubectl call: {args}")

    tc = _batch_case()
    deployer = Deployer(manifest_dir=str(tmp_path))
    monkeypatch.setattr(Deployer, "kubectl", kubectl)

    documents = deployer._patch_manifest(_patch(tmp_path, tc, _LLMISVC, _batch_gateway()), tc)

    assert _async_config(documents[1])["inferenceGateway"]["url"] == (
        "http://openshift-ai-inference-data-science-gateway-class.openshift-ingress.svc.cluster.local:80"
        f"/{_NAMESPACE}/{_NAME}"
    ), "dispatch must target the Gateway the model's HTTPRoute attaches to, not the first labelled one"


def test_dispatch_falls_back_to_label_discovery_without_an_ingress_config(tmp_path, monkeypatch):
    tc = _batch_case()
    deployer = Deployer(manifest_dir=str(tmp_path))
    monkeypatch.setattr(deployer, "kubectl", lambda *a, check=True: "")
    monkeypatch.setattr(deployer, "_gateway_service", lambda: ("svc/gw", "ns", 80))

    documents = deployer._patch_manifest(_patch(tmp_path, tc, _LLMISVC, _batch_gateway()), tc)

    assert (
        _async_config(documents[1])["inferenceGateway"]["url"]
        == f"http://gw.ns.svc.cluster.local:80/{_NAMESPACE}/{_NAME}"
    )


def test_dispatch_falls_back_when_the_configured_gateway_has_no_service(tmp_path, monkeypatch):
    def kubectl(self, *args, check=True):
        if args[:2] == ("get", "cm"):
            return '{"kserveIngressGateway": "openshift-ingress/openshift-ai-inference"}'
        if args[:2] == ("get", "svc") and "-l" in args:
            return ""
        raise AssertionError(f"unexpected kubectl call: {args}")

    tc = _batch_case()
    deployer = Deployer(manifest_dir=str(tmp_path))
    monkeypatch.setattr(Deployer, "kubectl", kubectl)
    monkeypatch.setattr(deployer, "_gateway_service", lambda: ("svc/data-science-gateway", "openshift-ingress", 443))

    documents = deployer._patch_manifest(_patch(tmp_path, tc, _LLMISVC, _batch_gateway()), tc)

    async_config = _async_config(documents[1])
    assert async_config["inferenceGateway"]["url"] == (
        f"https://data-science-gateway.openshift-ingress.svc.cluster.local:443/{_NAMESPACE}/{_NAME}"
    )
    assert async_config["tls"]["insecureSkipVerify"] is True
