from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from conformance.modelexpress import (
    ADDRESS_ANNOTATION,
    EngineWiring,
    ServerRef,
    engine_capabilities,
    engine_wiring_problems,
    loaded_strategies,
    manifest_servers,
    network_policy_problems,
    parse_probe_output,
    patch_modelexpress_refs,
    scc_binding_problems,
)

_FIXTURES = Path(__file__).parent / "fixtures"

_MANIFEST = """
apiVersion: modelexpress.opendatahub.io/v1alpha1
kind: ModelExpressServer
metadata:
  name: modelexpress
spec:
  security:
    allowedServiceAccounts:
    - namespace: llm-conformance-test
      serviceAccount: modelexpress-p2p-kserve
---
apiVersion: serving.kserve.io/v1alpha2
kind: LLMInferenceService
metadata:
  name: modelexpress-p2p
  annotations:
    serving.kserve.io/exp-modelexpress-address: https://modelexpress.llm-conformance-test.svc.cluster.local:8001
"""


def _fixture(name: str) -> dict:
    return json.loads((_FIXTURES / name).read_text())


def _documents(text: str = _MANIFEST) -> list[dict]:
    return list(yaml.safe_load_all(text))


def test_patch_follows_the_test_namespace_and_renamed_service():
    documents = _documents()

    patch_modelexpress_refs(documents, "modelexpress-p2p", "mx-run", "team-a")

    server, llmisvc = documents
    assert server["spec"]["security"]["allowedServiceAccounts"] == [
        {"namespace": "team-a", "serviceAccount": "mx-run-kserve"}
    ]
    assert (
        llmisvc["metadata"]["annotations"][ADDRESS_ANNOTATION] == "https://modelexpress.team-a.svc.cluster.local:8001"
    )


def test_patch_keeps_an_address_that_names_no_local_server():
    documents = _documents(_MANIFEST.replace("name: modelexpress\n", "name: modelexpress\n  namespace: shared-mx\n", 1))

    patch_modelexpress_refs(documents, "modelexpress-p2p", "mx-run", "team-a")

    assert (
        documents[1]["metadata"]["annotations"][ADDRESS_ANNOTATION]
        == "https://modelexpress.llm-conformance-test.svc.cluster.local:8001"
    )


def test_patch_handles_a_bare_host_port_address():
    documents = _documents(_MANIFEST.replace("https://modelexpress.", "modelexpress."))

    patch_modelexpress_refs(documents, "modelexpress-p2p", "mx-run", "team-a")

    assert documents[1]["metadata"]["annotations"][ADDRESS_ANNOTATION] == "modelexpress.team-a.svc.cluster.local:8001"


def test_manifest_servers_default_to_the_test_namespace(tmp_path):
    manifest = tmp_path / "mx.yaml"
    manifest.write_text(_MANIFEST)

    assert manifest_servers(manifest, "team-a") == [ServerRef("modelexpress", "team-a")]
    assert manifest_servers(tmp_path / "missing.yaml", "team-a") == []


def test_wiring_reads_annotations_and_defaults_the_ca_key():
    llmisvc = _fixture("llmisvc.json")

    wiring = EngineWiring.from_service(llmisvc)

    assert wiring.tls
    assert wiring.grpc_port == 8001
    assert wiring.token_audience == "modelexpress"
    assert (wiring.ca_configmap, wiring.ca_key) == ("openshift-service-ca.crt", "service-ca.crt")
    llmisvc["metadata"]["annotations"].pop("serving.kserve.io/exp-modelexpress-ca-bundle-key")
    assert EngineWiring.from_service(llmisvc).ca_key == "cabundle.crt"


def test_rendered_engine_pod_matches_its_annotations():
    pod = _fixture("engine-pod.json")
    wiring = EngineWiring.from_service(_fixture("llmisvc.json"))

    assert engine_wiring_problems(pod, wiring) == []
    assert engine_capabilities(pod) == ["IPC_LOCK"]


def test_engine_pod_without_the_ca_bundle_is_reported():
    pod = _fixture("engine-pod.json")
    wiring = EngineWiring.from_service(_fixture("llmisvc.json"))
    container = next(c for c in pod["spec"]["containers"] if c["name"] == "main")
    container["env"] = [e for e in container["env"] if e["name"] != "MODEL_EXPRESS_TLS_CA_FILE"]
    container["volumeMounts"] = [m for m in container["volumeMounts"] if m["name"] != "modelexpress-ca"]

    problems = engine_wiring_problems(pod, wiring)

    assert any("MODEL_EXPRESS_TLS_CA_FILE" in p for p in problems)
    assert any("/etc/modelexpress/ca mounts ConfigMap None" in p for p in problems)


def test_engine_pod_with_the_wrong_token_audience_is_reported():
    pod = _fixture("engine-pod.json")
    wiring = EngineWiring.from_service(_fixture("llmisvc.json"))

    problems = engine_wiring_problems(pod, EngineWiring(wiring.address, "other", wiring.ca_configmap, wiring.ca_key))

    assert problems == ["/var/run/secrets/modelexpress holds projected tokens for ['modelexpress'], want ['other']"]


def test_restrictive_network_policy_matches_the_server():
    assert network_policy_problems(_fixture("networkpolicy.json"), _fixture("server.json")) == []


def test_compatibility_policy_fails_a_restrictive_server():
    policy = _fixture("networkpolicy.json")
    for rule in policy["spec"]["ingress"]:
        rule.pop("from", None)

    problems = network_policy_problems(policy, _fixture("server.json"))

    assert any("admits any source" in p for p in problems)
    assert any("gRPC port 8001 is admitted from None" in p for p in problems)


def test_policy_owned_by_another_object_is_reported():
    policy = _fixture("networkpolicy.json")
    policy["metadata"]["ownerReferences"] = []
    server = copy.deepcopy(_fixture("server.json"))
    server["spec"].pop("networkPolicy")

    assert network_policy_problems(policy, server) == [
        "NetworkPolicy/modelexpress is not owned by ModelExpressServer/modelexpress (owners: [])"
    ]


def test_scc_rolebinding_grants_the_scc_to_the_engine_account():
    rolebinding = _fixture("scc-rolebinding.json")

    assert scc_binding_problems(rolebinding, "modelexpress-p2p-kserve") == []
    assert scc_binding_problems(rolebinding, "someone-else") == [
        "RoleBinding subjects [('ServiceAccount', 'modelexpress-p2p-kserve')] miss ServiceAccount/someone-else"
    ]


def test_cold_replica_falls_through_to_the_strategy_that_loaded():
    log = (_FIXTURES / "engine-cold.log").read_text()

    strategies = loaded_strategies(log)

    assert strategies
    assert "rdma" not in strategies.values()


def test_scaled_out_replica_loads_over_p2p():
    assert set(loaded_strategies((_FIXTURES / "engine-p2p.log").read_text()).values()) == {"rdma"}


def test_a_worker_whose_every_strategy_failed_loaded_nothing():
    log = "\n".join(
        [
            "(EngineCore pid=1) INFO [Worker 0] Trying strategy: rdma",
            "(EngineCore pid=1) WARNING [Worker 0] Strategy rdma failed, trying next: no source",
            "(EngineCore pid=1) INFO [Worker 0] Trying strategy: default",
            "(EngineCore pid=1) WARNING [Worker 0] Strategy default raised unexpected error, trying next: boom",
        ]
    )

    assert loaded_strategies(log) == {}


def test_probe_output_is_read_past_client_logging():
    output = 'INFO grpc channel ready\n{"ok": false, "code": "PERMISSION_DENIED", "detail": "not allowed"}\n'

    result = parse_probe_output(output)

    assert (result.ok, result.code, result.describe()) == (False, "PERMISSION_DENIED", "PERMISSION_DENIED not allowed")
    assert parse_probe_output('{"ok": true, "sources": [{"workerId": "a1"}]}').worker_ids == ("a1",)
