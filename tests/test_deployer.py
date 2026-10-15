"""Unit/smoke tests for the llm-d-e2e framework — no cluster required.

Validates config loading, client helpers, deployer retry/fast-fail logic,
manifest setup, and scaffolding scripts without kubectl against a live cluster.
Run with: ``uv run pytest tests/test_smoke.py -v`` (or ``make unittest``).

Coverage areas:
  Config — duration parsing; load testcase/profile/dir; LoRA single/multi YAML
  Metrics — Prometheus text exposition parsing
  Client — bearer token headers; chat() string vs message-list prompts
  Deployer — is_deployed tracking; workload pod listing; webhook/CRD transient
    apply retries; wait_for_ready persistent-error fast-fail and timeout messages;
    operator ImagePullBackOff surfacing; env_overrides on decode+prefill
  Manifests — --setup pruning of stale YAML; _require_manifest skip helpers
  Scaffolding — scripts/new-testcase.sh generates loadable config; rejects dupes
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from conformance.config import load_testcase, load_profile, load_testcases_from_dir, parse_duration
from conformance.metrics import ScrapeResult, parse_prometheus, validate_flow_control, validate_scheduler
from conformance.client import LLMClient


def test_parse_duration():
    assert parse_duration("15m").total_seconds() == 900
    assert parse_duration("2h").total_seconds() == 7200
    assert parse_duration("300s").total_seconds() == 300
    assert parse_duration("1h30m").total_seconds() == 5400


def test_load_testcase():
    tc = load_testcase("configs/testcases/single-gpu-smoke.yaml")
    assert tc.name == "single-gpu-smoke"
    assert tc.model.name == "Qwen/Qwen3-0.6B"
    assert tc.deployment.manifest_path == "single-gpu-smoke.yaml"
    assert tc.validation.health_port == 8000
    assert tc.validation.test_prompts


def test_manifest_model_and_lora_override_testcase(tmp_path):
    from conformance.config import LoRAConfig
    from conformance.deployer import Deployer

    tc = load_testcase("configs/testcases/lora-single.yaml")
    tc.model.name = "stale/model"
    tc.model.uri = "hf://stale/model"
    tc.model.lora = LoRAConfig(adapters=[{"name": "stale-adapter", "uri": "hf://stale/adapter"}])

    manifest_path = tmp_path / "lora-single.yaml"
    manifest_path.write_text(
        """apiVersion: serving.kserve.io/v1alpha2
kind: LLMInferenceService
metadata:
  name: lora-single
spec:
  model:
    name: Qwen/Qwen3-0.6B
    uri: hf://Qwen/Qwen3-0.6B
    lora:
      adapters:
        - name: tldr-adapter
          uri: hf://phh/Qwen3-0.6B-TLDR-Lora
      maxRank: 16
      maxAdapters: 1
"""
    )

    patched = Deployer(manifest_dir=str(tmp_path))._patch_manifest(manifest_path, tc)

    assert tc.model.name == "Qwen/Qwen3-0.6B"
    assert tc.model.uri == "hf://Qwen/Qwen3-0.6B"
    assert tc.model.lora is not None
    assert tc.model.lora.adapters == [{"name": "tldr-adapter", "uri": "hf://phh/Qwen3-0.6B-TLDR-Lora"}]
    assert tc.model.lora.max_rank == 16
    assert tc.model.lora.max_adapters == 1
    assert patched[0]["spec"]["model"]["name"] == "Qwen/Qwen3-0.6B"
    assert patched[0]["spec"]["model"]["lora"]["adapters"][0]["name"] == "tldr-adapter"


def test_multi_document_manifest_model_is_applied(tmp_path):
    from conformance.config import apply_manifest_model_config
    from conformance.deployer import MANAGED_LABEL, Deployer

    tc = load_testcase("configs/testcases/lora-single.yaml")
    tc.model.name = "stale/model"
    tc.model.uri = "hf://stale/model"
    tc.model.lora = None
    manifest_path = tmp_path / "multi.yaml"
    model = """name: Qwen/Qwen3-0.6B
uri: hf://Qwen/Qwen3-0.6B
lora:
  adapters:
    - name: tldr-adapter
      uri: hf://phh/Qwen3-0.6B-TLDR-Lora
  maxRank: 16
  maxAdapters: 1
"""
    manifest_path.write_text(
        """apiVersion: serving.kserve.io/v1alpha2
kind: LLMInferenceService
metadata:
  name: lora-single-a
spec:
  model:
"""
        + "".join("    " + line for line in model.splitlines(keepends=True))
        + "---\n"
        + """apiVersion: v1
kind: ConfigMap
metadata:
  name: unrelated
data:
  note: ignored
---
apiVersion: serving.kserve.io/v1alpha2
kind: LLMInferenceService
metadata:
  name: lora-single-b
spec:
  model:
"""
        + "".join("    " + line for line in model.splitlines(keepends=True))
    )

    apply_manifest_model_config(tc, manifest_path)

    assert tc.model.name == "Qwen/Qwen3-0.6B"
    assert tc.model.uri == "hf://Qwen/Qwen3-0.6B"
    assert tc.model.lora is not None
    assert tc.model.lora.adapters == [{"name": "tldr-adapter", "uri": "hf://phh/Qwen3-0.6B-TLDR-Lora"}]
    assert tc.model.lora.max_rank == 16
    assert tc.model.lora.max_adapters == 1

    patched = Deployer(manifest_dir=str(tmp_path))._patch_manifest(manifest_path, tc)
    assert [doc["kind"] for doc in patched] == ["LLMInferenceService", "ConfigMap", "LLMInferenceService"]
    assert [doc["metadata"]["name"] for doc in (patched[0], patched[2])] == ["lora-single", "lora-single-b"]
    assert all(doc["spec"]["model"]["name"] == "Qwen/Qwen3-0.6B" for doc in (patched[0], patched[2]))
    # Every deployed service carries the ownership label the start-of-run sweep selects on.
    assert all(doc["metadata"]["labels"][MANAGED_LABEL] == "true" for doc in (patched[0], patched[2]))
    assert "labels" not in patched[1]["metadata"]


def test_manifest_service_names_keep_all_documents_and_use_testcase_name_for_primary(tmp_path):
    from conformance.deployer import Deployer

    manifest_path = tmp_path / "multi.yaml"
    manifest_path.write_text(
        """apiVersion: serving.kserve.io/v1alpha2
kind: LLMInferenceService
metadata:
  name: service-a
spec: {}
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: unrelated
---
apiVersion: serving.kserve.io/v1alpha2
kind: LLMInferenceService
metadata:
  name: service-b
spec: {}
"""
    )
    tc = load_testcase("configs/testcases/lora-single.yaml")
    tc.name = "case"
    tc.deployment.manifest_path = "multi.yaml"

    deployer = Deployer(manifest_dir=str(tmp_path))

    assert deployer.manifest_service_names(tc) == ["case", "service-b"]


def test_discover_mode_validates_only_services_the_target_deployed(tmp_path, monkeypatch):
    """An existing target without a secondary pool must not wait on that pool or count its replicas."""
    from conformance.deployer import Deployer

    (tmp_path / "multi.yaml").write_text(
        """apiVersion: serving.kserve.io/v1alpha2
kind: LLMInferenceService
metadata:
  name: service-a
spec:
  replicas: 2
---
apiVersion: serving.kserve.io/v1alpha2
kind: LLMInferenceService
metadata:
  name: service-b
spec:
  replicas: 3
"""
    )
    tc = load_testcase("configs/testcases/single-gpu-smoke.yaml")
    tc.name = "case"
    tc.deployment.manifest_path = "multi.yaml"
    deployer = Deployer(manifest_dir=str(tmp_path))
    existing = {"case"}
    monkeypatch.setattr(deployer, "check_resource_exists", lambda kind, name: name in existing)

    names = deployer.existing_service_names(tc)
    assert names == ["case"]
    assert deployer.manifest_replicas(tc, names) == 2

    existing.add("service-b")
    names = deployer.existing_service_names(tc)
    assert names == ["case", "service-b"]
    assert deployer.manifest_replicas(tc, names) == 5

    # The primary is always checked so a missing target still fails.
    existing.clear()
    assert deployer.existing_service_names(tc) == ["case"]


def test_wait_for_pods_checks_all_manifest_services(monkeypatch):
    from conformance.deployer import Deployer

    deployer = Deployer()
    responses = {
        "app.kubernetes.io/name=case,app.kubernetes.io/component=llminferenceservice-workload": "pod-a=Running",
        "app.kubernetes.io/name=case,app.kubernetes.io/component=llminferenceservice-workload-prefill": "pod-prefill=Running",
        "app.kubernetes.io/name=service-b,app.kubernetes.io/component=llminferenceservice-workload": "pod-b=Running",
    }
    monkeypatch.setattr(
        deployer,
        "kubectl",
        lambda *args, **kwargs: responses.get(args[5], ""),
    )
    monkeypatch.setattr(deployer, "_check_crashloop", lambda name: [])

    assert deployer.wait_for_pods(["case", "service-b"], timeout=1, min_pods=3) == [
        "pod-a",
        "pod-prefill",
        "pod-b",
    ]


def test_cleanup_deletes_all_manifest_services(tmp_path, monkeypatch):
    from conformance.deployer import Deployer

    manifest_path = tmp_path / "multi.yaml"
    manifest_path.write_text(
        """apiVersion: serving.kserve.io/v1alpha2
kind: LLMInferenceService
metadata:
  name: service-a
spec: {}
---
apiVersion: serving.kserve.io/v1alpha2
kind: LLMInferenceService
metadata:
  name: service-b
spec: {}
"""
    )
    tc = load_testcase("configs/testcases/lora-single.yaml")
    tc.name = "case"
    tc.deployment.manifest_path = "multi.yaml"
    deployer = Deployer(manifest_dir=str(tmp_path))
    deleted_services = []

    def fake_kubectl(*args, **kwargs):
        if args[:2] == ("delete", "llminferenceservice"):
            deleted_services.append(args[2])
        return ""

    monkeypatch.setattr(deployer, "kubectl", fake_kubectl)

    deployer.cleanup(tc, timeout=1)

    assert deleted_services == ["case", "service-b"]


@pytest.mark.parametrize("failed_service", ["case", "service-b"])
def test_cleanup_failed_delete_stays_pending_for_teardown_retry(tmp_path, monkeypatch, failed_service):
    import subprocess

    from conformance.deployer import Deployer

    from conftest import _teardown_deployer

    (tmp_path / "multi.yaml").write_text(
        """apiVersion: serving.kserve.io/v1alpha2
kind: LLMInferenceService
metadata:
  name: service-a
spec: {}
---
apiVersion: serving.kserve.io/v1alpha2
kind: LLMInferenceService
metadata:
  name: service-b
spec: {}
"""
    )
    tc = load_testcase("configs/testcases/single-gpu-smoke.yaml")
    tc.name = "case"
    tc.deployment.manifest_path = "multi.yaml"
    deployer = Deployer(manifest_dir=str(tmp_path))
    deployer._applied[tc.name] = tc
    remaining = {"case", "service-b"}
    fail_once = True

    def fake_run(cmd, **kwargs):
        nonlocal fail_once
        if cmd[1:3] == ["delete", "llminferenceservice"]:
            name = cmd[3]
            if name == failed_service and fail_once:
                fail_once = False
                return subprocess.CompletedProcess(cmd, 1, "", "API server unavailable")
            remaining.discard(name)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr("conformance.deployer.subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="API server unavailable"):
        deployer.cleanup(tc, timeout=1)

    assert deployer.needs_cleanup(tc.name)
    # Every other service is still deleted on the failing pass, not only on retry.
    assert remaining == {failed_service}

    _teardown_deployer(deployer, no_cleanup=False)

    assert not remaining
    assert not deployer.needs_cleanup(tc.name)


def test_failed_multi_document_apply_is_cleaned_up(tmp_path, monkeypatch):
    """kubectl apply can create earlier documents before a later one is rejected;
    phase 99 must still delete every manifest service after a failed apply."""
    from conformance.deployer import Deployer

    manifest_path = tmp_path / "multi.yaml"
    manifest_path.write_text(
        """apiVersion: serving.kserve.io/v1alpha2
kind: LLMInferenceService
metadata:
  name: service-a
spec: {}
---
apiVersion: serving.kserve.io/v1alpha2
kind: LLMInferenceService
metadata:
  name: service-b
spec: {}
"""
    )
    tc = load_testcase("configs/testcases/lora-single.yaml")
    tc.name = "case"
    tc.deployment.manifest_path = "multi.yaml"
    deployer = Deployer(manifest_dir=str(tmp_path))
    deleted_services = []

    def fake_kubectl(*args, **kwargs):
        if args[:2] == ("delete", "llminferenceservice"):
            deleted_services.append(args[2])
        return ""

    def failing_apply(path, namespace=""):
        raise RuntimeError("admission webhook denied service-b")

    monkeypatch.setattr(deployer, "kubectl", fake_kubectl)
    monkeypatch.setattr(deployer, "ensure_namespace", lambda: None)
    monkeypatch.setattr(deployer, "_ensure_clean_slate", lambda name: None)
    monkeypatch.setattr(deployer, "_apply_with_webhook_retry", failing_apply)

    result = deployer.deploy(tc)

    assert not result.success
    assert "service-b" in result.error
    assert not deployer.is_deployed(tc.name)
    assert deployer.needs_cleanup(tc.name)

    deployer.cleanup(tc, timeout=1)

    assert deleted_services == ["case", "service-b"]
    assert not deployer.needs_cleanup(tc.name)


def _record_applies(deployer, monkeypatch) -> list[tuple[str, list[dict]]]:
    """Stub cluster calls; record each ``kubectl apply`` as (namespace, documents)."""
    import yaml

    applies = []

    def fake_kubectl(*args, **kwargs):
        if args[0] == "apply":
            applies.append((args[2], list(yaml.safe_load_all(Path(args[4]).read_text()))))
        return ""

    monkeypatch.setattr(deployer, "kubectl", fake_kubectl)
    monkeypatch.setattr(deployer, "ensure_namespace", lambda: None)
    monkeypatch.setattr(deployer, "_ensure_clean_slate", lambda name: None)
    monkeypatch.setattr(deployer, "ensure_metrics_rbac", lambda name: None)
    return applies


def test_single_namespace_manifest_is_one_apply_into_test_namespace(tmp_path, monkeypatch):
    """Manifests without declared namespaces (every llm-d testcase) apply exactly as before."""
    from conformance.deployer import Deployer

    (tmp_path / "multi.yaml").write_text(
        "apiVersion: serving.kserve.io/v1alpha2\nkind: LLMInferenceService\nmetadata:\n  name: a\nspec: {}\n"
        "---\n"
        "apiVersion: serving.kserve.io/v1alpha2\nkind: LLMInferenceService\nmetadata:\n  name: b\nspec: {}\n"
    )
    tc = load_testcase("configs/testcases/lora-single.yaml")
    tc.name = "case"
    tc.deployment.manifest_path = "multi.yaml"
    deployer = Deployer(manifest_dir=str(tmp_path), namespace="llm-conformance-test")
    applies = _record_applies(deployer, monkeypatch)

    assert deployer.deploy(tc).success

    assert [(ns, [doc["metadata"]["name"] for doc in docs]) for ns, docs in applies] == [
        ("llm-conformance-test", ["case", "b"])
    ]


_MAAS_MANIFEST = """apiVersion: serving.kserve.io/v1alpha2
kind: LLMInferenceService
metadata:
  name: maas-single-gpu
spec: {}
---
apiVersion: maas.opendatahub.io/v1alpha1
kind: MaaSModelRef
metadata:
  name: maas-single-gpu
spec:
  modelRef:
    kind: LLMInferenceService
    name: maas-single-gpu
---
apiVersion: maas.opendatahub.io/v1alpha1
kind: MaaSAuthPolicy
metadata:
  name: maas-single-gpu-auth
  namespace: models-as-a-service
spec:
  modelRefs:
  - name: maas-single-gpu
    namespace: llm-conformance-test
"""


def test_cross_namespace_documents_apply_into_their_namespace_and_follow_overrides(tmp_path, monkeypatch):
    """MaaS policies live in models-as-a-service; refs follow the testcase name and --namespace."""
    from conformance.deployer import Deployer

    (tmp_path / "maas.yaml").write_text(_MAAS_MANIFEST)
    tc = load_testcase("configs/testcases/single-gpu.yaml")
    tc.name = "renamed"
    tc.deployment.manifest_path = "maas.yaml"
    deployer = Deployer(manifest_dir=str(tmp_path), namespace="team-ns")
    applies = _record_applies(deployer, monkeypatch)

    assert deployer.deploy(tc).success

    assert [(ns, [doc["kind"] for doc in docs]) for ns, docs in applies] == [
        ("team-ns", ["LLMInferenceService", "MaaSModelRef"]),
        ("models-as-a-service", ["MaaSAuthPolicy"]),
    ]
    assert applies[0][1][1]["spec"]["modelRef"]["name"] == "renamed"
    assert applies[1][1][0]["spec"]["modelRefs"] == [{"name": "maas-single-gpu", "namespace": "team-ns"}]


def test_cleanup_deletes_companion_documents_before_services(tmp_path, monkeypatch):
    from conformance.deployer import Deployer

    (tmp_path / "maas.yaml").write_text(_MAAS_MANIFEST)
    tc = load_testcase("configs/testcases/single-gpu.yaml")
    tc.deployment.manifest_path = "maas.yaml"
    deployer = Deployer(manifest_dir=str(tmp_path), namespace="llm-conformance-test")
    deletes = []

    def fake_kubectl(*args, **kwargs):
        if args[0] == "delete":
            deletes.append((args[1], args[2], args[4]))
        return ""

    monkeypatch.setattr(deployer, "kubectl", fake_kubectl)
    monkeypatch.setattr(deployer, "cleanup_metrics_rbac", lambda name: None)

    deployer.cleanup(tc, timeout=1)

    assert deletes == [
        ("maasauthpolicy", "maas-single-gpu-auth", "models-as-a-service"),
        ("maasmodelref", "maas-single-gpu", "llm-conformance-test"),
        ("llminferenceservice", tc.name, "llm-conformance-test"),
    ]


def test_deploy_failure_before_apply_needs_no_cleanup(tmp_path, monkeypatch):
    from conformance.deployer import Deployer

    manifest_path = tmp_path / "case.yaml"
    manifest_path.write_text("apiVersion: serving.kserve.io/v1alpha2\nkind: LLMInferenceService\nspec: {}\n")
    tc = load_testcase("configs/testcases/lora-single.yaml")
    tc.name = "case"
    tc.deployment.manifest_path = "case.yaml"
    deployer = Deployer(manifest_dir=str(tmp_path))

    def failing_namespace():
        raise RuntimeError("namespace forbidden")

    monkeypatch.setattr(deployer, "kubectl", lambda *args, **kwargs: "")
    monkeypatch.setattr(deployer, "ensure_namespace", failing_namespace)

    result = deployer.deploy(tc)

    assert not result.success
    assert not deployer.needs_cleanup(tc.name)


def test_cleanup_pending_deletes_uncleaned_cases_and_honors_tc_cleanup(tmp_path, monkeypatch):
    """Session teardown deletes cases an interrupted run left applied, but keeps cleanup: false cases."""
    from conformance.deployer import Deployer

    for name in ("interrupted", "keep-me"):
        (tmp_path / f"{name}.yaml").write_text(
            f"apiVersion: serving.kserve.io/v1alpha2\nkind: LLMInferenceService\nmetadata:\n  name: {name}\nspec: {{}}\n"
        )
    deployer = Deployer(manifest_dir=str(tmp_path))
    for name, cleanup in (("interrupted", True), ("keep-me", False)):
        tc = load_testcase("configs/testcases/single-gpu-smoke.yaml")
        tc.name = name
        tc.cleanup = cleanup
        tc.deployment.manifest_path = f"{name}.yaml"
        deployer._applied[name] = tc
    deleted_services = []

    def fake_kubectl(*args, **kwargs):
        if args[:2] == ("delete", "llminferenceservice"):
            deleted_services.append(args[2])
        return ""

    monkeypatch.setattr(deployer, "kubectl", fake_kubectl)

    deployer.cleanup_pending()

    assert deleted_services == ["interrupted"]
    assert not deployer.needs_cleanup("interrupted")
    assert deployer.needs_cleanup("keep-me")


def test_remove_stale_services_deletes_only_suite_owned_leftovers(monkeypatch):
    """Services left by a killed run hold GPUs and must go; services the suite did not deploy must stay."""
    from conformance.deployer import MANAGED_LABEL, Deployer

    deployer = Deployer()
    owned = {"flow-control", "single-gpu"}
    services = {"flow-control": True, "other-team": False, "single-gpu": True}

    def fake_kubectl(*args, **kwargs):
        selector = args[args.index("-l") + 1] if "-l" in args else ""
        if selector == f"{MANAGED_LABEL}=true":
            return " ".join(name for name, managed in services.items() if managed)
        return " ".join(services)

    cleaned = []
    monkeypatch.setattr(deployer, "kubectl", fake_kubectl)
    monkeypatch.setattr(deployer, "_ensure_clean_slate", cleaned.append)

    assert deployer.remove_stale_services() == sorted(owned)
    assert cleaned == sorted(owned)


def test_deployer_teardown_cleans_up_when_port_forward_stop_fails():
    """A port-forward that will not exit must not leave GPU-holding services behind."""
    import subprocess

    from conftest import _teardown_deployer

    class FakeDeployer:
        cleaned = False

        def stop_port_forward(self):
            raise subprocess.TimeoutExpired("kubectl port-forward", 5)

        def cleanup_pending(self):
            self.cleaned = True

    deployer = FakeDeployer()
    with pytest.raises(subprocess.TimeoutExpired):
        _teardown_deployer(deployer, no_cleanup=False)
    assert deployer.cleaned

    kept = FakeDeployer()
    with pytest.raises(subprocess.TimeoutExpired):
        _teardown_deployer(kept, no_cleanup=True)
    assert not kept.cleaned


def test_multi_document_manifest_with_conflicting_models_fails(tmp_path):
    from conformance.config import apply_manifest_model_config

    tc = load_testcase("configs/testcases/lora-single.yaml")
    manifest_path = tmp_path / "conflict.yaml"
    manifest_path.write_text(
        """apiVersion: serving.kserve.io/v1alpha2
kind: LLMInferenceService
metadata:
  name: model-a
spec:
  model:
    name: model-a
    uri: hf://model-a
---
apiVersion: serving.kserve.io/v1alpha2
kind: LLMInferenceService
metadata:
  name: model-b
spec:
  model:
    name: model-b
    uri: hf://model-a
"""
    )

    with pytest.raises(
        ValueError,
        match=r"manifest .*conflict\.yaml: LLMInferenceService documents declare different spec\.model name/uri/lora",
    ):
        apply_manifest_model_config(tc, manifest_path)


@pytest.mark.parametrize("model_spec_yaml", ["spec: not-a-mapping\n", "spec:\n  model: not-a-mapping\n"])
def test_manifest_non_mapping_spec_or_model_is_absent(tmp_path, model_spec_yaml):
    from conformance.config import apply_manifest_model_config

    tc = load_testcase("configs/testcases/lora-single.yaml")
    expected_uri = tc.model.uri
    expected_lora = tc.model.lora
    manifest_path = tmp_path / "non-mapping.yaml"
    manifest_path.write_text(
        """apiVersion: serving.kserve.io/v1alpha2
kind: LLMInferenceService
metadata:
  name: lora-single
"""
        + model_spec_yaml
    )

    apply_manifest_model_config(tc, manifest_path, model_name_override="cli-alias")

    assert tc.model.name == "cli-alias"
    assert tc.model.uri == expected_uri
    assert tc.model.lora == expected_lora


def test_manifest_without_llmisvc_keeps_testcase_and_cli_metadata(tmp_path):
    from conformance.config import apply_manifest_model_config

    tc = load_testcase("configs/testcases/lora-single.yaml")
    expected_uri = tc.model.uri
    expected_lora = tc.model.lora
    manifest_path = tmp_path / "configmap.yaml"
    manifest_path.write_text("apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: unrelated\n")

    apply_manifest_model_config(tc, manifest_path, model_name_override="cli-alias")

    assert tc.model.name == "cli-alias"
    assert tc.model.uri == expected_uri
    assert tc.model.lora == expected_lora


def test_discover_mode_uses_testcase_model_and_lora_with_cli_name_override(tmp_path):
    from conftest import _resolve_test_cases

    testcase_dir = tmp_path / "testcases"
    manifest_dir = tmp_path / "deploy" / "manifests"
    testcase_dir.mkdir()
    manifest_dir.mkdir(parents=True)
    (testcase_dir / "case.yaml").write_text(
        """name: case
model:
  name: testcase/model
  uri: hf://testcase/model
  lora:
    adapters:
      - name: testcase-adapter
        uri: hf://testcase/adapter
deployment:
  manifestPath: case.yaml
"""
    )
    (manifest_dir / "case.yaml").write_text(
        """apiVersion: serving.kserve.io/v1alpha2
kind: LLMInferenceService
metadata:
  name: case
spec:
  model:
    name: manifest/model
    uri: hf://manifest/model
    lora:
      adapters:
        - name: manifest-adapter
          uri: hf://manifest/adapter
"""
    )

    class PytestConfigStub:
        rootpath = tmp_path

        def getoption(self, name):
            return {
                "--testcase-dir": str(testcase_dir),
                "--profile": "",
                "--testcase": "",
                "--mode": "discover",
                "--model": "cli-alias",
            }[name]

    [tc] = _resolve_test_cases(PytestConfigStub())

    assert tc.model.name == "cli-alias"
    assert tc.model.uri == "hf://testcase/model"
    assert tc.model.lora is not None
    assert tc.model.lora.adapters == [{"name": "testcase-adapter", "uri": "hf://testcase/adapter"}]


def test_cli_model_override_takes_precedence_over_manifest(tmp_path):
    from conformance.config import load_testcase
    from conformance.deployer import Deployer

    manifest_path = tmp_path / "case.yaml"
    manifest_path.write_text(
        """apiVersion: serving.kserve.io/v1alpha2
kind: LLMInferenceService
metadata:
  name: case
spec:
  model:
    name: Qwen/Qwen3-0.6B
    uri: hf://Qwen/Qwen3-0.6B
"""
    )
    tc = load_testcase("configs/testcases/single-gpu-smoke.yaml")
    tc.name = "case"
    tc.deployment.manifest_path = "case.yaml"

    deployer = Deployer(manifest_dir=str(tmp_path), model_override="served-alias")
    patched = deployer._patch_manifest(manifest_path, tc)

    assert patched[0]["spec"]["model"]["name"] == "served-alias"
    assert patched[0]["spec"]["model"]["uri"] == "hf://Qwen/Qwen3-0.6B"
    assert tc.model.name == "served-alias"


def test_load_profile():
    profile = load_profile("configs/profiles/smoke.yaml")
    assert profile.name == "smoke"
    assert "single-gpu-smoke" in profile.test_cases


def test_load_all_testcases():
    cases = load_testcases_from_dir("configs/testcases")
    assert len(cases) >= 1
    names = [tc.name for tc in cases]
    assert "single-gpu-smoke" in names


def test_dir_loaders_skip_readme(tmp_path):
    """README.md (and README*.yaml) next to configs must not be loaded as cases/profiles."""
    from conformance.config import iter_config_yamls, load_profiles_from_dir, load_testcases_from_dir

    cases_dir = tmp_path / "testcases"
    profiles_dir = tmp_path / "profiles"
    cases_dir.mkdir()
    profiles_dir.mkdir()

    (cases_dir / "README.md").write_text("# docs\n")
    (cases_dir / "ok.yaml").write_text("name: ok\nmodel:\n  name: m\ndeployment:\n  manifestPath: x.yaml\n")
    (cases_dir / "README.yaml").write_text("name: should-skip\n")
    (profiles_dir / "README.md").write_text("# docs\n")
    (profiles_dir / "smoke.yaml").write_text("name: smoke\ntestCases:\n  - ok\n")

    assert [p.name for p in iter_config_yamls(cases_dir)] == ["ok.yaml"]
    assert [p.name for p in iter_config_yamls(profiles_dir)] == ["smoke.yaml"]
    assert [tc.name for tc in load_testcases_from_dir(cases_dir)] == ["ok"]
    assert [p.name for p in load_profiles_from_dir(profiles_dir)] == ["smoke"]


def test_parse_prometheus_text():
    text = """# HELP vllm:request_success_total Total requests
# TYPE vllm:request_success_total counter
vllm:request_success_total{model_name="Qwen/Qwen3-0.6B"} 42.0
vllm:gpu_cache_usage_perc 0.15
"""
    metrics = parse_prometheus(text)
    assert "vllm:request_success_total" in metrics
    assert metrics["vllm:request_success_total"][0].value == 42.0
    assert metrics["vllm:request_success_total"][0].labels["model_name"] == "Qwen/Qwen3-0.6B"
    assert metrics["vllm:gpu_cache_usage_perc"][0].value == 0.15


def test_router_main_metrics_are_validated():
    text = """# TYPE llm_d_epp_scheduler_e2e_duration_seconds histogram
llm_d_epp_scheduler_e2e_duration_seconds_count 1
llm_d_epp_ready_endpoints 1
llm_d_epp_flow_control_dispatch_cycle_duration_seconds_count 1
llm_d_epp_flow_control_pool_saturation 0.5
llm_d_epp_flow_control_request_enqueue_duration_seconds_count 1
llm_d_epp_flow_control_request_queue_duration_seconds_count 1
"""
    result = ScrapeResult(source="epp", metrics=parse_prometheus(text))

    scheduler_checks = validate_scheduler([result])
    assert all(check.passed for check in scheduler_checks)
    assert any(check.name == "ready_endpoints" for check in scheduler_checks)
    assert all(check.passed for check in validate_flow_control([result]))


def test_legacy_router_metrics_remain_supported():
    text = """inference_extension_scheduler_e2e_duration_seconds_count 1
inference_pool_ready_pods 1
inference_extension_flow_control_dispatch_cycle_duration_seconds_count 1
inference_extension_flow_control_pool_saturation 0.5
inference_extension_flow_control_request_enqueue_duration_seconds_count 1
inference_extension_flow_control_request_queue_duration_seconds_count 1
"""
    result = ScrapeResult(source="epp", metrics=parse_prometheus(text))

    scheduler_checks = validate_scheduler([result])
    assert all(check.passed for check in scheduler_checks)
    assert any(check.name == "ready_endpoints" for check in scheduler_checks)
    assert all(check.passed for check in validate_flow_control([result]))


def test_llm_client_init():
    c = LLMClient(base_url="http://localhost:8000", bearer_token="test-token")
    assert c._client.headers.get("authorization") == "Bearer test-token"
    c.close()


def test_deployer_is_deployed():
    from conformance.deployer import Deployer

    d = Deployer()
    assert not d.is_deployed("foo")
    d._deployed.add("foo")
    assert d.is_deployed("foo")
    d._deployed.discard("foo")
    assert not d.is_deployed("foo")


def test_cluster_gpu_count_total(monkeypatch):
    """cluster_gpu_count sums allocatable nvidia.com/gpu across nodes, ignoring blank (CPU) nodes."""
    from conformance.deployer import Deployer

    d = Deployer()
    # jsonpath output: one line per node, blank for nodes without GPUs.
    monkeypatch.setattr(d, "kubectl", lambda *a, **k: "1\n\n2\n")
    assert d.cluster_gpu_count() == 3


def test_list_workload_pods_parses_and_filters_blanks(monkeypatch):
    """list_workload_pods returns pod names from jsonpath output, dropping blank lines."""
    from conformance.deployer import Deployer

    d = Deployer()
    monkeypatch.setattr(d, "kubectl", lambda *a, **k: "pod-a\n\npod-b\n")
    assert d.list_workload_pods("my-isvc") == ["pod-a", "pod-b"]


def _write_manifest(tmp_path, body: str) -> "tuple":
    """Write a manifest into a temp manifest_dir; return (Deployer, tc) wired to it.

    A ``spec:``-only body is wrapped in a minimal LLMInferenceService document so
    the kind filter in Deployer._manifest_specs recognizes it; full documents
    (with their own ``kind:``) are written verbatim.
    """
    from conformance.deployer import Deployer

    if body.lstrip().startswith("spec:"):
        body = "apiVersion: serving.kserve.io/v1alpha2\nkind: LLMInferenceService\n" + body
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    (manifest_dir / "case.yaml").write_text(body)
    d = Deployer(manifest_dir=str(manifest_dir))
    tc = load_testcase("configs/testcases/single-gpu-smoke.yaml")
    tc.deployment.manifest_path = "case.yaml"
    return d, tc


def test_manifest_replicas_sums_decode_and_prefill(tmp_path):
    """manifest_replicas = spec.replicas + spec.prefill.replicas (P/D)."""
    d, tc = _write_manifest(
        tmp_path,
        "spec:\n  replicas: 2\n  prefill:\n    replicas: 3\n",
    )
    assert d.manifest_replicas(tc) == 5


def test_manifest_replicas_defaults_to_one(tmp_path):
    """Absent spec.replicas defaults to 1 (matches KServe)."""
    d, tc = _write_manifest(tmp_path, "spec:\n  model:\n    name: x\n")
    assert d.manifest_replicas(tc) == 1


def test_manifest_gpu_needed_sums_decode_and_prefill(tmp_path):
    """manifest_gpu_needed = sum over decode+prefill of replicas x peak container gpu limit."""
    body = (
        "spec:\n"
        "  replicas: 2\n"
        "  template:\n"
        "    containers:\n"
        "      - name: main\n"
        "        resources:\n"
        "          limits:\n"
        "            nvidia.com/gpu: 1\n"
        "  prefill:\n"
        "    replicas: 3\n"
        "    template:\n"
        "      containers:\n"
        "        - name: main\n"
        "          resources:\n"
        "            limits:\n"
        "              nvidia.com/gpu: 2\n"
    )
    d, tc = _write_manifest(tmp_path, body)
    assert d.manifest_gpu_needed(tc) == 2 * 1 + 3 * 2  # 8


def test_manifest_gpu_needed_zero_when_no_gpu(tmp_path):
    """A CPU-only manifest requests zero GPUs (so _require_gpu won't skip on GPU count)."""
    body = "spec:\n  replicas: 1\n  template:\n    containers:\n      - name: main\n"
    d, tc = _write_manifest(tmp_path, body)
    assert d.manifest_gpu_needed(tc) == 0


def test_manifest_counts_sum_across_multi_document_manifest(tmp_path):
    """Multi-document manifests (e.g. multi-pool) are summed across LLMISVC docs, not crashed on."""
    doc = (
        "apiVersion: serving.kserve.io/v1alpha2\n"
        "kind: LLMInferenceService\n"
        "spec:\n"
        "  replicas: 1\n"
        "  template:\n"
        "    containers:\n"
        "      - name: main\n"
        "        resources:\n"
        "          limits:\n"
        "            nvidia.com/gpu: 1\n"
    )
    d, tc = _write_manifest(tmp_path, doc + "---\n" + doc)
    assert d.manifest_replicas(tc) == 2  # 1 + 1
    assert d.manifest_gpu_needed(tc) == 2  # 1 + 1


def test_manifest_counts_ignore_non_llmisvc_documents(tmp_path):
    """Non-LLMInferenceService documents in a manifest are skipped, not counted."""
    body = (
        "apiVersion: v1\n"
        "kind: ConfigMap\n"
        "metadata:\n  name: noise\n"
        "data:\n  replicas: '99'\n"
        "---\n"
        "apiVersion: serving.kserve.io/v1alpha2\n"
        "kind: LLMInferenceService\n"
        "spec:\n  replicas: 2\n"
    )
    d, tc = _write_manifest(tmp_path, body)
    assert d.manifest_replicas(tc) == 2  # ConfigMap ignored


def test_require_gpu_skips_in_mock_mode():
    """A requiresGpu test case skips in mock mode."""
    sys.path.insert(0, str(Path(__file__).parent))
    import test_conformance as tc_mod

    tc = load_testcase("configs/testcases/single-gpu-smoke.yaml")
    tc.deployment.requires_gpu = True
    with pytest.raises(pytest.skip.Exception, match="has requiresGpu set"):
        tc_mod._require_gpu(deployer=None, tc=tc, mock_mode=True, test_mode="deploy")


def test_require_gpu_runs_when_cluster_has_enough_gpu():
    """With enough GPUs available (manifest need <= cluster), the case does not skip."""
    sys.path.insert(0, str(Path(__file__).parent))
    import test_conformance as tc_mod

    class FakeDeployer:
        def manifest_gpu_needed(self, tc):
            return 2

        def cluster_gpu_count(self):
            return 2

    tc = load_testcase("configs/testcases/single-gpu-smoke.yaml")
    tc.deployment.requires_gpu = True
    tc_mod._require_gpu(deployer=FakeDeployer(), tc=tc, mock_mode=False, test_mode="deploy")


def test_require_gpu_skips_when_not_enough_gpus():
    """Skips when the cluster has fewer GPUs than the manifest requests."""
    sys.path.insert(0, str(Path(__file__).parent))
    import test_conformance as tc_mod

    class FakeDeployer:
        def manifest_gpu_needed(self, tc):
            return 3

        def cluster_gpu_count(self):
            return 1

    tc = load_testcase("configs/testcases/single-gpu-smoke.yaml")
    with pytest.raises(pytest.skip.Exception, match="needs 3 GPU"):
        tc_mod._require_gpu(deployer=FakeDeployer(), tc=tc, mock_mode=False, test_mode="deploy")


def test_apply_with_webhook_retry_succeeds_after_transient_error(monkeypatch):
    """Regression: deploy() used to fail immediately if the LLMInferenceService
    admission webhook wasn't serving yet (e.g. right after install/upgrade)."""
    from conformance.deployer import Deployer

    d = Deployer()
    calls = []

    def fake_kubectl(*args, **kwargs):
        calls.append(args)
        if len(calls) < 3:
            raise RuntimeError(
                "kubectl apply failed: Internal error occurred: failed calling webhook "
                '"llminferenceservice.kserve-webhook-server.v1alpha2.defaulter": '
                'failed to call webhook: Post "https://...": no endpoints available for service "llmisvc-webhook-server-service"'
            )
        return ""

    monkeypatch.setattr(d, "kubectl", fake_kubectl)
    monkeypatch.setattr("conformance.deployer.time.sleep", lambda _: None)

    d._apply_with_webhook_retry("dummy.yaml", timeout=60, interval=1)
    assert len(calls) == 3


def test_apply_with_webhook_retry_raises_immediately_on_other_errors(monkeypatch):
    """Non-webhook errors (e.g. a real manifest problem) must not be retried."""
    from conformance.deployer import Deployer

    d = Deployer()
    calls = []

    def fake_kubectl(*args, **kwargs):
        calls.append(args)
        raise RuntimeError("kubectl apply failed: error validating data: unknown field")

    monkeypatch.setattr(d, "kubectl", fake_kubectl)
    monkeypatch.setattr("conformance.deployer.time.sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="unknown field"):
        d._apply_with_webhook_retry("dummy.yaml", timeout=60, interval=1)
    assert len(calls) == 1


def test_apply_with_webhook_retry_times_out(monkeypatch):
    """If the webhook never comes up within the grace period, fail with a clear error."""
    from conformance.deployer import Deployer

    d = Deployer()

    def fake_kubectl(*args, **kwargs):
        raise RuntimeError(
            'kubectl apply failed: failed calling webhook "llminferenceservice...": no endpoints available for service'
        )

    monkeypatch.setattr(d, "kubectl", fake_kubectl)
    monkeypatch.setattr("conformance.deployer.time.sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="waiting for webhook"):
        d._apply_with_webhook_retry("dummy.yaml", timeout=0.05, interval=0.01)


def test_wait_for_ready_fails_fast_on_persistent_error(monkeypatch):
    """wait_for_ready should fail fast when any Ready=False condition with the
    same reason+message persists across 3 consecutive polls, regardless of the
    specific reason string (RBAC, missing CRD, webhook, etc.)."""
    from conformance.deployer import Deployer
    from conformance.config import load_testcase

    d = Deployer()
    tc = load_testcase("configs/testcases/single-gpu-smoke.yaml")

    def fake_kubectl(*args, **kwargs):
        for arg in args:
            if "status}" in str(arg) and "Ready" in str(arg) and "message" not in str(arg) and "reason" not in str(arg):
                return "False"
            if "reason}" in str(arg):
                return "SchedulerReconcileError"
            if "message}" in str(arg):
                return (
                    'roles.rbac.authorization.k8s.io "epp-role" is forbidden: '
                    "user is attempting to grant RBAC permissions not currently held"
                )
            if "CrashLoopBackOff" in str(arg) or "containerStatuses" in str(arg):
                return ""
        return ""

    monkeypatch.setattr(d, "kubectl", fake_kubectl)
    monkeypatch.setattr("conformance.deployer.time.sleep", lambda _: None)

    logs = []
    with pytest.raises(RuntimeError, match="Persistent controller error.*SchedulerReconcileError.*RBAC permissions"):
        d.wait_for_ready(tc, timeout=600, print_fn=logs.append)

    # Should have logged the message field
    assert any("RBAC permissions" in line for line in logs), f"Expected RBAC error message in log output, got: {logs}"


def test_wait_for_ready_fails_fast_on_any_repeated_error(monkeypatch):
    """Any controller error reason should trigger fast-fail when it persists,
    not just a hardcoded list."""
    from conformance.deployer import Deployer
    from conformance.config import load_testcase

    d = Deployer()
    tc = load_testcase("configs/testcases/single-gpu-smoke.yaml")

    def fake_kubectl(*args, **kwargs):
        for arg in args:
            if "status}" in str(arg) and "Ready" in str(arg) and "message" not in str(arg) and "reason" not in str(arg):
                return "False"
            if "reason}" in str(arg):
                return "SomeFutureUnknownError"
            if "message}" in str(arg):
                return "CRD llm-d.ai/v1 InferenceObjective not found on the cluster"
            if "containerStatuses" in str(arg):
                return ""
        return ""

    monkeypatch.setattr(d, "kubectl", fake_kubectl)
    monkeypatch.setattr("conformance.deployer.time.sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="Persistent controller error.*SomeFutureUnknownError.*not found"):
        d.wait_for_ready(tc, timeout=600, print_fn=lambda _: None)


def test_wait_for_ready_does_not_fast_fail_on_changing_reasons(monkeypatch):
    """If the error reason/message keeps changing between polls, that
    indicates the controller is making progress -- do NOT fast-fail."""
    from conformance.deployer import Deployer
    from conformance.config import load_testcase

    d = Deployer()
    tc = load_testcase("configs/testcases/single-gpu-smoke.yaml")

    poll = 0

    def fake_kubectl(*args, **kwargs):
        nonlocal poll
        for arg in args:
            if "status}" in str(arg) and "Ready" in str(arg) and "message" not in str(arg) and "reason" not in str(arg):
                return "False"
            if "reason}" in str(arg):
                poll += 1
                # Return a different reason each poll
                return f"TransientError{poll}"
            if "message}" in str(arg):
                return f"Some transient message variant {poll}"
            if "containerStatuses" in str(arg):
                return ""
        return ""

    monkeypatch.setattr(d, "kubectl", fake_kubectl)
    monkeypatch.setattr("conformance.deployer.time.sleep", lambda _: None)

    # Should time out normally, NOT raise RuntimeError
    with pytest.raises(TimeoutError, match="not ready after"):
        d.wait_for_ready(tc, timeout=0.01, print_fn=lambda _: None)


def test_wait_for_ready_includes_message_in_timeout(monkeypatch):
    """When wait_for_ready times out, the TimeoutError should include the last
    known reason and message for actionable diagnostics."""
    from conformance.deployer import Deployer
    from conformance.config import load_testcase

    d = Deployer()
    tc = load_testcase("configs/testcases/single-gpu-smoke.yaml")

    poll = 0

    def fake_kubectl(*args, **kwargs):
        nonlocal poll
        for arg in args:
            if "status}" in str(arg) and "Ready" in str(arg) and "message" not in str(arg) and "reason" not in str(arg):
                return "False"
            if "reason}" in str(arg):
                poll += 1
                return f"TransientReason{poll}"
            if "message}" in str(arg):
                return f"Controller is retrying something (attempt {poll})"
            if "containerStatuses" in str(arg):
                return ""
        return ""

    monkeypatch.setattr(d, "kubectl", fake_kubectl)
    monkeypatch.setattr("conformance.deployer.time.sleep", lambda _: None)

    with pytest.raises(TimeoutError, match=r"not ready after.*TransientReason.*Controller is retrying"):
        d.wait_for_ready(tc, timeout=0.01, print_fn=lambda _: None)


def test_wait_for_ready_logs_message_field(monkeypatch):
    """wait_for_ready should include the condition .message in progress logs."""
    from conformance.deployer import Deployer
    from conformance.config import load_testcase

    d = Deployer()
    tc = load_testcase("configs/testcases/single-gpu-smoke.yaml")

    def fake_kubectl(*args, **kwargs):
        for arg in args:
            if "status}" in str(arg) and "Ready" in str(arg) and "message" not in str(arg) and "reason" not in str(arg):
                return "False"
            if "reason}" in str(arg):
                return "SchedulerReconcileError"
            if "message}" in str(arg):
                return "Missing RBAC permissions for llm-d.ai resources"
            if "containerStatuses" in str(arg):
                return ""
        return ""

    monkeypatch.setattr(d, "kubectl", fake_kubectl)
    monkeypatch.setattr("conformance.deployer.time.sleep", lambda _: None)

    logs = []
    with pytest.raises(RuntimeError):
        d.wait_for_ready(tc, timeout=600, print_fn=logs.append)

    # Verify message field appears in log output
    assert any("message=" in line and "Missing RBAC" in line for line in logs), (
        f"Expected condition message in log lines, got: {logs}"
    )


def test_apply_with_webhook_retry_does_not_retry_unrelated_connectivity_errors(monkeypatch):
    """Connectivity substrings alone (no 'webhook' mention) must not trigger retries.

    Regression: an early version matched bare substrings like 'connection refused' or
    'eof' anywhere in the error, which could misfire on unrelated failures (e.g. a
    manifest field containing 'geofence') or mask a real, non-webhook outage behind a
    misleading 'waiting for webhook' message.
    """
    from conformance.deployer import Deployer

    d = Deployer()
    calls = []

    def fake_kubectl(*args, **kwargs):
        calls.append(args)
        raise RuntimeError("kubectl apply failed: error validating data: invalid geofence field")

    monkeypatch.setattr(d, "kubectl", fake_kubectl)
    monkeypatch.setattr("conformance.deployer.time.sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="geofence"):
        d._apply_with_webhook_retry("dummy.yaml", timeout=60, interval=1)
    assert len(calls) == 1


def test_apply_with_webhook_retry_always_attempts_once(monkeypatch):
    """Even with timeout=0, at least one apply attempt must happen."""
    from conformance.deployer import Deployer

    d = Deployer()
    calls = []

    def fake_kubectl(*args, **kwargs):
        calls.append(args)
        return ""

    monkeypatch.setattr(d, "kubectl", fake_kubectl)

    d._apply_with_webhook_retry("dummy.yaml", timeout=0, interval=1)
    assert len(calls) == 1


def test_apply_with_webhook_retry_recovers_from_crd_not_registered(monkeypatch):
    """CRD/API-version-not-registered errors (a distinct upgrade race from webhook
    readiness) must also be retried. Seen in CI as:
    'the server could not find the requested resource' right after a CRD is
    installed/updated but the client's API discovery hasn't caught up yet.
    """
    from conformance.deployer import Deployer

    d = Deployer()
    calls = []

    def fake_kubectl(*args, **kwargs):
        calls.append(args)
        if len(calls) < 2:
            raise RuntimeError("Error from server (NotFound): the server could not find the requested resource")
        return ""

    monkeypatch.setattr(d, "kubectl", fake_kubectl)
    monkeypatch.setattr("conformance.deployer.time.sleep", lambda _: None)

    d._apply_with_webhook_retry("dummy.yaml", timeout=60, interval=1)
    assert len(calls) == 2


def test_apply_with_webhook_retry_no_matches_for_kind_is_transient(monkeypatch):
    """'no matches for kind' (stale discovery cache after a CRD apply) is also transient."""
    from conformance.deployer import Deployer

    d = Deployer()
    calls = []

    def fake_kubectl(*args, **kwargs):
        calls.append(args)
        if len(calls) < 2:
            raise RuntimeError('no matches for kind "LLMInferenceService" in version "serving.kserve.io/v1alpha1"')
        return ""

    monkeypatch.setattr(d, "kubectl", fake_kubectl)
    monkeypatch.setattr("conformance.deployer.time.sleep", lambda _: None)

    d._apply_with_webhook_retry("dummy.yaml", timeout=60, interval=1)
    assert len(calls) == 2


def test_setup_manifests_removes_stale_files(tmp_path, monkeypatch):
    """Switching manifest branches must remove stale files from the previous branch.

    Regression: _setup_manifests used to copy new files on top of existing ones
    without pruning. Switching main→3.4-stable left flow-control-tokens.yaml
    behind, causing it to appear available when the 3.4 EPP would crash on it.
    """
    from unittest.mock import MagicMock, patch
    import conformance.cli as cli_mod

    monkeypatch.chdir(tmp_path)

    # Simulate manifests left over from a previous `--setup main` run
    manifest_dir = tmp_path / "deploy" / "manifests"
    manifest_dir.mkdir(parents=True)
    for stale in ["flow-control-tokens.yaml", "flow-control.yaml", "pd-performance.yaml"]:
        (manifest_dir / stale).write_text("stale: true")

    # Pre-create what `git clone` would produce for 3.4-stable
    clone_dirs = []

    def fake_run(cmd, **kwargs):
        result = MagicMock()
        result.returncode = 0
        result.stdout = "abc1234deadbeef\n"
        result.stderr = ""
        if cmd[:2] == ["git", "clone"]:
            clone_dir = Path(cmd[-1])
            clone_dir.mkdir(parents=True)
            clone_dirs.append(clone_dir)
            for new in ["single-gpu.yaml", "cache-aware.yaml"]:
                (clone_dir / new).write_text("branch: 3.4-stable")
        return result

    with patch.object(cli_mod, "subprocess") as mock_sub:
        mock_sub.run.side_effect = fake_run
        cli_mod._setup_manifests("3.4-stable")

    remaining = {f.name for f in manifest_dir.glob("*.yaml")}
    assert "flow-control-tokens.yaml" not in remaining
    assert "flow-control.yaml" not in remaining
    assert "pd-performance.yaml" not in remaining
    assert "single-gpu.yaml" in remaining
    assert "cache-aware.yaml" in remaining
    assert len(clone_dirs) == 1
    assert not clone_dirs[0].parent.exists()


def test_setup_manifests_uses_custom_repo(tmp_path, monkeypatch):
    """--manifest-repo <URL> must clone from the given repo, not the default.

    The custom URL has to reach ``git clone`` and be recorded in .manifest-ref
    so a later run can tell which fork the manifests came from.
    """
    from unittest.mock import MagicMock, patch
    import conformance.cli as cli_mod

    monkeypatch.chdir(tmp_path)

    custom_repo = "https://github.com/my-org/my-repo.git"

    clone_cmds = []
    clone_dirs = []

    def fake_run(cmd, **kwargs):
        result = MagicMock()
        result.returncode = 0
        result.stdout = "abc1234deadbeef\n"
        result.stderr = ""
        if cmd[:2] == ["git", "clone"]:
            clone_cmds.append(cmd)
            clone_dir = Path(cmd[-1])
            clone_dir.mkdir(parents=True)
            clone_dirs.append(clone_dir)
            (clone_dir / "single-gpu.yaml").write_text("branch: my-branch")
        return result

    with patch.object(cli_mod, "subprocess") as mock_sub:
        mock_sub.run.side_effect = fake_run
        cli_mod._setup_manifests("my-branch", custom_repo)

    # custom repo URL is shown in the git clone cmd.
    assert clone_cmds, "git clone was never invoked"
    assert custom_repo in clone_cmds[0]
    assert cli_mod.MANIFEST_REPO not in clone_cmds[0]

    ref_file = tmp_path / "deploy" / "manifests" / ".manifest-ref"
    assert f"repo: {custom_repo}" in ref_file.read_text()
    assert len(clone_dirs) == 1
    assert not clone_dirs[0].parent.exists()


def test_setup_manifests_defaults_to_upstream_repo(tmp_path, monkeypatch):
    """Without --manifest-repo, _setup_manifests clones the upstream default."""
    from unittest.mock import MagicMock, patch
    import conformance.cli as cli_mod

    monkeypatch.chdir(tmp_path)

    clone_cmds = []
    clone_dirs = []

    def fake_run(cmd, **kwargs):
        result = MagicMock()
        result.returncode = 0
        result.stdout = "abc1234deadbeef\n"
        result.stderr = ""
        if cmd[:2] == ["git", "clone"]:
            clone_cmds.append(cmd)
            clone_dir = Path(cmd[-1])
            clone_dir.mkdir(parents=True)
            clone_dirs.append(clone_dir)
            (clone_dir / "single-gpu.yaml").write_text("branch: main")
        return result

    with patch.object(cli_mod, "subprocess") as mock_sub:
        mock_sub.run.side_effect = fake_run
        cli_mod._setup_manifests("main")

    assert clone_cmds, "git clone was never invoked"
    assert cli_mod.MANIFEST_REPO in clone_cmds[0]
    assert len(clone_dirs) == 1
    assert not clone_dirs[0].parent.exists()


def test_require_manifest_skips_when_missing(tmp_path):
    """test_01_prereq and test_02_deploy skip when the manifest file is absent."""
    from dataclasses import dataclass

    @dataclass
    class FakeDeployConfig:
        manifest_path: str = "nonexistent.yaml"

    @dataclass
    class FakeTestCase:
        deployment: FakeDeployConfig = None

        def __post_init__(self):
            self.deployment = FakeDeployConfig()

    sys.path.insert(0, str(Path(__file__).parent))
    import test_conformance as tc_mod

    original = tc_mod._MANIFEST_DIR
    try:
        tc_mod._MANIFEST_DIR = tmp_path
        with pytest.raises(pytest.skip.Exception, match="nonexistent.yaml"):
            tc_mod._require_manifest(FakeTestCase())
    finally:
        tc_mod._MANIFEST_DIR = original


def test_require_manifest_does_not_skip_when_present(tmp_path):
    """_require_manifest should not skip when the manifest exists."""
    from dataclasses import dataclass

    @dataclass
    class FakeDeployConfig:
        manifest_path: str = "exists.yaml"

    @dataclass
    class FakeTestCase:
        deployment: FakeDeployConfig = None

        def __post_init__(self):
            self.deployment = FakeDeployConfig()

    (tmp_path / "exists.yaml").write_text("kind: LLMInferenceService")

    sys.path.insert(0, str(Path(__file__).parent))
    import test_conformance as tc_mod

    original = tc_mod._MANIFEST_DIR
    try:
        tc_mod._MANIFEST_DIR = tmp_path
        tc_mod._require_manifest(FakeTestCase())
    finally:
        tc_mod._MANIFEST_DIR = original


def test_new_testcase_script_generates_loadable_config(tmp_path, monkeypatch):
    """new-testcase.sh must produce a config YAML that load_testcase() can parse."""
    import subprocess

    import yaml

    script = Path(__file__).parent.parent / "scripts" / "new-testcase.sh"
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs" / "testcases").mkdir(parents=True)
    (tmp_path / "deploy" / "manifests").mkdir(parents=True)

    result = subprocess.run([str(script), "my-gen-test"], capture_output=True, text=True)
    assert result.returncode == 0, f"Script failed: {result.stderr}"

    config_path = tmp_path / "configs" / "testcases" / "my-gen-test.yaml"
    assert config_path.exists()

    tc = load_testcase(str(config_path))
    assert tc.name == "my-gen-test"
    assert tc.deployment.manifest_path == "my-gen-test.yaml"
    assert tc.validation.health_port == 8000
    assert tc.validation.test_prompts == ["What is 2+2?"]
    assert tc.validation.metrics_check.check_vllm is True
    assert tc.validation.metrics_check.check_scheduler is True
    assert tc.model.name == "Qwen/Qwen3-0.6B"

    manifest_path = tmp_path / "deploy" / "manifests" / "my-gen-test.yaml"
    assert manifest_path.exists()
    manifest = yaml.safe_load(manifest_path.read_text())
    assert manifest["kind"] == "LLMInferenceService"
    assert manifest["metadata"]["name"] == "my-gen-test"
    assert manifest["spec"]["replicas"] == 1


def test_wait_for_ready_surfaces_operator_image_pull_errors(monkeypatch):
    """When wait_for_ready hits a persistent error and operator pods have
    ImagePullBackOff, the error message should include the failing image."""
    from conformance.deployer import Deployer
    from conformance.config import load_testcase

    d = Deployer()
    tc = load_testcase("configs/testcases/single-gpu-smoke.yaml")

    def fake_kubectl(*args, **kwargs):
        for arg in args:
            if "status}" in str(arg) and "Ready" in str(arg) and "message" not in str(arg) and "reason" not in str(arg):
                return "False"
            if "reason}" in str(arg):
                return "SchedulerReconcileError"
            if "message}" in str(arg):
                return "failed to reconcile scheduler resources"
            if "containerStatuses" in str(arg):
                return ""
        return ""

    monkeypatch.setattr(d, "kubectl", fake_kubectl)
    monkeypatch.setattr("conformance.deployer.time.sleep", lambda _: None)
    monkeypatch.setattr(
        d,
        "_check_operator_image_issues",
        lambda: [
            "redhat-ods-applications/kserve-module-controller-manager-abc: "
            "ImagePullBackOff (quay.io/rhoai/odh-kserve-module-operator-rhel9:latest)"
        ],
    )

    logs = []
    with pytest.raises(RuntimeError, match=r"(?s)Operator image pull failures.*ImagePullBackOff.*kserve-module"):
        d.wait_for_ready(tc, timeout=600, print_fn=logs.append)

    assert any("WARNING" in line and "ImagePullBackOff" in line for line in logs), (
        f"Expected WARNING about ImagePullBackOff in logs, got: {logs}"
    )


def test_wait_for_ready_timeout_includes_image_pull_errors(monkeypatch):
    """When wait_for_ready times out (no persistent error detected), the
    TimeoutError should still include operator ImagePullBackOff info."""
    from conformance.deployer import Deployer
    from conformance.config import load_testcase

    d = Deployer()
    tc = load_testcase("configs/testcases/single-gpu-smoke.yaml")

    poll = 0

    def fake_kubectl(*args, **kwargs):
        nonlocal poll
        for arg in args:
            if "status}" in str(arg) and "Ready" in str(arg) and "message" not in str(arg) and "reason" not in str(arg):
                return "False"
            if "reason}" in str(arg):
                poll += 1
                return f"TransientReason{poll}"
            if "message}" in str(arg):
                return f"retrying attempt {poll}"
            if "containerStatuses" in str(arg):
                return ""
        return ""

    monkeypatch.setattr(d, "kubectl", fake_kubectl)
    monkeypatch.setattr("conformance.deployer.time.sleep", lambda _: None)
    monkeypatch.setattr(
        d,
        "_check_operator_image_issues",
        lambda: ["redhat-ods-applications/bad-operator-pod-xyz: ErrImagePull (quay.io/rhoai/some-private-image:v1)"],
    )

    with pytest.raises(TimeoutError, match=r"(?s)Operator image pull failures.*ErrImagePull.*some-private-image"):
        d.wait_for_ready(tc, timeout=0.01, print_fn=lambda _: None)


def test_check_operator_image_issues_returns_empty_when_healthy(monkeypatch):
    """No false positives when all operator pods are healthy."""
    from conformance.deployer import Deployer

    d = Deployer()

    def fake_kubectl(*args, **kwargs):
        return ""

    monkeypatch.setattr(d, "kubectl", fake_kubectl)
    assert d._check_operator_image_issues() == []


def test_check_operator_image_issues_parses_image_pull_back_off(monkeypatch):
    """_check_operator_image_issues correctly parses ImagePullBackOff from kubectl output."""
    from conformance.deployer import Deployer

    d = Deployer()

    def fake_kubectl(*args, **kwargs):
        joined = " ".join(str(a) for a in args)
        if "redhat-ods-applications" in joined:
            return "kserve-ctrl-abc|ImagePullBackOff=quay.io/rhoai/private-img:v1 |"
        return ""

    monkeypatch.setattr(d, "kubectl", fake_kubectl)
    issues = d._check_operator_image_issues()
    assert len(issues) == 1
    assert "ImagePullBackOff" in issues[0]
    assert "quay.io/rhoai/private-img:v1" in issues[0]
    assert "kserve-ctrl-abc" in issues[0]


def test_load_lora_single_testcase():
    """LoRA single-adapter testcase YAML should parse correctly."""
    tc = load_testcase("configs/testcases/lora-single.yaml")
    assert tc.name == "lora-single"
    assert tc.model.lora is not None
    assert len(tc.model.lora.adapters) == 1
    assert tc.model.lora.adapters[0]["name"] == "sql-adapter"
    assert tc.model.lora.adapters[0]["uri"] == "hf://edbeeching/opt-125m-lora"
    assert tc.model.lora.max_adapters == 0
    assert tc.validation.metrics_check.check_lora is True


def test_load_lora_multi_testcase():
    """LoRA multi-adapter testcase YAML should parse with all adapters and settings."""
    tc = load_testcase("configs/testcases/lora-multi.yaml")
    assert tc.name == "lora-multi"
    assert tc.model.lora is not None
    assert len(tc.model.lora.adapters) == 2
    adapter_names = [a["name"] for a in tc.model.lora.adapters]
    assert "sql-adapter" in adapter_names
    assert "code-adapter" in adapter_names
    assert tc.model.lora.max_rank == 64
    assert tc.model.lora.max_adapters == 2


def test_load_testcase_without_lora():
    """Testcase YAML without LoRA should have lora=None."""
    tc = load_testcase("configs/testcases/single-gpu-smoke.yaml")
    assert tc.model.lora is None


def test_chat_string_prompt_wraps_as_user_message(monkeypatch):
    """chat() with a plain string should wrap it as [{'role': 'user', 'content': ...}]."""
    captured = {}

    def fake_post(url, json=None, **kwargs):
        captured["json"] = json

        class FakeResp:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {"choices": [{"message": {"content": "ok"}}], "usage": {"total_tokens": 1}}

        return FakeResp()

    c = LLMClient(base_url="http://localhost:8000")
    monkeypatch.setattr(c._client, "post", fake_post)
    c.chat(model="test-model", prompt="hello")
    assert captured["json"]["messages"] == [{"role": "user", "content": "hello"}]
    c.close()


def test_chat_list_prompt_passes_through(monkeypatch):
    """chat() with a list of message dicts should pass them through unmodified."""
    captured = {}

    def fake_post(url, json=None, **kwargs):
        captured["json"] = json

        class FakeResp:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {"choices": [{"message": {"content": "ok"}}], "usage": {"total_tokens": 1}}

        return FakeResp()

    msgs = [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "hi"}]
    c = LLMClient(base_url="http://localhost:8000")
    monkeypatch.setattr(c._client, "post", fake_post)
    c.chat(model="test-model", prompt=msgs)
    assert captured["json"]["messages"] == msgs
    c.close()


def test_messages_string_prompt_wraps_as_user_message(monkeypatch):
    """messages() should wrap a string as a user message and include anthropic-version header."""
    captured = {}

    def fake_post(url, json=None, headers=None, **kwargs):
        captured["json"] = json
        captured["headers"] = headers

        class FakeResp:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {"content": [{"type": "text", "text": "ok"}], "usage": {"input_tokens": 5, "output_tokens": 1}}

        return FakeResp()

    c = LLMClient(base_url="http://localhost:8000")
    monkeypatch.setattr(c._client, "post", fake_post)
    resp = c.messages(model="test-model", prompt="hello")
    assert captured["json"]["messages"] == [{"role": "user", "content": "hello"}]
    assert captured["json"]["model"] == "test-model"
    assert captured["headers"]["anthropic-version"] == "2023-06-01"
    assert "max_tokens" in captured["json"]
    assert resp["content"][0]["text"] == "ok"
    assert resp["usage"]["output_tokens"] == 1
    c.close()


def test_responses_prompt_sends_as_input(monkeypatch):
    """responses() should send input and max_output_tokens in the request body."""
    captured = {}

    def fake_post(url, json=None, **kwargs):
        captured["url"] = url
        captured["json"] = json

        class FakeResp:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "output": [{"content": [{"type": "output_text", "text": "ok"}]}],
                    "usage": {"output_tokens": 1},
                }

        return FakeResp()

    c = LLMClient(base_url="http://localhost:8000")
    monkeypatch.setattr(c._client, "post", fake_post)
    resp = c.responses(model="test-model", prompt="hello")
    assert captured["json"]["input"] == "hello"
    assert captured["json"]["model"] == "test-model"
    assert "max_output_tokens" in captured["json"]
    assert "messages" not in captured["json"]
    assert resp["output"][0]["content"][0]["text"] == "ok"
    assert resp["usage"]["output_tokens"] == 1
    c.close()


def test_mock_manifest_uses_kserve_tls_certificate():
    """Mock inference must present the KServe-mounted certificate to the gateway."""
    from conformance.deployer import Deployer

    spec = {
        "template": {
            "containers": [{"name": "main"}],
        }
    }

    Deployer()._replace_vllm_image(spec, "simulator:latest", "test-model")

    args = spec["template"]["containers"][0]["args"]
    assert args[args.index("--ssl-certfile") + 1] == "/var/run/kserve/tls/tls.crt"
    assert args[args.index("--ssl-keyfile") + 1] == "/var/run/kserve/tls/tls.key"
    assert "--self-signed-certs" not in args


def test_env_overrides_applied_to_decode_and_prefill():
    """_patch_manifest should inject env_overrides into main containers of both templates."""
    import yaml
    from conformance.deployer import Deployer
    from conformance.config import load_testcase
    from pathlib import Path
    import tempfile

    manifest = {
        "apiVersion": "serving.kserve.io/v1alpha2",
        "kind": "LLMInferenceService",
        "metadata": {"name": "test"},
        "spec": {
            "model": {"uri": "hf://test/model", "name": "test/model"},
            "template": {
                "containers": [
                    {"name": "main", "env": [{"name": "EXISTING", "value": "keep"}]},
                ]
            },
            "prefill": {
                "replicas": 1,
                "template": {
                    "containers": [
                        {"name": "main", "env": [{"name": "EXISTING", "value": "keep"}]},
                    ]
                },
            },
        },
    }

    tc = load_testcase("configs/testcases/single-gpu-smoke.yaml")
    tc.deployment.env_overrides = {"FOO": "bar", "EXISTING": "overwritten"}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(manifest, f)
        tmp = f.name

    try:
        d = Deployer()
        result = d._patch_manifest(Path(tmp), tc)
        spec = result[0]["spec"]

        for section_name in ("template", "prefill"):
            if section_name == "prefill":
                containers = spec["prefill"]["template"]["containers"]
            else:
                containers = spec["template"]["containers"]
            main = [c for c in containers if c["name"] == "main"][0]
            env_dict = {e["name"]: e["value"] for e in main["env"]}
            assert env_dict["FOO"] == "bar", f"{section_name}: FOO not injected"
            assert env_dict["EXISTING"] == "overwritten", f"{section_name}: EXISTING not overwritten"
    finally:
        Path(tmp).unlink()


def test_new_testcase_script_rejects_duplicate(tmp_path, monkeypatch):
    """new-testcase.sh must refuse to overwrite an existing config."""
    import subprocess

    script = Path(__file__).parent.parent / "scripts" / "new-testcase.sh"
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs" / "testcases").mkdir(parents=True)
    (tmp_path / "deploy" / "manifests").mkdir(parents=True)

    subprocess.run([str(script), "dupe-test"], capture_output=True, text=True)
    result = subprocess.run([str(script), "dupe-test"], capture_output=True, text=True)
    assert result.returncode != 0
    assert "already exists" in result.stdout or "already exists" in result.stderr
