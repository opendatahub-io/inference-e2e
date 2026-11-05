"""Unit tests for the phase 02 batch-platform gate — no cluster required.

Run with: ``uv run pytest tests/batch/test_deploy_gate.py -v``

A batch testcase's manifest contains an ``LLMBatchGateway`` document, so applying
it on a platform without that CRD fails with "no matches for kind" halfway
through the manifest. Phase 02 therefore probes the platform first and reuses
``require_batch_platform()``: ``--batch-gateway auto`` skips with the platform
diagnostic, ``require`` fails fast with the same message, and a testcase without
``validation.batchGateway`` deploys exactly as before.

The phase method is driven against a fake ``Deployer`` so the decision — and the
fact that nothing is applied when the gate fires — is exercised without kubectl.
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).parents[1]))

import test_conformance as tc_mod
from conformance.batch import BATCH_CRD, BATCH_OPERATOR_DEPLOYMENT
from conformance.config import BatchGatewayCheck, DeployConfig, TestCase, ValidateConfig

_MANIFEST = "batch-gateway-async-sql.yaml"
_BATCH_CRD_DIAGNOSTIC = "batch gateway platform not available"


class _Result:
    """Stand-in for ``DeployResult``."""

    success = True
    error = ""
    duration = 0.0


class _Deployer:
    """Just enough of ``Deployer`` for ``LLMDPhases.test_02_deploy``."""

    def __init__(self, llmisvc_crd: bool = True, batch_crd: bool = True, operator: bool = True):
        self.namespace = "llm-conformance-test"
        self.llmisvc_crd = llmisvc_crd
        self.batch_crd = batch_crd
        self.operator = operator
        self.deploy_calls: list[str] = []
        self.kubectl_calls: list[tuple] = []

    def check_crd_exists(self, name: str) -> bool:
        return self.llmisvc_crd if name == tc_mod.LLMISVC_CRD else self.batch_crd

    def kubectl(self, *args: str, check: bool = True) -> str:
        self.kubectl_calls.append(args)
        kind = args[1] if len(args) > 1 else ""
        if kind == "pods":
            return "Running" if self.operator else "Pending"
        if kind == "crd":
            if self.batch_crd:
                return BATCH_CRD
            raise RuntimeError("Error from server (NotFound): crds.apiextensions.k8s.io not found")
        if kind == "deployment":
            raise RuntimeError("Error from server (NotFound): deployments.apps not found")
        if kind == "datasciencecluster":
            return "Disabled"
        raise AssertionError(f"unexpected kubectl call: {args}")

    def deploy(self, tc: TestCase) -> _Result:
        self.deploy_calls.append(tc.name)
        return _Result()


def _testcase(batch: bool, manifest: str = _MANIFEST, transport: str = "sql") -> TestCase:
    return TestCase(
        name="batch-gateway-async-sql" if batch else "single-gpu",
        deployment=DeployConfig(manifest_path=manifest),
        validation=ValidateConfig(batch_gateway=BatchGatewayCheck(enabled=batch, transport=transport)),
    )


@pytest.fixture
def manifest_dir(tmp_path, monkeypatch) -> Path:
    """Point phase 02's manifest gate at a temporary directory holding ``_MANIFEST``."""
    (tmp_path / _MANIFEST).write_text("kind: LLMInferenceService\n")
    monkeypatch.setattr(tc_mod, "_MANIFEST_DIR", tmp_path)
    return tmp_path


def _run_phase(deployer: _Deployer, tc: TestCase, batch_gateway_mode: str) -> None:
    """Run phase 02 exactly as pytest would, minus the fixtures."""
    tc_mod.LLMDPhases().test_02_deploy(
        deployer=deployer,
        tc=tc,
        test_mode="deploy",
        mock_mode=True,
        batch_gateway_mode=batch_gateway_mode,
    )


# ---------------------------------------------------------------------------
# Batch testcases: auto skips, require fails, both before applying anything
# ---------------------------------------------------------------------------


def test_auto_mode_skips_a_batch_testcase_on_a_platform_without_the_batch_gateway(manifest_dir):
    deployer = _Deployer(batch_crd=False, operator=False)

    with pytest.raises(pytest.skip.Exception) as excinfo:
        _run_phase(deployer, _testcase(batch=True), "auto")

    message = str(excinfo.value)
    assert _BATCH_CRD_DIAGNOSTIC in message
    assert f"CRD {BATCH_CRD}=missing" in message
    assert "operator running=False" in message
    # The diagnostic from batch_platform_state() is part of the skip reason.
    assert f"CRD {BATCH_CRD} not found" in message
    # Nothing was applied: the gate runs before deploy, not after the CRD rejection.
    assert deployer.deploy_calls == []


def test_auto_mode_skips_when_the_crd_exists_but_the_operator_is_not_running(manifest_dir):
    deployer = _Deployer(batch_crd=True, operator=False)

    with pytest.raises(pytest.skip.Exception) as excinfo:
        _run_phase(deployer, _testcase(batch=True), "auto")

    message = str(excinfo.value)
    assert f"CRD {BATCH_CRD}=present" in message
    assert "operator running=False" in message
    assert f"operator Deployment {BATCH_OPERATOR_DEPLOYMENT} not found" in message
    assert deployer.deploy_calls == []


def test_phase_02_skips_when_the_operator_lacks_the_declared_transport(manifest_dir):
    """3.6-EA2's CRD has no asyncConfig.sql, so the SQL manifest would be rejected at apply."""

    class _NoSqlDeployer(_Deployer):
        def kubectl(self, *args: str, check: bool = True) -> str:
            if args[:2] == ("get", "crd") and "-o" in args:
                schema = {
                    "properties": {
                        "spec": {
                            "properties": {
                                "processor": {
                                    "properties": {"asyncConfig": {"properties": {"redis": {"type": "object"}}}}
                                }
                            }
                        }
                    }
                }
                return json.dumps({"spec": {"versions": [{"schema": {"openAPIV3Schema": schema}}]}})
            return super().kubectl(*args, check=check)

    deployer = _NoSqlDeployer()

    with pytest.raises(pytest.skip.Exception) as excinfo:
        _run_phase(deployer, _testcase(batch=True), "auto")

    assert "does not support the SQL async transport" in str(excinfo.value)
    assert deployer.deploy_calls == []


def test_phase_02_does_not_check_the_transport_when_the_case_declares_none(manifest_dir):
    """A case that declares no transport (older operator) must not be gated on one."""

    class _NoSqlDeployer(_Deployer):
        def kubectl(self, *args: str, check: bool = True) -> str:
            if args[:2] == ("get", "crd") and "-o" in args:
                raise AssertionError("no transport declared must not query the CRD schema")
            return super().kubectl(*args, check=check)

    deployer = _NoSqlDeployer()

    _run_phase(deployer, _testcase(batch=True, transport=""), "auto")  # must not raise

    assert deployer.deploy_calls == ["batch-gateway-async-sql"]


def test_require_mode_fails_a_batch_testcase_fast_with_the_same_diagnostic(manifest_dir):
    deployer = _Deployer(batch_crd=False, operator=False)

    with pytest.raises(pytest.fail.Exception) as excinfo:
        _run_phase(deployer, _testcase(batch=True), "require")

    message = str(excinfo.value)
    assert _BATCH_CRD_DIAGNOSTIC in message
    assert f"CRD {BATCH_CRD}=missing" in message
    assert "managementState=Disabled" in message
    assert deployer.deploy_calls == []


@pytest.mark.parametrize("mode", ["auto", "require"])
def test_batch_testcase_deploys_when_the_platform_supports_the_batch_gateway(manifest_dir, mode):
    deployer = _Deployer()

    _run_phase(deployer, _testcase(batch=True), mode)  # must not raise

    assert deployer.deploy_calls == ["batch-gateway-async-sql"]


# ---------------------------------------------------------------------------
# Non-batch testcases: the gate must not probe or skip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["auto", "require"])
def test_non_batch_testcase_deploys_even_without_batch_platform_support(manifest_dir, mode):
    deployer = _Deployer(batch_crd=False, operator=False)

    _run_phase(deployer, _testcase(batch=False), mode)  # must not raise

    assert deployer.deploy_calls == ["single-gpu"]
    # The gate is skipped entirely for testcases that do not enable batch validation.
    assert deployer.kubectl_calls == []


def test_discover_mode_still_skips_before_the_gate(manifest_dir):
    deployer = _Deployer(batch_crd=False, operator=False)

    with pytest.raises(pytest.skip.Exception, match="discover mode"):
        tc_mod.LLMDPhases().test_02_deploy(
            deployer=deployer,
            tc=_testcase(batch=True),
            test_mode="discover",
            mock_mode=True,
            batch_gateway_mode="auto",
        )

    assert deployer.deploy_calls == []
    assert deployer.kubectl_calls == []


def test_phase_02_requests_the_batch_gateway_mode_fixture():
    """Phase 02 must take ``batch_gateway_mode`` so ``auto``/``require`` reaches the gate."""
    params = inspect.signature(tc_mod.TestConformance.test_02_deploy).parameters
    assert "batch_gateway_mode" in params
