"""Unit tests for the batch-gateway phases (``tests/batch/phases.py``) — no cluster required.

Run with: ``uv run pytest tests/batch/ -v``

The phase helpers are driven from a fake ``Deployer`` so readiness polling, the
skip-versus-fail decision of ``--batch-gateway``, and the collection gating can
be exercised without kubectl or a cluster.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

import batch.phases as batch_phases
from batch.phases import (
    BATCH_ASYNC_CONDITION,
    BatchPhases,
    assert_http_error,
    async_configured,
    parse_conditions,
    require_batch_platform,
    require_batch_target,
    wait_for_batch_ready,
    wait_for_ingestion_failure,
)
from conformance.config import BatchGatewayCheck, DeployConfig, TestCase, ValidateConfig

_NAMESPACE = "llm-conformance-test"
_NAME = "batch-gateway-async-sql"


def _conditions(status: str, reason: str = "", message: str = "", async_status: str | None = None) -> list[dict]:
    """Status conditions for an LLMBatchGateway, all components in ``status``."""
    entries = [
        {"type": condition, "status": status, "reason": reason, "message": message}
        for condition in batch_phases.BATCH_READY_CONDITIONS
    ]
    entries.append({"type": "Ready", "status": status, "reason": reason, "message": message})
    if async_status is not None:
        entries.append({"type": BATCH_ASYNC_CONDITION, "status": async_status, "reason": reason, "message": message})
    return entries


class FakeDeployer:
    """Just enough of ``Deployer`` for the phase helpers."""

    def __init__(
        self,
        conditions: list[list[dict]] | None = None,
        crd: bool = True,
        operator: bool = True,
        deployed: bool = True,
        has_cr: bool = True,
        async_config: dict | None = None,
    ):
        self.namespace = _NAMESPACE
        self.crd = crd
        self.operator = operator
        self.deployed = deployed
        self.has_cr = has_cr
        self.async_config = async_config
        self._polls = [list(entry) for entry in conditions or [_conditions("True")]]
        self.calls: list[tuple] = []

    def kubectl(self, *args: str, check: bool = True) -> str:
        self.calls.append(args)
        kind = args[1] if len(args) > 1 else ""
        if kind == "llmbatchgateway":
            if not self.has_cr:
                if check:
                    raise RuntimeError("Error from server (NotFound): llmbatchgateways.batch.llm-d.ai not found")
                return "Error from server (NotFound): llmbatchgateways.batch.llm-d.ai not found"
            spec = {"processor": {"asyncConfig": self.async_config}} if self.async_config else {}
            return json.dumps({"spec": spec, "status": {"conditions": self._next_poll()}})
        if kind == "crd":
            if self.crd:
                return "llmbatchgateways.batch.llm-d.ai"
            raise RuntimeError("Error from server (NotFound): crds.apiextensions.k8s.io not found")
        if kind == "pods":
            return "Running" if self.operator else ""
        if kind == "deployment":
            if self.operator:
                return f"deployment.apps/{args[2]}"
            raise RuntimeError("Error from server (NotFound)")
        return ""

    def check_crd_exists(self, crd_name: str) -> bool:
        return self.crd

    def check_resource_exists(self, kind: str, name: str) -> bool:
        return self.has_cr

    def is_deployed(self, name: str) -> bool:
        return self.deployed

    def _next_poll(self) -> list[dict]:
        return self._polls.pop(0) if len(self._polls) > 1 else self._polls[0]


def _testcase(enabled: bool = True, manifest: str = "batch-gateway-async-sql.yaml") -> TestCase:
    return TestCase(
        name=_NAME,
        deployment=DeployConfig(manifest_path=manifest),
        validation=ValidateConfig(batch_gateway=BatchGatewayCheck(enabled=enabled)),
    )


def test_parse_conditions_maps_type_to_status_reason_message():
    raw = json.dumps(
        {
            "status": {
                "conditions": [
                    {"type": "Ready", "status": "True", "reason": "AllComponentsReady", "message": "all good"},
                    {"type": "APIServerAvailable", "status": "False", "reason": "ComponentsNotReady"},
                ]
            }
        }
    )
    conditions = parse_conditions(raw)
    assert conditions["Ready"] == ("True", "AllComponentsReady", "all good")
    assert conditions["APIServerAvailable"] == ("False", "ComponentsNotReady", "")


@pytest.mark.parametrize("raw", ["", "not json", '{"status": {}}', "{}"])
def test_parse_conditions_tolerates_absent_or_unparsable_status(raw):
    assert parse_conditions(raw) == {}


def test_wait_for_batch_ready_returns_all_true_conditions():
    deployer = FakeDeployer(conditions=[_conditions("True", async_status="True")])
    conditions = wait_for_batch_ready(deployer, _NAME, timeout=1, require_async=True, interval=0)
    assert conditions["Ready"][0] == "True"
    assert conditions[BATCH_ASYNC_CONDITION][0] == "True"


def test_wait_for_batch_ready_polls_until_components_report_true():
    deployer = FakeDeployer(
        conditions=[
            _conditions("False", "ComponentsNotReady", "apiserver starting"),
            _conditions("True"),
        ]
    )
    printed: list[str] = []
    conditions = wait_for_batch_ready(deployer, _NAME, timeout=5, interval=0, print_fn=printed.append)
    assert conditions["Ready"][0] == "True"
    assert any("ComponentsNotReady" in line for line in printed), printed


def test_wait_for_batch_ready_fails_fast_on_a_persistent_reason():
    deployer = FakeDeployer(conditions=[_conditions("False", "ComponentsNotReady", "apiserver crashloop")])
    with pytest.raises(RuntimeError, match="stuck.*ComponentsNotReady"):
        wait_for_batch_ready(deployer, _NAME, timeout=600, interval=0)


def test_wait_for_batch_ready_times_out_with_the_last_conditions():
    deployer = FakeDeployer(conditions=[_conditions("Unknown", "", "")])
    with pytest.raises(TimeoutError, match="not ready after 0.05s.*APIServerAvailable=Unknown"):
        wait_for_batch_ready(deployer, _NAME, timeout=0.05, interval=0)


def test_async_configured_reads_the_processor_async_config():
    assert async_configured(FakeDeployer(async_config={"sql": {}}), _NAME) is True
    assert async_configured(FakeDeployer(async_config=None), _NAME) is False


def test_async_configured_is_false_when_the_cr_cannot_be_read():
    assert async_configured(FakeDeployer(has_cr=False), _NAME) is False


def test_require_batch_platform_passes_when_crd_and_operator_exist():
    require_batch_platform(FakeDeployer(crd=True, operator=True), mode="require")


def test_require_batch_platform_skips_in_auto_mode_with_a_diagnostic():
    deployer = FakeDeployer(crd=False, operator=False)
    with pytest.raises(pytest.skip.Exception, match="batch gateway platform not available.*CRD.*missing"):
        require_batch_platform(deployer, mode="auto")


def test_require_batch_platform_fails_in_require_mode_with_a_diagnostic():
    deployer = FakeDeployer(crd=True, operator=False)
    with pytest.raises(pytest.fail.Exception, match="operator running=False"):
        require_batch_platform(deployer, mode="require")


def test_require_batch_target_skips_when_the_manifest_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(batch_phases, "_MANIFEST_DIR", tmp_path)
    with pytest.raises(pytest.skip.Exception, match="manifest not found for this branch"):
        require_batch_target(FakeDeployer(), _testcase(), test_mode="deploy", mode="auto")


def test_require_batch_target_skips_when_the_deploy_failed(tmp_path, monkeypatch):
    monkeypatch.setattr(batch_phases, "_MANIFEST_DIR", tmp_path)
    (tmp_path / "batch-gateway-async-sql.yaml").write_text("kind: LLMBatchGateway\n")
    with pytest.raises(pytest.skip.Exception, match="deploy failed or was skipped"):
        require_batch_target(FakeDeployer(deployed=False), _testcase(), test_mode="deploy", mode="auto")


def test_require_batch_target_ignores_the_deploy_gate_in_discover_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(batch_phases, "_MANIFEST_DIR", tmp_path)
    (tmp_path / "batch-gateway-async-sql.yaml").write_text("kind: LLMBatchGateway\n")
    require_batch_target(FakeDeployer(deployed=False), _testcase(), test_mode="discover", mode="auto")


ITEMS = (
    "test_40a_batch_ready",
    "test_40b_batch_files",
    "test_40c_batch_inference",
    "test_40d_batch_files_contract",
    "test_40e_batch_validation",
    "test_40f_batch_ingestion_rejections",
    "test_40g_batch_partial_results",
    "test_40h_batch_file_reuse",
    "test_40i_batch_cancel",
    "test_40j_batch_multi_tenant",
)
# 40k is gated on concurrency as well, so it is exercised separately.
CONCURRENCY_ITEM = "test_40k_batch_concurrency"


@pytest.mark.parametrize("method_name", ITEMS)
def test_batch_phases_are_gated_on_batch_gateway_enabled(method_name):
    marks = [mark for mark in getattr(BatchPhases, method_name).pytestmark if mark.name == "applies_when"]
    assert marks, f"{method_name} must carry an applies_when mark"
    predicate = marks[0].args[0]
    assert predicate(_testcase(enabled=True)) is True
    assert predicate(_testcase(enabled=False)) is False


def test_concurrency_phase_is_gated_on_enabled_and_a_positive_count():
    marks = [mark for mark in getattr(BatchPhases, CONCURRENCY_ITEM).pytestmark if mark.name == "applies_when"]
    predicate = marks[0].args[0]

    assert predicate(_testcase(enabled=True)) is True
    assert predicate(_testcase(enabled=False)) is False

    off = TestCase(
        name=_NAME,
        deployment=DeployConfig(manifest_path="batch-gateway-async-sql.yaml"),
        validation=ValidateConfig(batch_gateway=BatchGatewayCheck(enabled=True, concurrency=0)),
    )
    assert predicate(off) is False, "concurrency: 0 must deselect the phase, not run it"


# ---------------------------------------------------------------------------
# Submission helpers ported alongside the upstream e2e suite
# ---------------------------------------------------------------------------


def _http_error(status: int, body: dict | None = None) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://batch.test/v1/batches")
    response = httpx.Response(status, json=body if body is not None else {"error": {"message": "boom"}})
    return httpx.HTTPStatusError(f"HTTP {status}", request=request, response=response)


def test_assert_http_error_accepts_the_expected_status():
    def call():
        raise _http_error(400)

    assert_http_error(400, call, "invalid batch")


def test_assert_http_error_rejects_another_status():
    def call():
        raise _http_error(404)

    with pytest.raises(AssertionError, match="expected HTTP 400, got 404"):
        assert_http_error(400, call, "invalid batch")


def test_assert_http_error_requires_an_error_message():
    def call():
        raise _http_error(400, body={"error": {}})

    with pytest.raises(AssertionError, match="no error.message"):
        assert_http_error(400, call, "invalid batch")


def test_assert_http_error_rejects_a_successful_call():
    with pytest.raises(AssertionError, match="request succeeded"):
        assert_http_error(400, lambda: None, "invalid batch")


class _PollingClient:
    def __init__(self, statuses: list[str]):
        self._statuses = list(statuses)

    def get_batch(self, _batch_id: str) -> dict:
        status = self._statuses.pop(0) if len(self._statuses) > 1 else self._statuses[0]
        return {"id": "batch-1", "status": status, "request_counts": {"total": 1}}


def test_wait_for_ingestion_failure_returns_on_failed():
    client = _PollingClient(["validating", "in_progress", "failed"])

    batch = wait_for_ingestion_failure(client, "batch-1", timeout=5, interval=0)

    assert batch["status"] == "failed"


def test_wait_for_ingestion_failure_rejects_another_terminal_status():
    client = _PollingClient(["completed"])

    with pytest.raises(AssertionError, match="instead of 'failed'"):
        wait_for_ingestion_failure(client, "batch-1", timeout=5, interval=0)


def test_wait_for_ingestion_failure_times_out():
    client = _PollingClient(["in_progress"])

    with pytest.raises(TimeoutError, match="never failed"):
        wait_for_ingestion_failure(client, "batch-1", timeout=0.01, interval=0)


class _FakeItem:
    def __init__(self, name: str, tc: TestCase, marker=None):
        self.name = name
        self.callspec = types.SimpleNamespace(params={"tc": tc})
        self._marker = marker

    def get_closest_marker(self, name: str):
        return self._marker if name == "applies_when" else None


def test_collection_deselects_batch_phases_for_non_batch_testcases():
    """The phase markers feed ``conftest.pytest_collection_modifyitems``; non-batch cases drop them."""
    import conftest

    marker = pytest.mark.applies_when.with_args(lambda tc: tc.validation.batch_gateway.enabled)
    batch_item = _FakeItem("test_40a_batch_ready[batch]", _testcase(enabled=True), marker)
    other_item = _FakeItem("test_40a_batch_ready[single-gpu]", _testcase(enabled=False), marker)
    plain_item = _FakeItem("test_09a_inference[single-gpu]", _testcase(enabled=False), None)
    deselected: list = []
    config = types.SimpleNamespace(hook=types.SimpleNamespace(pytest_deselected=lambda items: deselected.extend(items)))
    items = [batch_item, other_item, plain_item]

    conftest.pytest_collection_modifyitems(config, items)

    assert items == [batch_item, plain_item]
    assert deselected == [other_item]
