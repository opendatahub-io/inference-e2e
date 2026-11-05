"""Unit tests for batch phase collection and the ``--batch-gateway`` auto/require decision.

Run with: ``uv run pytest tests/batch/test_batch_collection.py -v``

No cluster is touched: the platform probe runs against a fake deployer, the
collection hook against fake pytest items, and the CLI/conftest wiring against
an in-process parser. These tests pin down three behaviours:

- batch phases are deselected for every testcase that does not enable
  ``validation.batchGateway`` (non-batch collection is unchanged),
- ``--batch-gateway auto`` turns a missing platform into a skip with a
  diagnostic, while ``require`` turns the same condition into a failure, and
- the flag exists in both ``conformance.cli`` and ``tests/conftest.py``.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from batch.phases import BatchPhases, require_batch_platform
from conformance.batch import BATCH_CRD, BATCH_OPERATOR_DEPLOYMENT, batch_sql_transport_supported
from conformance.config import BatchGatewayCheck, DeployConfig, TestCase, ValidateConfig

_BATCH_METHODS = (
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
    "test_40k_batch_concurrency",
)


# ---------------------------------------------------------------------------
# Platform probe: auto skips, require fails
# ---------------------------------------------------------------------------


class _Platform:
    """Just enough of ``Deployer`` for ``require_batch_platform``."""

    def __init__(self, crd: bool = True, operator: bool = True):
        self.crd = crd
        self.operator = operator

    def check_crd_exists(self, crd_name: str) -> bool:
        return self.crd

    def kubectl(self, *args: str, check: bool = True) -> str:
        kind = args[1] if len(args) > 1 else ""
        if kind == "pods":
            return "Running" if self.operator else "Pending"
        if kind == "crd":
            if self.crd:
                return BATCH_CRD
            raise RuntimeError("Error from server (NotFound): crds.apiextensions.k8s.io not found")
        if kind == "deployment":
            if self.operator:
                return f"deployment.apps/{BATCH_OPERATOR_DEPLOYMENT}"
            raise RuntimeError("Error from server (NotFound): deployments.apps not found")
        if kind == "datasciencecluster":
            return "Managed"
        raise AssertionError(f"unexpected kubectl call: {args}")


class _CrdSchemaPlatform(_Platform):
    """``_Platform`` that also answers the CRD schema query the transport probe makes."""

    def __init__(self, transports: tuple[str, ...] = ("redis", "sql")):
        super().__init__()
        self.transports = transports
        self.schema_queries = 0

    def kubectl(self, *args: str, check: bool = True) -> str:
        if args[:2] == ("get", "crd") and "-o" in args:
            self.schema_queries += 1
            properties = {name: {"type": "object"} for name in self.transports}
            async_config = {"properties": properties}
            processor = {"properties": {"asyncConfig": async_config}}
            schema = {"properties": {"spec": {"properties": {"processor": processor}}}}
            return json.dumps({"spec": {"versions": [{"schema": {"openAPIV3Schema": schema}}]}})
        return super().kubectl(*args, check=check)


def test_sql_transport_probe_reads_the_installed_crd_schema():
    """RHOAI 3.6-EA2's CRD has asyncConfig.redis only; 3.6 GA adds asyncConfig.sql."""
    assert batch_sql_transport_supported(_CrdSchemaPlatform(("redis", "sql")).kubectl) is True
    assert batch_sql_transport_supported(_CrdSchemaPlatform(("redis",)).kubectl) is False


def test_sql_transport_probe_assumes_supported_when_the_crd_is_unreadable():
    """A parse failure must not hide a deployment the platform may well support."""

    def unreadable(*args: str, check: bool = True) -> str:
        return "llmbatchgateways.batch.llm-d.ai"

    assert batch_sql_transport_supported(unreadable) is True


def test_auto_mode_skips_when_the_operator_lacks_the_sql_transport():
    platform = _CrdSchemaPlatform(("redis",))

    with pytest.raises(pytest.skip.Exception) as excinfo:
        require_batch_platform(platform, "auto", "sql")

    message = str(excinfo.value)
    assert "does not support the SQL async transport" in message
    assert "spec.processor.asyncConfig.sql" in message
    assert "RHOAI 3.6 GA" in message


def test_require_mode_fails_when_the_operator_lacks_the_sql_transport():
    with pytest.raises(pytest.fail.Exception) as excinfo:
        require_batch_platform(_CrdSchemaPlatform(("redis",)), "require", "sql")

    assert "does not support the SQL async transport" in str(excinfo.value)


def test_a_sql_capable_operator_passes_the_transport_gate():
    platform = _CrdSchemaPlatform(("redis", "sql"))

    require_batch_platform(platform, "require", "sql")  # must not raise

    assert platform.schema_queries == 1


def test_no_transport_declared_skips_the_schema_query():
    platform = _CrdSchemaPlatform(("redis",))

    require_batch_platform(platform, "auto")  # must not raise

    assert platform.schema_queries == 0


def test_auto_mode_skips_when_the_platform_lacks_the_batch_gateway():
    with pytest.raises(pytest.skip.Exception) as excinfo:
        require_batch_platform(_Platform(crd=False, operator=False), "auto")

    message = str(excinfo.value)
    assert "batch gateway platform not available" in message
    assert f"CRD {BATCH_CRD}=missing" in message
    assert "operator running=False" in message
    # The diagnostic from batch_platform_state() is part of the skip reason.
    assert f"CRD {BATCH_CRD} not found" in message


def test_require_mode_fails_with_the_same_diagnostic():
    with pytest.raises(pytest.fail.Exception) as excinfo:
        require_batch_platform(_Platform(crd=True, operator=False), "require")

    message = str(excinfo.value)
    assert "batch gateway platform not available" in message
    assert f"CRD {BATCH_CRD}=present" in message
    assert "operator running=False" in message
    assert "managementState=Managed" in message


@pytest.mark.parametrize("mode", ["auto", "require"])
def test_platform_probe_passes_when_crd_and_operator_exist(mode):
    require_batch_platform(_Platform(), mode)  # must not raise


# ---------------------------------------------------------------------------
# Collection: batch phases only for batch testcases
# ---------------------------------------------------------------------------


def _testcase(enabled: bool) -> TestCase:
    return TestCase(
        name="batch-gateway-async-sql" if enabled else "single-gpu",
        deployment=DeployConfig(manifest_path="batch-gateway-async-sql.yaml" if enabled else "single-gpu.yaml"),
        validation=ValidateConfig(batch_gateway=BatchGatewayCheck(enabled=enabled)),
    )


def _applies_when_mark(method_name: str):
    marks = [mark for mark in getattr(BatchPhases, method_name).pytestmark if mark.name == "applies_when"]
    assert len(marks) == 1, f"{method_name} must carry exactly one applies_when mark"
    return marks[0]


class _FakeItem:
    """Minimal pytest item: a name, a ``tc`` callspec param, and one optional marker."""

    def __init__(self, name: str, tc: TestCase, marker=None):
        self.name = name
        self.callspec = types.SimpleNamespace(params={"tc": tc})
        self._marker = marker

    def get_closest_marker(self, name: str):
        return self._marker if name == "applies_when" else None


def _run_collection_hook(items: list) -> list:
    """Run the real ``conftest`` hook and return the deselected items."""
    import conftest

    deselected: list = []
    config = types.SimpleNamespace(hook=types.SimpleNamespace(pytest_deselected=lambda items: deselected.extend(items)))
    conftest.pytest_collection_modifyitems(config, items)
    return deselected


@pytest.mark.parametrize("method_name", _BATCH_METHODS)
def test_batch_phases_are_gated_by_a_predicate_not_the_method_itself(method_name):
    predicate = _applies_when_mark(method_name).args[0]

    assert callable(predicate)
    assert predicate is not getattr(BatchPhases, method_name)
    assert predicate(_testcase(enabled=True)) is True
    assert predicate(_testcase(enabled=False)) is False


def test_concurrency_zero_deselects_only_the_concurrency_phase():
    no_concurrency = TestCase(
        name="batch-gateway-async-sql",
        deployment=DeployConfig(manifest_path="batch-gateway-async-sql.yaml"),
        validation=ValidateConfig(batch_gateway=BatchGatewayCheck(enabled=True, concurrency=0)),
    )
    items = [
        _FakeItem(f"{method_name}[batch]", no_concurrency, _applies_when_mark(method_name))
        for method_name in _BATCH_METHODS
    ]

    deselected = _run_collection_hook(items)

    assert [item.name for item in deselected] == ["test_40k_batch_concurrency[batch]"]
    assert len(items) == len(_BATCH_METHODS) - 1


def test_non_batch_testcase_deselects_every_batch_phase_and_nothing_else():
    non_batch = _testcase(enabled=False)
    batch_items = [
        _FakeItem(f"{method_name}[single-gpu]", non_batch, _applies_when_mark(method_name))
        for method_name in _BATCH_METHODS
    ]
    unchanged = [_FakeItem("test_09a_inference[single-gpu]", non_batch)]
    items = batch_items + unchanged

    deselected = _run_collection_hook(items)

    assert items == unchanged
    assert deselected == batch_items


def test_batch_testcase_keeps_every_batch_phase():
    batch_tc = _testcase(enabled=True)
    items = [
        _FakeItem(f"{method_name}[batch-gateway-async-sql]", batch_tc, _applies_when_mark(method_name))
        for method_name in _BATCH_METHODS
    ]
    original = list(items)

    assert _run_collection_hook(items) == []
    assert items == original


# ---------------------------------------------------------------------------
# CLI and conftest wiring for --batch-gateway
# ---------------------------------------------------------------------------


def test_cli_forwards_an_explicit_batch_gateway_mode(monkeypatch):
    import conformance.cli as cli

    commands: list[list[str]] = []
    monkeypatch.setattr(sys, "argv", ["llm-d-e2e", "-t", "single-gpu", "--batch-gateway", "require"])
    monkeypatch.setattr(cli, "_run_pytest", lambda cmd: commands.append(cmd) or 0)

    with pytest.raises(SystemExit) as result:
        cli.main()

    assert result.value.code == 0
    assert "--batch-gateway" in commands[0]
    assert commands[0][commands[0].index("--batch-gateway") + 1] == "require"


def test_cli_defaults_to_auto(monkeypatch):
    import conformance.cli as cli

    commands: list[list[str]] = []
    monkeypatch.setattr(sys, "argv", ["llm-d-e2e", "-t", "single-gpu"])
    monkeypatch.setattr(cli, "_run_pytest", lambda cmd: commands.append(cmd) or 0)

    with pytest.raises(SystemExit):
        cli.main()

    assert commands[0][commands[0].index("--batch-gateway") + 1] == "auto"


def test_conftest_option_defaults_to_auto_and_accepts_require_only():
    import conftest

    parser = argparse.ArgumentParser()
    parser.addoption = parser.add_argument  # pytest's hook uses addoption; argparse calls it add_argument
    conftest.pytest_addoption(parser)

    assert parser.parse_args([]).batch_gateway == "auto"
    assert parser.parse_args(["--batch-gateway", "require"]).batch_gateway == "require"
    with contextlib.redirect_stderr(io.StringIO()), pytest.raises(SystemExit):
        parser.parse_args(["--batch-gateway", "sometimes"])
