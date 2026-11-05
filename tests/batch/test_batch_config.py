"""Unit tests for camelCase ``batchGateway`` config parsing — no cluster required.

Run with: ``uv run pytest tests/batch/test_batch_config.py -v``

The batch-gateway checks live under ``validation.batchGateway`` in the testcase
YAML, so these tests load a minimal testcase from ``tmp_path`` and assert the
camelCase → ``BatchGatewayCheck`` mapping, the duration fields, and the
defaults that keep every existing testcase batch-free.
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from conformance.config import (
    BatchGatewayCheck,
    load_profile,
    load_testcase,
    load_testcases_from_dir,
)

_BATCH_TESTCASE = """\
name: batch-smoke
description: batch gateway smoke
model:
  name: Qwen/Qwen3-0.6B
deployment:
  manifestPath: batch-gateway-async-sql.yaml
validation:
  testPrompts:
    - "Say hi."
  batchGateway:
    enabled: true
    endpoint: http://127.0.0.1:8000
    dispatchUrl: https://custom.example.com/ns/model
    tenant: team-a
    requestTimeout: 90s
    pollInterval: 5s
    completionWindow: 1h
    model: Qwen/Qwen3-0.6B
    prompts:
      - "One"
      - "Two"
    expectOutputs: 2
    verifyTls: false
    concurrency: 5
    cancelRequests: 4
"""


def _load_yaml(tmp_path: Path, body: str):
    path = tmp_path / "case.yaml"
    path.write_text(body)
    return load_testcase(path)


def test_batch_gateway_check_parses_camel_case_keys(tmp_path):
    tc = _load_yaml(tmp_path, _BATCH_TESTCASE)

    check = tc.validation.batch_gateway
    assert isinstance(check, BatchGatewayCheck)
    assert check.enabled is True
    assert check.endpoint == "http://127.0.0.1:8000"
    assert check.dispatch_url == "https://custom.example.com/ns/model"
    assert check.tenant == "team-a"
    assert check.request_timeout == timedelta(seconds=90)
    assert check.poll_interval == timedelta(seconds=5)
    assert check.completion_window == "1h"
    assert check.model == "Qwen/Qwen3-0.6B"
    assert check.prompts == ["One", "Two"]
    assert check.expect_outputs == 2
    assert check.verify_tls is False
    assert check.concurrency == 5
    assert check.cancel_requests == 4
    # The suite prompts remain independent of the batch prompts.
    assert tc.validation.test_prompts == ["Say hi."]


def test_batch_gateway_defaults_when_absent(tmp_path):
    tc = _load_yaml(
        tmp_path,
        """\
name: no-batch
model:
  name: Qwen/Qwen3-0.6B
deployment:
  manifestPath: single-gpu-smoke.yaml
validation:
  testPrompts:
    - "Hello"
""",
    )

    check = tc.validation.batch_gateway
    assert check == BatchGatewayCheck()
    assert check.enabled is False
    assert check.endpoint == ""
    assert check.dispatch_url == "", "an unset dispatchUrl must leave the deployer's derivation in charge"
    assert check.tenant == "default"
    assert check.request_timeout == timedelta(minutes=10)
    assert check.poll_interval == timedelta(seconds=15)
    assert check.completion_window == "24h"
    assert check.model == ""
    assert check.prompts == []
    assert check.expect_outputs == 0
    assert check.verify_tls is True


def test_empty_batch_gateway_block_uses_the_defaults(tmp_path):
    tc = _load_yaml(
        tmp_path,
        """\
name: empty-batch
deployment:
  manifestPath: x.yaml
validation:
  batchGateway:
""",
    )

    assert tc.validation.batch_gateway == BatchGatewayCheck()


def test_partial_batch_gateway_block_keeps_the_other_defaults(tmp_path):
    tc = _load_yaml(
        tmp_path,
        """\
name: partial-batch
deployment:
  manifestPath: x.yaml
validation:
  batchGateway:
    enabled: true
    prompts:
      - "Only prompt"
""",
    )

    check = tc.validation.batch_gateway
    assert check.enabled is True
    assert check.prompts == ["Only prompt"]
    assert check.expect_outputs == 0
    assert check.request_timeout == timedelta(minutes=10)
    assert check.poll_interval == timedelta(seconds=15)
    assert check.verify_tls is True


def test_concurrency_defaults_to_three_and_accepts_zero(tmp_path):
    default = _load_yaml(
        tmp_path,
        """\
name: defaults
validation:
  batchGateway:
    enabled: true
""",
    ).validation.batch_gateway
    assert default.concurrency == 3
    assert default.cancel_requests == 8

    off = _load_yaml(
        tmp_path,
        """\
name: no-concurrency
validation:
  batchGateway:
    enabled: true
    concurrency: 0
""",
    ).validation.batch_gateway
    assert off.concurrency == 0, "explicit 0 must survive parsing (it deselects phase 40k)"


def test_existing_testcases_do_not_enable_the_batch_gateway():
    """Only the batch profile's testcase may opt in; every existing case stays batch-free."""
    batch_manifests = {"batch-gateway-async-sql.yaml"}

    for tc in load_testcases_from_dir("configs/testcases"):
        if tc.deployment.manifest_path in batch_manifests:
            continue
        assert tc.validation.batch_gateway.enabled is False, (
            f"{tc.name} must not enable batchGateway; add a dedicated batch testcase instead"
        )


def test_the_shipped_batch_testcase_parses_with_batch_enabled():
    """Guards the silent-zero-coverage path: a typo in the YAML key would disable all 11 phases.

    ``_build`` ignores keys it does not know, so ``batchGateways:`` (plural), a
    misspelling, or a wrong nesting level leaves ``enabled`` at its default.
    Nothing else notices: the cluster run simply deselects every batch phase and
    reports success, so this assertion is the only thing standing between a
    green run and zero batch coverage.
    """
    cases = {case.name: case for case in load_testcases_from_dir("configs/testcases")}
    batch = cases["batch-gateway-async-sql"]

    check = batch.validation.batch_gateway
    assert check.enabled is True, "the shipped batch testcase must enable validation.batchGateway"
    assert check.concurrency > 0, "phase 40k needs a positive concurrency"
    assert check.cancel_requests > 0, "phase 40i needs cancelRequests"
    assert batch.validation.test_prompts, "phase 40c builds its batch from testPrompts"
    assert batch.deployment.manifest_path == "batch-gateway-async-sql.yaml"


def test_the_batch_profile_selects_the_batch_testcase():
    """A typo in the profile name is the other silent way to run zero batch coverage."""
    profile = load_profile("configs/profiles/batch-gateway.yaml")

    assert profile.test_cases == ["batch-gateway-async-sql"]
