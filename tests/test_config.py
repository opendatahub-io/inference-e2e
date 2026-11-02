"""Unit tests for config loading, duration parsing, and YAML handling.

Run with: ``uv run pytest tests/test_config.py -v``
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from conformance.config import (
    chat_prompt_to_messages,
    iter_config_yamls,
    load_profile,
    load_profiles_from_dir,
    load_testcase,
    load_testcases_from_dir,
    parse_duration,
)
from conformance.metrics import ScrapeResult, parse_prometheus, validate_flow_control, validate_scheduler


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


def test_load_agentic_serving_testcase():
    """agentic-serving testcase should parse with tools in chatPrompts preserved as dicts."""
    tc = load_testcase("configs/testcases/agentic-serving.yaml")
    assert tc.name == "agentic-serving"
    assert tc.model.name == "Qwen/Qwen3-1.7B"
    assert tc.deployment.requires_gpu is True
    assert tc.validation.inference_check is True
    assert tc.validation.tool_call_max_tokens == 1024
    assert tc.validation.metrics_check.check_vllm is True
    assert tc.validation.metrics_check.check_scheduler is True
    assert len(tc.validation.chat_prompts) == 1
    entry = tc.validation.chat_prompts[0]
    assert isinstance(entry, dict)
    assert "tools" in entry
    assert entry["tools"][0]["function"]["name"] == "get_weather"


def test_chat_prompt_to_messages_with_tools():
    """chat_prompt_to_messages should extract system/user from entries that also have tools."""
    entry = {
        "system": "You are a helpful assistant.",
        "user": "What is the weather?",
        "tools": [{"type": "function", "function": {"name": "get_weather"}}],
    }
    msgs = chat_prompt_to_messages(entry)
    assert len(msgs) == 2
    assert msgs[0] == {"role": "system", "content": "You are a helpful assistant."}
    assert msgs[1] == {"role": "user", "content": "What is the weather?"}


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
