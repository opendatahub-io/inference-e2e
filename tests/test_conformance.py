"""LLM-D conformance test suite.

Each test case runs through ordered phases (phases skip when disabled in
config, when the manifest is missing, or when deploy failed / discover mode):

  01. Prerequisites — CRD exists; manifest file present for branch
  02. Deploy — apply LLMInferenceService manifest (skipped in discover mode)
  03. Service — wait for Service creation
  04. Gateway — wait for Gateway to be programmed
  05. Pods — wait for pods to be Running (CrashLoopBackOff early fail)
  06. Ready — wait for LLMInferenceService Ready=True
  07. Health — GET /health returns 200 (direct pod; bypasses gateway EPP)
  08. Models — GET /v1/models lists base model (+ LoRA adapters if configured)
  09a. Inference — chat/completions (+ LoRA adapter inference if configured)
  09b. Messages/Responses — Anthropic /v1/messages + OpenAI /v1/responses
  09c. Tool calling — chatPrompts with tools; validates tool_calls in response
  10. Metrics (vLLM) — scrape workload pods; validate_vllm_basic
  11. Metrics (cache) — prefix KV cache hits (vLLM + EPP; soft-fail in --mock)
  12. Metrics (P/D) — prefill/decode token distribution
  13. Metrics (scheduler) — EPP processed-request metrics
  14. Metrics (flow control) — EPP dispatch activity
  15. Metrics (LoRA) — adapter state on vLLM pods
  20. Benchmark — GuideLLM Job; optional warmup; performance thresholds
  21. Metrics (post-benchmark) — re-validate P/D after load
  30a-e. MaaS — governed route checks (tests/maas/phases.py; skip without a MaaSModelRef)
  99. Cleanup — delete LLMInferenceService (honors --nocleanup / tc.cleanup)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from conformance.benchmark import run_benchmark
from conformance.config import TestCase, chat_prompt_to_messages
from conformance.client import LLMClient
from conformance.deployer import Deployer
from conformance.metrics import (
    Scraper,
    dump_raw_metrics,
    validate_cache_aware,
    validate_flow_control,
    validate_kvcache_offloading_cpu,
    validate_kvcache_offloading_fs,
    validate_lora,
    validate_pd,
    validate_scheduler,
    validate_vllm_basic,
)
from maas.phases import MaaSPhases
from maas.praxis_phases import PraxisPhases

LLMISVC_CRD = "llminferenceservices.serving.kserve.io"
_MANIFEST_DIR = Path(__file__).resolve().parent.parent / "deploy" / "manifests"


def _log(msg: str, capsys=None):
    print(f"  → {msg}")


def _require_manifest(tc: TestCase) -> None:
    if not (_MANIFEST_DIR / tc.deployment.manifest_path).exists():
        pytest.skip(f"manifest not found for this branch: {tc.deployment.manifest_path}")


def _require_deployed(deployer: Deployer, tc: TestCase, test_mode: str) -> None:
    if test_mode == "discover":
        return
    if not deployer.is_deployed(tc.name):
        pytest.skip(f"skipped — deploy failed or was skipped for '{tc.name}'")


def _target_service_names(deployer: Deployer, tc: TestCase, test_mode: str) -> list[str]:
    """Services to validate: every manifest service, or only those the existing target deployed in discover mode."""
    if test_mode == "discover":
        return deployer.existing_service_names(tc)
    return deployer.manifest_service_names(tc)


def _require_gpu(deployer: Deployer, tc: TestCase, mock_mode: bool, test_mode: str) -> None:
    if test_mode == "discover":
        return
    if mock_mode:
        if tc.deployment.requires_gpu:
            pytest.skip(f"'{tc.name}' has requiresGpu set — cannot be validated in mock mode")
        return
    needed = deployer.manifest_gpu_needed(tc)
    if needed <= 0:
        return
    available = deployer.cluster_gpu_count()
    if available < needed:
        pytest.skip(f"'{tc.name}' needs {needed} GPU(s) but cluster has {available}")


def _metric(flag: str):
    """Phase applies when metrics checks are enabled and ``flag`` is set on ``metricsCheck``."""
    return lambda tc: tc.validation.metrics_check.enabled and getattr(tc.validation.metrics_check, flag)


def _tool_prompts(tc: TestCase) -> list[dict]:
    return [e for e in (tc.validation.chat_prompts or []) if isinstance(e, dict) and e.get("tools")]


def _check_threshold(name: str, value: float, min_value: float | None = None, max_value: float | None = None) -> bool:
    passed = True
    if min_value is not None and value < min_value:
        _log(f"FAIL: {name} = {value:.4f} < minimum {min_value}")
        passed = False
    if max_value is not None and value > max_value:
        _log(f"FAIL: {name} = {value:.4f} > maximum {max_value}")
        passed = False
    if passed:
        _log(f"PASS: {name} = {value:.4f}")
    return passed


class LLMDPhases:
    """Ordered llm-d conformance phases for each test case."""

    def test_01_prereq(self, deployer: Deployer, tc: TestCase, mock_mode: bool, test_mode: str):
        """LLMInferenceService CRD must be installed and manifest must exist."""
        _require_manifest(tc)
        _require_gpu(deployer, tc, mock_mode, test_mode)
        found = deployer.check_crd_exists(LLMISVC_CRD)
        _log(f"CRD {LLMISVC_CRD}: {'found' if found else 'NOT FOUND'}")
        if not found:
            image_issues = deployer._check_operator_image_issues()
            if image_issues:
                _log("Operator image pull failures detected:")
                for issue in image_issues:
                    _log(f"  {issue}")
                assert False, (
                    f"CRD {LLMISVC_CRD} not found — operator pods have image pull failures:\n  "
                    + "\n  ".join(image_issues)
                )
            assert False, f"CRD {LLMISVC_CRD} not found"

    def test_02_deploy(self, deployer: Deployer, tc: TestCase, test_mode: str, mock_mode: bool):
        """Deploy the LLMInferenceService manifest."""
        if test_mode == "discover":
            pytest.skip("discover mode — skipping deploy")
        _require_manifest(tc)
        _require_gpu(deployer, tc, mock_mode, test_mode)
        if not deployer.check_crd_exists(LLMISVC_CRD):
            pytest.skip(f"CRD {LLMISVC_CRD} not found — cannot deploy")
        _log(f"Deploying {tc.deployment.manifest_path} as '{tc.name}'")
        result = deployer.deploy(tc)
        _log(f"Deploy {'succeeded' if result.success else 'FAILED'} in {result.duration:.1f}s")
        if result.error:
            _log(f"Error: {result.error}")
        assert result.success, f"Deploy failed: {result.error}"

    def test_03_service(self, deployer: Deployer, tc: TestCase, test_mode: str):
        """Service should be created for the LLMInferenceService."""
        _require_deployed(deployer, tc, test_mode)
        _log(f"Waiting for Service for '{tc.name}'...")
        svc = deployer.wait_for_service(tc.name)
        _log(f"Service found: {svc}")

    def test_04_gateway(self, deployer: Deployer, tc: TestCase, test_mode: str):
        """Gateway should be programmed with an address."""
        _require_deployed(deployer, tc, test_mode)
        _log("Waiting for Gateway to be programmed...")
        addr = deployer.wait_for_gateway()
        _log(f"Gateway address: {addr}")
        assert addr, "Gateway has no address"

    def test_05_pods(self, deployer: Deployer, tc: TestCase, test_mode: str):
        """Pods should be Running without crashes."""
        _require_deployed(deployer, tc, test_mode)
        timeout = tc.deployment.ready_timeout.total_seconds()
        _log(f"Waiting for pods to be Running (timeout: {timeout:.0f}s)")
        service_names = _target_service_names(deployer, tc, test_mode)
        expected = deployer.manifest_replicas(tc, service_names)
        pods = deployer.wait_for_pods(service_names, timeout=timeout, print_fn=_log, min_pods=expected)
        _log(f"All pods running: {', '.join(pods)}")
        assert len(pods) >= expected, f"Expected {expected} pods, got {len(pods)}"

    def test_06_ready(self, deployer: Deployer, tc: TestCase, test_mode: str):
        """LLMInferenceService should become Ready."""
        _require_deployed(deployer, tc, test_mode)
        for service_name in _target_service_names(deployer, tc, test_mode):
            _log(f"Waiting for '{service_name}' Ready=True")
            deployer.wait_for_ready(tc, print_fn=_log, service_name=service_name)
            _log(f"'{service_name}' is Ready")

    def test_07_health(self, pod_client: LLMClient, tc: TestCase, pod_endpoint: str):
        """Health endpoint should return 200 (direct pod access, bypasses gateway EPP)."""
        max_retries = max(tc.validation.retry_attempts, 10)
        interval = tc.validation.retry_interval.total_seconds() or 15
        _log(f"Pod endpoint: {pod_endpoint}")
        _log(f"Checking health (up to {max_retries} attempts, {interval:.0f}s interval)")
        for attempt in range(max_retries):
            try:
                pod_client.health_check()
                _log(f"Health check PASSED (attempt {attempt + 1})")
                return
            except Exception as e:
                err_short = str(e).split("\n")[0][:120]
                _log(f"Attempt {attempt + 1}/{max_retries}: {err_short}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(interval)

    def test_08_models(self, pod_client: LLMClient, tc: TestCase):
        """Model should be listed in /v1/models (direct pod access, bypasses gateway EPP)."""
        resp = pod_client.list_models()
        models = [m["id"] for m in resp.get("data", [])]
        _log(f"Models listed: {models}")
        assert models, "No models returned"
        assert tc.model.name in models, f"Expected model {tc.model.name!r} in /v1/models response, got {models}"
        if tc.model.lora and tc.model.lora.adapters:
            for adapter in tc.model.lora.adapters:
                name = adapter["name"]
                assert name in models, f"LoRA adapter {name!r} not in /v1/models, got {models}"
            _log(f"All {len(tc.model.lora.adapters)} LoRA adapter(s) registered")

    @pytest.mark.applies_when.with_args(lambda tc: tc.validation.inference_check)
    def test_09a_inference(self, client: LLMClient, tc: TestCase):
        """Inference via /v1/chat/completions and /v1/completions."""

        def _assert_chat_response(resp: dict, label: str):
            choices = resp.get("choices", [])
            assert choices, f"No choices returned for {label}"
            content = choices[0].get("message", {}).get("content") or ""
            tokens = resp.get("usage", {}).get("total_tokens", 0)
            _log(f"Response: '{content[:80]}...' ({tokens} tokens)")
            assert content or tokens > 0, f"Empty response for {label}"
            assert tokens > 0, f"No tokens generated for {label}"

        plain_prompts = tc.validation.test_prompts or ["What is 2+2?"]
        repeat = max(1, tc.validation.inference_repeat)
        if repeat > 1:
            # Unique prefix per iteration so prefix-caching wont work
            plain_prompts = [f"({i}) {p}" for i in range(repeat) for p in plain_prompts]
            _log(f"Repeating prompts x{repeat} → {len(plain_prompts)} requests")

        plain_chat_prompts = [
            entry for entry in tc.validation.chat_prompts if not (isinstance(entry, dict) and entry.get("tools"))
        ]
        if plain_chat_prompts:
            for entry in plain_chat_prompts:
                messages = chat_prompt_to_messages(entry)
                assert messages, f"chatPrompts entry has no 'system' or 'user' key: {entry}"
                _log(f"Sending chat prompt ({len(messages)} message(s)): '{messages[-1]['content'][:50]}...'")
                resp = client.chat(model=tc.model.name, prompt=messages)
                _assert_chat_response(resp, "chat prompt")
        else:
            for prompt in plain_prompts:
                _log(f"Sending prompt: '{prompt[:50]}...'")
                resp = client.chat(model=tc.model.name, prompt=prompt)
                _assert_chat_response(resp, f"prompt: {prompt}")

        for prompt in plain_prompts:
            _log(f"Sending completions prompt: '{prompt[:50]}...'")
            resp = client.completions(model=tc.model.name, prompt=prompt)
            choices = resp.get("choices", [])
            assert choices, f"/v1/completions: no choices for prompt: {prompt}"
            assert resp.get("usage", {}).get("total_tokens", 0) > 0, "/v1/completions: no tokens generated"

        if tc.model.lora and tc.model.lora.adapters:
            for adapter in tc.model.lora.adapters:
                adapter_name = adapter["name"]
                _log(f"Sending LoRA inference with adapter '{adapter_name}'...")
                resp = client.chat(model=adapter_name, prompt="Test LoRA adapter inference")
                _assert_chat_response(resp, f"LoRA adapter: {adapter_name}")

    @pytest.mark.applies_when.with_args(lambda tc: tc.validation.check_messages or tc.validation.check_responses)
    def test_09b_messages_responses(self, client: LLMClient, tc: TestCase):
        """Inference via Anthropic /v1/messages and OpenAI /v1/responses."""

        plain_prompts = tc.validation.test_prompts or ["What is 3+3?"]

        if tc.validation.check_messages:
            _log("Testing Anthropic /v1/messages endpoint...")
            for prompt in plain_prompts:
                _log(f"Sending /v1/messages prompt: '{prompt[:50]}...'")
                resp = client.messages(model=tc.model.name, prompt=prompt)
                content_blocks = resp.get("content", [])
                assert content_blocks, f"/v1/messages: no content for prompt: {prompt}"
                text = content_blocks[0].get("text", "")
                in_tokens = resp.get("usage", {}).get("input_tokens", 0)
                out_tokens = resp.get("usage", {}).get("output_tokens", 0)
                _log(f"/v1/messages response: '{text[:80]}...' ({in_tokens}+{out_tokens} tokens)")
                assert text or out_tokens > 0, f"/v1/messages: empty response for prompt: {prompt}"
                assert out_tokens > 0, f"/v1/messages: no output tokens for prompt: {prompt}"

        if tc.validation.check_responses:
            _log("Testing OpenAI /v1/responses endpoint...")
            for prompt in plain_prompts:
                _log(f"Sending /v1/responses prompt: '{prompt[:50]}...'")
                resp = client.responses(model=tc.model.name, prompt=prompt)
                output = resp.get("output", [])
                assert output, f"/v1/responses: no output for prompt: {prompt}"
                output_text = ""
                for item in output:
                    for part in item.get("content", []):
                        if part.get("type") == "output_text":
                            output_text += part.get("text", "")
                tokens = resp.get("usage", {}).get("output_tokens", 0)
                _log(f"/v1/responses response: '{output_text[:80]}...' ({tokens} tokens)")
                assert output_text or tokens > 0, f"/v1/responses: empty response for prompt: {prompt}"
                assert tokens > 0, f"/v1/responses: no output tokens for prompt: {prompt}"

    @pytest.mark.applies_when.with_args(lambda tc: tc.validation.inference_check and _tool_prompts(tc))
    def test_09c_tool_calling(self, client: LLMClient, tc: TestCase):
        """Tool-calling: chatPrompts with tools should return structured tool_calls."""
        tool_entries = _tool_prompts(tc)

        for entry in tool_entries:
            messages = chat_prompt_to_messages(entry)
            tools = entry["tools"]
            defined_tools = {}
            for tool in tools:
                if not isinstance(tool, dict) or tool.get("type") != "function":
                    continue
                function = tool.get("function")
                name = function.get("name") if isinstance(function, dict) else None
                assert name, f"Tool definition must include function.name: {tool}"
                defined_tools[name] = function

            label = f"'{messages[-1]['content'][:50]}...'"
            _log(f"Sending tool-call prompt ({len(tools)} tool(s)): {label}")

            resp = client.chat(
                model=tc.model.name,
                prompt=messages,
                tools=tools,
                max_tokens=tc.validation.tool_call_max_tokens,
            )

            choice = resp.get("choices", [{}])[0]
            message = choice.get("message", {})
            tool_calls = message.get("tool_calls")
            finish = choice.get("finish_reason")
            tokens = resp.get("usage", {}).get("total_tokens", 0)
            assert tokens > 0, f"No tokens generated for {label}"
            assert tool_calls, (
                f"Expected tool_calls for {label} but got none "
                f"(finish_reason={finish}, content={str(message.get('content', ''))[:80]})"
            )
            called_names = [
                tc_item.get("function", {}).get("name") if isinstance(tc_item, dict) else None for tc_item in tool_calls
            ]
            _log(f"Tool calls: {called_names} (finish_reason={finish})")
            for tool_call in tool_calls:
                function = tool_call.get("function") if isinstance(tool_call, dict) else None
                function = function if isinstance(function, dict) else {}
                name = function.get("name")
                assert name in defined_tools, f"Model called unknown tool '{name}', defined: {set(defined_tools)}"

                raw_arguments = function.get("arguments")
                try:
                    arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
                except json.JSONDecodeError as exc:
                    pytest.fail(f"Tool '{name}' returned invalid JSON arguments: {exc}")
                assert isinstance(arguments, dict), f"Tool '{name}' arguments must be a JSON object"

                parameters = defined_tools[name].get("parameters", {})
                required = parameters.get("required", []) if isinstance(parameters, dict) else []
                missing = [key for key in required if key not in arguments]
                assert not missing, f"Tool '{name}' is missing required arguments: {missing}"
                empty = [key for key in required if isinstance(arguments[key], str) and not arguments[key].strip()]
                assert not empty, f"Tool '{name}' has empty required arguments: {empty}"

    @pytest.mark.applies_when.with_args(_metric("check_vllm"))
    def test_10_metrics_vllm(self, deployer: Deployer, scraper: Scraper, tc: TestCase, test_mode: str):
        """vLLM metrics should show successful requests."""
        _require_deployed(deployer, tc, test_mode)
        _log("Scraping vLLM metrics...")
        results = scraper.scrape_vllm(tc.name)
        _log(f"Scraped {len(results)} pod(s)")
        assert results, "No vLLM metrics scraped"
        checks = validate_vllm_basic(results)
        for c in checks:
            _log(f"  {c.name}: {'PASS' if c.passed else 'FAIL'} — {c.message}")
        failed = [c for c in checks if not c.passed]
        assert not failed, f"vLLM metric checks failed: {[c.message for c in failed]}"

    @pytest.mark.applies_when.with_args(_metric("check_prefix_cache"))
    def test_11_metrics_cache(
        self, deployer: Deployer, scraper: Scraper, tc: TestCase, mock_mode: bool, test_mode: str
    ):
        """Prefix cache metrics should show hits."""
        _require_deployed(deployer, tc, test_mode)
        deployer.ensure_metrics_rbac(tc.name)
        _log("Scraping cache-aware metrics...")
        vllm = scraper.scrape_vllm(tc.name)
        epp = scraper.scrape_epp(tc.name)
        checks = validate_cache_aware(vllm, epp)
        for c in checks:
            _log(f"  {c.name}: {'PASS' if c.passed else 'FAIL'} — {c.message}")
        if mock_mode:
            hit_checks = {"prefix_hits_aggregate", "prefix_hit_rate"}
            hard_fail = [c for c in checks if not c.passed and c.name not in hit_checks]
            soft_fail = [c for c in checks if not c.passed and c.name in hit_checks]
            if soft_fail:
                _log(
                    f"WARN: {len(soft_fail)} cache hit metric(s) zero — expected in mock (EPP has no real KV state for routing)"
                )
            assert not hard_fail, f"Cache metric checks failed: {[c.message for c in hard_fail]}"
        else:
            failed = [c for c in checks if not c.passed]
            assert not failed, f"Cache metric checks failed: {[c.message for c in failed]}"

    @pytest.mark.applies_when.with_args(_metric("check_pd"))
    def test_12_metrics_pd(self, deployer: Deployer, scraper: Scraper, tc: TestCase, test_mode: str, request):
        """P/D metrics should show token distribution."""
        _require_deployed(deployer, tc, test_mode)
        _log("Scraping P/D metrics...")
        vllm = scraper.scrape_vllm(tc.name)
        prefill = scraper.scrape_prefill(tc.name)
        report_dir = request.config.getoption("--report-dir")
        metrics_dir = f"{report_dir}/metrics-pre-benchmark"
        paths = dump_raw_metrics(vllm, metrics_dir, label="decode")
        paths += dump_raw_metrics(prefill, metrics_dir, label="prefill")
        for p in paths:
            _log(f"  saved {p}")
        checks = validate_pd(vllm, prefill)
        for c in checks:
            _log(f"  {c.name}: {'PASS' if c.passed else 'FAIL'} — {c.message}")
        failed = [c for c in checks if not c.passed]
        assert not failed, f"P/D metric checks failed: {[c.message for c in failed]}"

    @pytest.mark.applies_when.with_args(_metric("check_scheduler"))
    def test_13_metrics_scheduler(self, deployer: Deployer, scraper: Scraper, tc: TestCase, test_mode: str, request):
        """Scheduler/EPP metrics should show processed requests."""
        _require_deployed(deployer, tc, test_mode)
        deployer.ensure_metrics_rbac(tc.name)
        _log("Scraping scheduler/EPP metrics...")
        epp = scraper.scrape_epp(tc.name)
        _log(f"Scraped {len(epp)} EPP pod(s)")
        assert epp, "No EPP metrics scraped"
        report_dir = request.config.getoption("--report-dir")
        paths = dump_raw_metrics(epp, f"{report_dir}/metrics", label="epp")
        for p in paths:
            _log(f"  saved {p}")
        checks = validate_scheduler(epp)
        for c in checks:
            _log(f"  {c.name}: {'PASS' if c.passed else 'FAIL'} — {c.message}")
        failed = [c for c in checks if not c.passed]
        assert not failed, f"Scheduler metric checks failed: {[c.message for c in failed]}"

    @pytest.mark.applies_when.with_args(_metric("check_flow_control"))
    def test_14_metrics_flow_control(self, deployer: Deployer, scraper: Scraper, tc: TestCase, test_mode: str):
        """Flow control metrics should show dispatch activity."""
        _require_deployed(deployer, tc, test_mode)
        deployer.ensure_metrics_rbac(tc.name)
        _log("Scraping flow control metrics...")
        epp = scraper.scrape_epp(tc.name)
        _log(f"Scraped {len(epp)} EPP pod(s)")
        assert epp, "No EPP metrics scraped"
        checks = validate_flow_control(epp)
        for c in checks:
            _log(f"  {c.name}: {'PASS' if c.passed else 'FAIL'} — {c.message}")
        failed = [c for c in checks if not c.passed]
        assert not failed, f"Flow control metric checks failed: {[c.message for c in failed]}"

    @pytest.mark.applies_when.with_args(_metric("check_lora"))
    def test_15_metrics_lora(self, deployer: Deployer, scraper: Scraper, tc: TestCase, test_mode: str):
        """LoRA metrics should show adapter state on vLLM pods."""
        _require_deployed(deployer, tc, test_mode)
        _log("Scraping LoRA metrics from vLLM pods...")
        vllm = scraper.scrape_vllm(tc.name)
        _log(f"Scraped {len(vllm)} pod(s)")
        assert vllm, "No vLLM metrics scraped"
        checks = validate_lora(vllm)
        for c in checks:
            _log(f"  {c.name}: {'PASS' if c.passed else 'FAIL'} — {c.message}")
        failed = [c for c in checks if not c.passed]
        assert not failed, f"LoRA metric checks failed: {[c.message for c in failed]}"

    @pytest.mark.applies_when.with_args(_metric("check_kvcache_offloading"))
    def test_16_metrics_kvcache_offloading(self, deployer: Deployer, scraper: Scraper, tc: TestCase, test_mode: str):
        """KV-cache offloading metrics should show bytes offloaded on vLLM pods."""
        _require_deployed(deployer, tc, test_mode)
        _log("Scraping KV-cache offloading metrics from vLLM pods...")
        vllm = scraper.scrape_vllm(tc.name)
        _log(f"Scraped {len(vllm)} pod(s)")
        assert vllm, "No vLLM metrics scraped"
        checks = validate_kvcache_offloading_cpu(vllm)
        for c in checks:
            _log(f"  {c.name}: {'PASS' if c.passed else 'FAIL'} — {c.message}")
        failed = [c for c in checks if not c.passed]
        assert not failed, f"KV-cache offloading metric checks failed: {[c.message for c in failed]}"

    @pytest.mark.applies_when.with_args(_metric("check_kvcache_offloading_fs"))
    def test_17_kvcache_offloading_fs(self, deployer: Deployer, tc: TestCase, test_mode: str):
        _require_deployed(deployer, tc, test_mode)
        path = tc.validation.kv_offload_fs_path
        pods = deployer.list_workload_pods(tc.name)
        assert pods, f"No workload pods found for '{tc.name}'"
        _log(f"Checking FS offload disk usage under {path} on {len(pods)} pod(s)...")
        usage_by_pod = {pod: deployer.pod_disk_usage(pod, path) for pod in pods}
        checks = validate_kvcache_offloading_fs(usage_by_pod, path, tc.validation.kv_offload_fs_min_bytes)
        for c in checks:
            _log(f"  {c.name}: {'PASS' if c.passed else 'FAIL'} — {c.message}")
        failed = [c for c in checks if not c.passed]
        assert not failed, f"KV-cache FS offload checks failed: {[c.message for c in failed]}"

    @pytest.mark.applies_when.with_args(lambda tc: tc.validation.benchmark.enabled)
    def test_20_benchmark(self, deployer: Deployer, tc: TestCase, test_mode: str, guidellm_image: str):
        """Run GuideLLM benchmark and check performance thresholds."""
        _require_deployed(deployer, tc, test_mode)
        bc = tc.validation.benchmark

        gateway_addr = deployer.wait_for_gateway(timeout=60)
        target_url = f"http://{gateway_addr}/{deployer.namespace}/{tc.name}"
        image = guidellm_image or bc.image

        if bc.warmup_rate > 0:
            _log(f"[WARMUP] Running warmup benchmark (rate={bc.warmup_rate}, max_seconds={bc.warmup_max_seconds})")
            warmup = run_benchmark(
                kubectl_fn=deployer.kubectl,
                namespace=deployer.namespace,
                target_url=target_url,
                model=tc.model.name,
                image=image,
                rate=bc.warmup_rate,
                max_seconds=bc.warmup_max_seconds,
                data=bc.data,
                backend_type=bc.backend_type,
                request_type=bc.request_type,
                timeout=bc.timeout.total_seconds(),
                job_name="guidellm-warmup",
                print_fn=lambda msg: _log(f"[WARMUP] {msg}"),
            )
            _log(
                f"[WARMUP] done — {warmup.completed_requests}/{warmup.total_requests} completed, "
                f"otps={warmup.output_tokens_per_second:.1f}"
            )

        _log(f"Running GuideLLM benchmark against {target_url}")
        _log(f"rate={bc.rate}, max_seconds={bc.max_seconds}, image={image}")

        result = run_benchmark(
            kubectl_fn=deployer.kubectl,
            namespace=deployer.namespace,
            target_url=target_url,
            model=tc.model.name,
            image=image,
            rate=bc.rate,
            max_seconds=bc.max_seconds,
            data=bc.data,
            backend_type=bc.backend_type,
            request_type=bc.request_type,
            timeout=bc.timeout.total_seconds(),
            print_fn=_log,
        )

        assert result.completed_requests > 0, (
            f"No requests completed. total={result.total_requests}, failed={result.failed_requests}"
        )

        t = bc.thresholds
        failures = []
        if not _check_threshold(
            "output tokens/s", result.output_tokens_per_second, min_value=t.min_output_tokens_per_second
        ):
            failures.append(f"output tokens/s={result.output_tokens_per_second:.1f} < {t.min_output_tokens_per_second}")
        if not _check_threshold("TTFT median (ms)", result.ttft_median, max_value=t.max_ttft_median_ms):
            failures.append(f"TTFT median={result.ttft_median:.1f}ms > {t.max_ttft_median_ms}ms")
        if not _check_threshold("TTFT p95 (ms)", result.ttft_p95, max_value=t.max_ttft_p95_ms):
            failures.append(f"TTFT p95={result.ttft_p95:.1f}ms > {t.max_ttft_p95_ms}ms")
        if not _check_threshold("ITL median (ms)", result.itl_median, max_value=t.max_itl_median_ms):
            failures.append(f"ITL median={result.itl_median:.1f}ms > {t.max_itl_median_ms}ms")
        if not _check_threshold("ITL p95 (ms)", result.itl_p95, max_value=t.max_itl_p95_ms):
            failures.append(f"ITL p95={result.itl_p95:.1f}ms > {t.max_itl_p95_ms}ms")
        failed_ratio = result.failed_requests / max(result.total_requests, 1)
        if not _check_threshold("failed request ratio", failed_ratio, max_value=t.max_failed_ratio):
            failures.append(f"failed ratio={failed_ratio:.4f} > {t.max_failed_ratio}")
        assert not failures, f"Performance thresholds breached: {'; '.join(failures)}"

    @pytest.mark.applies_when.with_args(lambda tc: _metric("check_pd")(tc) and tc.validation.benchmark.enabled)
    def test_21_metrics_post_benchmark(
        self, deployer: Deployer, scraper: Scraper, tc: TestCase, test_mode: str, request
    ):
        """Scrape and validate P/D metrics after benchmark run."""
        _require_deployed(deployer, tc, test_mode)
        _log("Scraping post-benchmark P/D metrics...")
        decode = scraper.scrape_vllm(tc.name)
        prefill = scraper.scrape_prefill(tc.name)
        epp = scraper.scrape_epp(tc.name)
        report_dir = request.config.getoption("--report-dir")
        metrics_dir = f"{report_dir}/metrics-post-benchmark"
        paths = dump_raw_metrics(decode, metrics_dir, label="decode")
        paths += dump_raw_metrics(prefill, metrics_dir, label="prefill")
        paths += dump_raw_metrics(epp, metrics_dir, label="epp")
        for p in paths:
            _log(f"  saved {p}")
        checks = validate_pd(decode, prefill)
        for c in checks:
            _log(f"  {c.name}: {'PASS' if c.passed else 'FAIL'} — {c.message}")
        failed = [c for c in checks if not c.passed]
        assert not failed, f"Post-benchmark P/D metric checks failed: {[c.message for c in failed]}"


class CleanupPhase:
    """Final phase, collected after every other phase group."""

    @pytest.mark.applies_when.with_args(lambda tc: tc.cleanup)
    def test_99_cleanup(self, deployer: Deployer, tc: TestCase, no_cleanup: bool, test_mode: str):
        """Clean up deployed resources."""
        if no_cleanup:
            pytest.skip("--nocleanup set")
        if test_mode != "discover" and not deployer.needs_cleanup(tc.name):
            pytest.skip(f"nothing to clean up — no manifest was applied for '{tc.name}'")
        _log(f"Cleaning up '{tc.name}'...")
        deployer.cleanup(tc)
        _log("Cleanup complete")


class TestConformance(CleanupPhase, PraxisPhases, MaaSPhases, LLMDPhases):
    """Every phase for each test case.

    pytest collects inherited test methods in reverse-MRO order, so the phase
    groups run llm-d (01–21), then MaaS (30a–30e), then Praxis 3.6-only
    (30p–30q), then cleanup (99).
    """
