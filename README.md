# inference-e2e

End-to-end conformance tests for [llm-d](https://github.com/llm-d) / KServe
`LLMInferenceService` deployments on Kubernetes. Each test case deploys an
`LLMInferenceService` and walks ordered phases — deploy, readiness, health,
inference, metrics, cleanup — against a real cluster, or in mock mode with a
vLLM simulator and no GPUs.

Docs: [quick start](docs/quick-start.md) · [test cases](docs/test-cases.md) ·
[developing](docs/developing.md) · [troubleshooting](docs/troubleshooting.md)

## Quick start

```bash
uv sync
uv run llm-d-e2e --setup main
uv run llm-d-e2e -t single-gpu-smoke -v
```

Prerequisites, run modes, mock mode, authentication, reports, and running in a
container: [docs/quick-start.md](docs/quick-start.md).

## Test cases

```bash
uv run llm-d-e2e --list-testcases
```

```text
Test cases:
  agentic-serving              [1 GPU]  Agentic serving with tool-calling (Qwen3-4B-Instruct-2507), single GPU, 1 replica
  cache-aware                  [2 GPU]  2 GPUs with prefix KV cache-aware routing, 2 replicas (default: Qwen/Qwen3-0.6B)
  flow-control-tokens          [1 GPU]  Token-based flow control with concurrency-detector, max-num-seqs=2
  flow-control                 [1 GPU]  Flow control with utilization-detector (default), max-num-seqs=2
  kv-offloading-cpu            [1 GPU]  KV-cache offloading to CPU memory (lru eviction)
  kv-offloading-tiered         [1 GPU]  KV-cache offloading tiered to filesystem (CPU + secondary FS, lru eviction)
  lora-multi                   [1 GPU]  Multiple LoRA adapters — validates /v1/models lists all, inference works with each
  lora-single                  [1 GPU]  Single LoRA adapter smoke test — HF adapter with inference validation
  maas-single-gpu              [1 GPU]  single-gpu served through MaaS: llm-d inference checks, then MaaS auth, API key, and rate limit
  moe                          [8 GPU]  MoE with DP/EP across 4 nodes × 8 GPUs, RDMA/RoCE (default: deepseek-ai/DeepSeek-R1-0528)
  multi-pool                   [2 GPU]  Multiple InferencePools — validates VirtualService merge across pools (OSSM-12585)
  pd-performance               [16 GPU]  P/D performance benchmark — gpt-oss-120b with GuideLLM at high concurrency
  pd                           [3 GPU]  P/D disaggregation, 2 prefill + 1 decode (default: Qwen/Qwen3-0.6B)
  single-gpu-no-scheduler      [1 GPU]  1 GPU without scheduler, K8s native routing (default: Qwen/Qwen3-0.6B)
  single-gpu-smoke             [1 GPU]  1 GPU fast smoke test (default: Qwen/Qwen3-0.6B)
  single-gpu                   [1 GPU]  1 GPU with scheduler (default: Qwen/Qwen3-0.6B)
```

GPU counts come from the manifests, so they appear once `--setup` has run.
What every case deploys and verifies — hardware requirements, run commands,
and per-case notes — is documented in [docs/test-cases.md](docs/test-cases.md).
Select cases by name (`-t single-gpu,pd`) or run a profile
(`-p configs/profiles/smoke.yaml`).

## Development

Adding test cases, conformance phases, CLI flags, and metrics validators is
covered step-by-step in [docs/developing.md](docs/developing.md). CI runs lint
and unit tests on every PR.
