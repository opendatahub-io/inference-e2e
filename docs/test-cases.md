# Test cases

Every test case deploys an `LLMInferenceService` and walks the same ordered
phases — deploy, service, gateway, pods, ready, health, models, inference,
metrics, cleanup — selecting phases via its config. The config
(`configs/testcases/<name>.yaml`) is the source of truth for model, prompts and
validation; the manifest (in
[llm-d-conformance-manifests](https://github.com/opendatahub-io/llm-d-conformance-manifests))
is the source of truth for topology and GPU count.

Run one with `uv run llm-d-e2e -t <name> -v`, or a group with
`-p configs/profiles/<name>.yaml`. See `docs/quick-start.md` for setup and
`docs/developing.md` for adding new cases.

Hardware columns:

- **Sim** — ✓ runs in `--mock` mode (the simulator replaces vLLM, GPU requests
  are stripped). Cases marked — set `requiresGpu` and are skipped in mock mode.
- **GPUs** — total GPUs the deployment needs, including prefill and
  parallelism-scaled worker pods.
- **RDMA** — ✓ an RDMA-class network is required; — the case works over TCP.

| Test case | Sim | GPUs | RDMA | Description |
|---|---|---|---|---|
| `agentic-serving` | — | 1 | — | Agentic serving with tool-calling (Qwen3-4B-Instruct-2507), single GPU, 1 replica |
| `cache-aware` | ✓ | 2 | — | 2 GPUs with prefix KV cache-aware routing, 2 replicas (default: Qwen/Qwen3-0.6B) |
| `flow-control-tokens` | ✓ | 1 | — | Token-based flow control with concurrency-detector, max-num-seqs=2 |
| `flow-control` | ✓ | 1 | — | Flow control with utilization-detector (default), max-num-seqs=2 |
| `kv-offloading-cpu` | — | 1 | — | KV-cache offloading to CPU memory (lru eviction) |
| `kv-offloading-tiered` | — | 1 | — | KV-cache offloading tiered to filesystem (CPU + secondary FS, lru eviction) |
| `lora-multi` | ✓ | 1 | — | Multiple LoRA adapters — validates /v1/models lists all, inference works with each |
| `lora-single` | ✓ | 1 | — | Single LoRA adapter smoke test — HF adapter with inference validation |
| `maas-single-gpu` | ✓ | 1 | — | single-gpu served through MaaS: llm-d inference checks, then MaaS auth, API key, and rate limit |
| `moe` | — | 32 | ✓ | MoE with DP/EP across 4 nodes × 8 GPUs, RDMA/RoCE (default: deepseek-ai/DeepSeek-R1-0528) |
| `multi-pool` | ✓ | 2 | — | Multiple InferencePools — validates VirtualService merge across pools (OSSM-12585) |
| `pd-performance` | — | 16 | ✓ | P/D performance benchmark — gpt-oss-120b with GuideLLM at high concurrency |
| `pd` | — | 3 | — | P/D disaggregation, 2 prefill + 1 decode (default: Qwen/Qwen3-0.6B) |
| `single-gpu-no-scheduler` | ✓ | 1 | — | 1 GPU without scheduler, K8s native routing (default: Qwen/Qwen3-0.6B) |
| `single-gpu-smoke` | ✓ | 1 | — | 1 GPU fast smoke test (default: Qwen/Qwen3-0.6B) |
| `single-gpu` | ✓ | 1 | — | 1 GPU with scheduler (default: Qwen/Qwen3-0.6B) |

When adding or changing a case, update its row here and write its section
below (see the checklist in `docs/developing.md`).

---

## agentic-serving

- Deploys Qwen3-4B-Instruct-2507 on one GPU with the scheduler in the path
- Sends a chat prompt that offers the model a `get_weather` function tool
- The response must contain a well-formed `tool_calls` entry naming the tool
  and its arguments — proving tool-calling survives the full
  gateway → EPP → vLLM round trip
- Validates vLLM and scheduler metrics alongside

**Hardware:** 1 GPU — `requiresGpu`.

## cache-aware

- Deploys two replicas behind the prefix-cache-aware scheduler plugin
- Sends three requests sharing a long system prompt — requests two and three
  should find the prefix already in vLLM's KV cache from request one
- Validates prefix-cache metrics on both the workload pods (queries, hits) and
  the EPP: non-zero queries and hits, hit rate above zero
- Proves the router steered repeat prefixes to the replica that cached them

**Hardware:** simulated or 2 GPUs.

## flow-control

- Runs Qwen3-0.6B with `max-num-seqs=2` and a utilization-based saturation
  detector in the EPP, so the backend saturates quickly
- Validates the flow-control metrics (dispatch cycles, request enqueueing)
- Proves the EPP actively queues and sheds load instead of forwarding
  everything into an overloaded vLLM

**Hardware:** simulated or 1 GPU.

## flow-control-tokens

- The same scenario as `flow-control` with the token-based concurrency
  detector instead of the utilization-based one

**Hardware:** simulated or 1 GPU.

## kv-offloading-cpu

- Runs vLLM with a CPU-memory KV-cache offload tier (LRU eviction)
- Drives 60 inferences with unique prompts to overflow the GPU cache and
  force offload to the CPU tier
- Validates the offloading metrics to prove the tier is absorbing KV pages

**Hardware:** 1 GPU — `requiresGpu`; the point of the test is real KV-cache
behavior, so it is never simulated.

## kv-offloading-tiered

- Like `kv-offloading-cpu`, with a second filesystem tier below the CPU tier
- Validates metrics for both tiers (CPU offload and filesystem offload)

**Hardware:** 1 GPU — `requiresGpu`.

## lora-single

- Serves `facebook/opt-125m` with one LoRA adapter
  (`edbeeching/opt-125m-lora`) mounted
- Verifies `/v1/models` lists the adapter alongside the base model
- Verifies inference works through the adapter, and that the LoRA request
  metrics report it

**Hardware:** simulated or 1 GPU.

## lora-multi

- Serves `facebook/opt-125m` with two LoRA adapters
- Verifies `/v1/models` lists both
- Verifies inference works with each adapter, and that the LoRA metrics track
  requests per adapter

**Hardware:** simulated or 1 GPU.

## maas-single-gpu

- Runs the `single-gpu` deployment through Models-as-a-Service: the standard
  llm-d phases first, then the MaaS phases
- Verifies the governed route end to end:
  - an unauthenticated request is rejected with 401
  - creating an API key against the model returns 201
  - an authenticated request succeeds
  - exceeding the subscription's token rate limit returns 429

**Prerequisites:** the MaaS stack — RHCL (Kuadrant: Authorino, Limitador), the
MaaS Postgres database and `maas-db-config` secret, and the RHAII chart with
`components.aigateway.modelsAsAService.enabled=true`, which creates both the
inference gateway and `maas-default-gateway`. The manifest binds the service
to both gateways and carries its `MaaSModelRef`, `MaaSAuthPolicy` and
`MaaSSubscription`. On xKS, `validation.maas.endpointScheme: http` is set
because the MaaS gateway's port 443 is not reachable externally.

**Known issues:**

- Kuadrant policies stay Pending with "Gateway API provider (istio / envoy
  gateway) is not installed" when the Kuadrant operator started before Istio —
  restart it (`kubectl delete pod -n kuadrant-operators -l
  app.kubernetes.io/name=kuadrant-operator`)
- The `maas-api-key-cleanup` CronJob fails with `CreateContainerConfigError`
  until [models-as-a-service#1627](https://github.com/opendatahub-io/models-as-a-service/pull/1627)
  ships; it does not affect these tests

**Hardware:** simulated or 1 GPU (the MaaS stack must be installed either way).

## moe

- Serves `deepseek-ai/DeepSeek-R1-0528` — a large mixture-of-experts model
- Data parallelism 32 with `dataLocal: 8` and expert parallelism: 4 pods of
  8 GPUs (32 GPUs) across 4 nodes
- Every pod attaches to the RoCE network (`roce-p2`) and runs under a
  dedicated service account (`hfsa`)
- 90-minute readiness timeout — the model is slow to load
- One inference proves the deployment comes up and answers through the full
  stack; there are no MoE-specific metrics yet

**Hardware:** 4 nodes × 8 GPUs with RDMA — `requiresGpu`.

## multi-pool

- Deploys a service whose traffic is split across two InferencePools
- Verifies the VirtualService merge across pools works (OSSM-12585): both
  pools serve requests through one route

**Hardware:** simulated or 2 GPUs.

## pd

- Prefill/decode disaggregation: one decode replica and two prefill replicas
  of Qwen3-0.6B
- Each vLLM runs the NIXL KV-transfer connector (`kv_role: kv_both`) —
  prefill hands its computed KV cache to decode instead of decode recomputing
  it
- Inference requests flow through the gateway and EPP as usual
- Validates the P/D metrics:
  - prefill and decode token counts
  - NIXL transfer count above zero (KV actually moved)
  - zero failed transfers
  - decode's locally computed prompt tokens near zero — it received them
    from prefill

**Hardware:** 3 GPUs — `requiresGpu`; NIXL runs over TCP by default
(`UCX_TLS=tcp,sm,self,cuda_copy,cuda_ipc`), so RDMA is not required. Never
simulatable — the test's whole subject is real KV transfer.

## pd-performance

- A P/D performance benchmark rather than a conformance check — and the only
  case that runs a large model
- Deploys `openai/gpt-oss-120b` across 16 GPUs (2 decode replicas at TP=4 +
  4 prefill replicas at TP=2) with RDMA
- Runs a [GuideLLM](https://github.com/vllm-project/guidellm) benchmark
  (rate 200, 240 s, long-prompt profile) and asserts thresholds:

| Check | Threshold |
|-------|-----------|
| Output tokens/s | >= 8000 |
| TTFT median / p95 | <= 2000 / 5000 ms |
| ITL median / p95 | <= 50 / 100 ms |
| Failed request ratio | <= 5% |
| NIXL transfers | > 0, with 0 failures |
| Decode KV transfer > local compute | P/D topology working |

- Thresholds live in `configs/testcases/pd-performance.yaml`
  (`validation.benchmark.thresholds`)
- The manifest mounts a 500Gi PVC (`model-cache-pvc`) that must already hold
  the model — pre-download `openai/gpt-oss-120b` into it before running this
  case
- Node placement and the GuideLLM image are overridable
  (`--decode-node-selector`, `--prefill-node-selector`, `--guidellm-image`)

Note: this test is unique in this suite and may be removed in the future.

**Hardware:** 16 GPUs with RDMA — `requiresGpu`.

## single-gpu

- Deploys one Qwen3-0.6B replica with the scheduler (EPP) in the routing path
- Adds to the smoke test a second inference prompt, plus the Anthropic
  `/v1/messages` and OpenAI `/v1/responses` endpoints
- Validates vLLM request-success counters and scheduler end-to-end duration
  and ready-endpoint gauges

**Hardware:** simulated or 1 GPU.

## single-gpu-no-scheduler

- Deploys a single Qwen3-0.6B replica with no EPP — no scheduler pod is
  deployed at all
- The HTTPRoute sends traffic straight to the workload Service with plain
  Kubernetes routing
- Proves the platform serves inference without the scheduler in the path

**Hardware:** simulated or 1 GPU.

## single-gpu-smoke

- Deploys one Qwen3-0.6B replica and waits for Ready
- Checks `/health` and `/v1/models`, completes one chat request, cleans up
- No metrics, no extra endpoints — the baseline phases only

**Hardware:** simulated or 1 GPU. The recommended first run against any new
cluster.
