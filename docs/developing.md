# Developing test cases

How to add a test case, conformance phase, config field, CLI flag, or metrics
validator. For running the suite see `docs/quick-start.md`; for what each
existing case does see `docs/test-cases.md`.

## Choose your path

| Path | Best for | Jump to |
|---|---|---|
| **Quick start (generator)** | A new test case with a standard config — boilerplate generated for you | [Generator](#generator-scriptsnew-testcasesh) |
| **Manual setup** | Custom topology, or learning how the pieces fit | [Manual setup](#manual-setup) |
| **New conformance phase** | Validating something new, not just a new case | [Conformance phases](#conformance-phases) |
| **New metrics check** | New Prometheus metrics validation | [Adding a metrics validator](#adding-a-metrics-validator) |

## The two parts of a test case

1. **Test case config** — `configs/testcases/<name>.yaml`: model, prompts,
   validation criteria, which metrics checks to enable. Keys are camelCase;
   they map to snake_case dataclass fields in `src/conformance/config.py`.
2. **LLMInferenceService manifest** — the Kubernetes manifest applied to the
   cluster. It lives in
   [llm-d-conformance-manifests](https://github.com/opendatahub-io/llm-d-conformance-manifests)
   (branch `main` for the latest, version branches like `3.5-GA` for older
   trains) and is cloned locally into `deploy/manifests/` by
   `uv run llm-d-e2e --setup <branch>`.

The manifest owns the deployed topology — replicas, GPUs, scheduler config;
the config owns what to validate against it. GPU counts in docs and skips are
read from the manifest, not the config.

## Generator: `scripts/new-testcase.sh`

```bash
./scripts/new-testcase.sh my-new-test
```

Creates `configs/testcases/my-new-test.yaml` (config) and
`deploy/manifests/my-new-test.yaml` (manifest template to move to the
manifests repo). It refuses to overwrite existing files.

1. Edit the config: set `description`, adjust model/resources, and enable the
   metrics checks that match your topology (table below).
2. Edit the manifest: add your scheduler/EPP config under
   `spec.router.scheduler.config.inline`; copy from an existing manifest
   (`cache-aware.yaml`, `flow-control.yaml`) for non-trivial topologies.
3. Move the manifest to the manifests repo — always add to `main` first, then
   cherry-pick to version branches if compatible (see
   [Manifest branches](#manifest-branches)).
4. Pull and verify:

```bash
uv run llm-d-e2e --setup main          # or: --setup my-branch --manifest-repo https://github.com/<you>/llm-d-conformance-manifests.git
uv run llm-d-e2e --list-testcases      # ✓ my-new-test → my-new-test.yaml
uv run llm-d-e2e -t my-new-test --mock -v
```

## Manual setup

### 1. Create the manifest

In the manifests repo, starting from `single-gpu-smoke.yaml`:

```yaml
apiVersion: serving.kserve.io/v1alpha2
kind: LLMInferenceService
metadata:
  name: my-test-case
spec:
  model:
    uri: hf://Qwen/Qwen3-0.6B
    name: Qwen/Qwen3-0.6B
  replicas: 1
  router:
    scheduler:
      template:
        imagePullSecrets:
        - name: rhai-pull-secret
        containers:
        - name: main
    route: {}
    gateway: {}
  template:
    imagePullSecrets:
    - name: rhai-pull-secret
    containers:
      - name: main
        resources:
          limits:
            cpu: '2'
            memory: 8Gi
            nvidia.com/gpu: "1"
          requests:
            cpu: '1'
            memory: 4Gi
            nvidia.com/gpu: "1"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
            scheme: HTTPS
          initialDelaySeconds: 60
          periodSeconds: 30
          timeoutSeconds: 30
          failureThreshold: 5
```

### 2. Create the config

`configs/testcases/<name>.yaml`, minimal:

```yaml
name: my-test-case
description: "Short description of what this test validates"
model:
  name: Qwen/Qwen3-0.6B
  uri: hf://Qwen/Qwen3-0.6B
  displayName: qwen3-0.6b-my-test
  category: single-node-gpu
deployment:
  manifestPath: my-test-case.yaml
  readyTimeout: 10m
validation:
  healthEndpoint: /health
  healthPort: 8000
  healthScheme: HTTPS
  inferenceCheck: true
  testPrompts:
    - "What is 2+2?"
  expectedCodes: [200]
  timeout: 2m
  retryAttempts: 3
  retryInterval: 15s
cleanup: true
```

Key fields:

| Field | Purpose |
|---|---|
| `name` | Must match the filename (without `.yaml`) |
| `deployment.manifestPath` | Manifest filename in `deploy/manifests/` |
| `deployment.readyTimeout` | How long to wait for Ready=True (`10m`, `2h`) |
| `deployment.requiresGpu` | `true` = never simulated: skipped in `--mock` mode |
| `deployment.prefill` | Prefill section for P/D topologies (resources) |
| `validation.testPrompts` | Prompts sent to `/v1/chat/completions` and `/v1/completions` |
| `validation.chatPrompts` | Structured `[{system, user}]` prompts (prefix-cache tests) or tool-calling prompts |
| `validation.checkMessages` / `checkResponses` | Also exercise `/v1/messages` / `/v1/responses` |
| `validation.benchmark` | GuideLLM benchmark with thresholds (see `pd-performance`) |
| `cleanup` | Delete the deployment after the run |

### Metrics checks

Enable only what your topology proves:

| Flag | Phase | Validates | Enable for |
|---|---|---|---|
| `checkVLLM` | 10 | vLLM request-success counters | Any workload |
| `checkPrefixCache` | 11 | Prefix KV cache queries/hits, hit rate | Cache-aware routing |
| `checkPD` | 12, 21 | P/D token distribution, NIXL transfers | Prefill/decode split |
| `checkScheduler` | 13 | EPP end-to-end duration, ready endpoints | Any topology with an EPP |
| `checkFlowControl` | 14 | Flow-control dispatch/enqueue activity | Flow-control EPP |
| `checkLora` | 15 | LoRA adapter requests per adapter | LoRA adapters |
| `checkKvcacheOffloading` | 16 | KV offload to CPU tier | KV-cache offloading |
| `checkKvcacheOffloadingFs` | 17 | KV offload to filesystem tier | Tiered offloading |
| `checkNIXL` | — | Reserved (no validator yet; `checkPD` covers NIXL transfers) | — |

```yaml
validation:
  metricsCheck:
    enabled: true
    checkVLLM: true
    checkScheduler: true
```

Rules of thumb: single-GPU → `checkVLLM` + `checkScheduler`; cache-aware → add
`checkPrefixCache`; flow control → add `checkFlowControl`; P/D → `checkPD`.

### 3. Add to a profile (optional)

Add the case name to `testCases` in `configs/profiles/<name>.yaml`, or create a
new profile there.

## GPU requirements

Two independent gates:

- **`requiresGpu: true`** means the case is never simulated — in `--mock` mode
  it is skipped ("cannot be validated in mock mode"). Set it when the point of
  the test is real GPU behavior (P/D KV transfer, KV offloading, MoE,
  tool-calling with a real model).
- In real runs, the suite sums `nvidia.com/gpu` requested by the manifest
  across decode and prefill sections and **auto-skips the case when the
  cluster has fewer GPUs** — no flag needed, no Pending pods.

## Manifest branches

Manifests are version-specific because EPP plugins differ per train:

| RHAII version | Branch | EPP features |
|---|---|---|
| 3.4 | `3.4-GA` | `precise-prefix-cache-scorer`, basic flow control |
| 3.5+ | `main` | `precise-prefix-cache-producer`, `concurrency-detector`, token-based saturation |

- Always add new manifests to `main` first; cherry-pick to version branches
  when compatible. A manifest using 3.5 plugins will CrashLoop the EPP on 3.4.
- Keep `main` and the newest release branch in sync — divergence causes
  stale-manifest failures in nightly CI.

## Conformance phases

Every test case runs these ordered phases (pytest collects them per case);
config-disabled phases are deselected at collection:

| Phase | Method | Validates |
|---|---|---|
| 01 | `test_01_prereq` | CRD exists, manifest present, GPUs available |
| 02 | `test_02_deploy` | Apply the LLMInferenceService manifest |
| 03 | `test_03_service` | Service creation |
| 04 | `test_04_gateway` | The service's Gateway (from its HTTPRoute `parentRefs`) is Programmed |
| 05 | `test_05_pods` | Pods Running without crashes |
| 06 | `test_06_ready` | LLMInferenceService Ready=True |
| 07 | `test_07_health` | GET /health (direct pod, bypasses EPP) |
| 08 | `test_08_models` | GET /v1/models (direct pod; lists LoRA adapters when configured) |
| 09a | `test_09a_inference` | Chat completions + completions (through the gateway) |
| 09b | `test_09b_messages_responses` | Anthropic /v1/messages + OpenAI /v1/responses |
| 09c | `test_09c_tool_calling` | tool_calls in the response to a tools prompt |
| 10 | `test_10_metrics_vllm` | vLLM request-success metrics |
| 11 | `test_11_metrics_cache` | Prefix KV cache hit metrics |
| 12 | `test_12_metrics_pd` | P/D token distribution + NIXL transfer metrics |
| 13 | `test_13_metrics_scheduler` | EPP processed-request metrics |
| 14 | `test_14_metrics_flow_control` | Flow-control dispatch activity |
| 15 | `test_15_metrics_lora` | LoRA adapter state metrics |
| 16 | `test_16_metrics_kvcache_offloading` | KV offload to CPU tier |
| 17 | `test_17_kvcache_offloading_fs` | KV offload to filesystem tier |
| 20 | `test_20_benchmark` | GuideLLM benchmark with thresholds |
| 21 | `test_21_metrics_post_benchmark` | P/D metrics after benchmark load |
| 30a–e | MaaS phases | MaaSModelRef ready, 401 unauthenticated, API key, authenticated inference, rate limit |
| 99 | `test_99_cleanup` | Delete the manifest's resources |

### Adding a phase

1. Pick a number between existing phases and add the method to `LLMDPhases` in
   `tests/test_conformance.py` (MaaS phases: `MaaSPhases` in `tests/maas/phases.py`).
2. Gate it on a config flag with `@pytest.mark.applies_when.with_args(...)` so
   it is deselected for cases that don't enable it.
3. Guard runtime conditions with the helpers — `_require_deployed()` skips if
   deploy failed; `pytest.skip()` for anything else.
4. Log with `_log()`; request fixtures by parameter name:

| Fixture | Scope | Provides |
|---|---|---|
| `deployer` | session | kubectl wrapper, deploy/cleanup, port-forwards |
| `tc` | parametrized | `TestCase` dataclass for the current case |
| `client` | class | `LLMClient` at the gateway (inference) |
| `pod_client` | class | `LLMClient` at the pod (health/models) |
| `endpoint` / `pod_endpoint` | class | Port-forward URLs |
| `scraper` | class | Prometheus metrics scraper |
| `test_mode` / `mock_mode` / `no_cleanup` | session | Run-mode flags |

## Adding a config field

1. Add the snake_case field to the dataclass in `src/conformance/config.py`
   (a nested type needs a `_build()` branch).
2. Use the camelCase key in the testcase YAML.

## Adding a CLI flag

1. `parser.add_argument()` in `cli.py:main()`.
2. Matching `parser.addoption()` in `tests/conftest.py`.
3. An entry in cli.py's `flag_map` (value flags) or an `if args.<flag>` append
   (boolean flags).
4. Read it with `request.config.getoption("--flag-name")` in fixtures/phases.

## Adding a metrics validator

One worked example — a NIXL transfer check. Four files:

1. **`src/conformance/config.py`** — the flag on `MetricsCheck`:
   `check_nixl: bool = False` (YAML: `checkNIXL`).
2. **`src/conformance/metrics.py`** — the validator:

```python
NIXL_KV_TRANSFER = "nixl:kv_transfer_count_total"

def validate_nixl(vllm: list[ScrapeResult]) -> list[CheckResult]:
    checks = []
    for r in vllm:
        count = r.get(NIXL_KV_TRANSFER)
        checks.append(CheckResult(
            name="nixl_kv_transfer", metric=NIXL_KV_TRANSFER, source=r.source,
            value=count or 0, passed=count is not None and count > 0,
            message=f"nixl_kv_transfer_count={count}",
        ))
    return checks
```

3. **`tests/test_conformance.py`** — the phase, gated and guarded:

```python
@pytest.mark.applies_when.with_args(_metric("check_nixl"))
def test_18_metrics_nixl(self, deployer: Deployer, scraper: Scraper, tc: TestCase, test_mode: str):
    _require_deployed(deployer, tc, test_mode)
    results = scraper.scrape_vllm(tc.name)
    checks = validate_nixl(results)
    for c in checks:
        _log(f"  {c.name}: {'PASS' if c.passed else 'FAIL'} — {c.message}")
    assert not [c for c in checks if not c.passed]
```

4. **The testcase YAML** — `metricsCheck: {enabled: true, checkNIXL: true}`.

## Checklist

- [ ] Manifest in the manifests repo, `main` first
- [ ] `configs/testcases/<name>.yaml` with `manifestPath` matching
- [ ] `metricsCheck` flags match the topology
- [ ] `requiresGpu: true` if the case must never be simulated
- [ ] Added to relevant profiles
- [ ] `docs/test-cases.md` updated — a row in the summary table and a section for the case
- [ ] `--list-testcases` shows ✓; case passes in `--mock` (if simulatable)
- [ ] `uv run pytest tests/ --ignore=tests/test_conformance.py` and
      `uv run ruff check src/ tests/` pass
