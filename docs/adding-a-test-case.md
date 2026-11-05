# Adding a Test Case

Step-by-step guide to add a new conformance test case to llm-d-e2e.

## Choose Your Path

| Path | Best for | Jump to |
|---|---|---|
| **Quick start (script)** | New test case with standard config — generates all boilerplate for you | [Quick Start: Generate Boilerplate](#quick-start-generate-boilerplate) |
| **Manual setup** | Custom topology, non-standard config, or learning how the pieces fit | [Step 1: Create the Manifest](#step-1-create-the-manifest) |
| **New conformance phase** | Adding a new test phase or metrics validator (not just a new test case) | [Adding a New Conformance Phase](#adding-a-new-conformance-phase) |
| **New metrics check** | Adding a new Prometheus metrics validation (e.g. NIXL) | [To add a new metrics check](#to-add-a-new-metrics-check-full-example-adding-nixl-kv-transfer-validation) |

## Overview

A test case has two parts:

1. **Test case config** (`configs/testcases/<name>.yaml`) — defines model, resources, validation criteria, and which metrics to check.
2. **LLMInferenceService manifest** (`deploy/manifests/<name>.yaml`) — the Kubernetes manifest that gets applied to the cluster. Lives in the [llm-d-conformance-manifests](https://github.com/opendatahub-io/llm-d-conformance-manifests) repo.

The test framework loads the config, patches the manifest (mock image swap, pull secrets, auth), applies it via `kubectl`, then runs ordered phases: deploy, service, gateway, pods, ready, health, models, inference, metrics, cleanup.

## Step 1: Create the Manifest

Create the LLMInferenceService YAML in the [conformance-manifests repo](https://github.com/opendatahub-io/llm-d-conformance-manifests). Use `single-gpu-smoke.yaml` as a minimal starting point:

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

Commit this to the appropriate branch in `llm-d-conformance-manifests` (e.g. `main` for 3.5+, `3.4-GA` for 3.4).

## Step 2: Create the Test Case Config

Create `configs/testcases/<name>.yaml`. All keys use **camelCase** in YAML (they map to **snake_case** Python fields automatically).

### Minimal config (no metrics checks)

```yaml
name: my-test-case
description: "Short description of what this test validates"
model:
  name: Qwen/Qwen3-0.6B
  uri: hf://Qwen/Qwen3-0.6B
  displayName: qwen3-0.6b-my-test
  category: single-node-gpu
  cache:
    enabled: true
    storageSize: 10Gi
    keepPVC: true
    timeout: 15m
deployment:
  manifestPath: my-test-case.yaml
  replicas: 1
  readyTimeout: 10m
  resources:
    cpu: "2"
    memory: 8Gi
    gpus: 1
    rdma: false
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

### Key fields explained

| Field | Purpose |
|---|---|
| `name` | Must match the filename (without `.yaml`) |
| `deployment.manifestPath` | Filename of the manifest in `deploy/manifests/` |
| `deployment.replicas` | Expected replica count (used to verify pods) |
| `deployment.readyTimeout` | How long to wait for Ready=True (string: `10m`, `2h`) |
| `deployment.resources.gpus` | GPU count per replica (stripped in `--mock` mode) |
| `deployment.requiresGpu` | If `true`, the test case is **skipped unless `--need-gpu` is passed** (see [GPU Gate](#gpu-gate)) |
| `validation.testPrompts` | Prompts sent to `/v1/chat/completions`, `/v1/completions`, `/v1/messages`, `/v1/responses` |
| `validation.chatPrompts` | Alternative: structured `[{system, user}]` prompts for prefix cache testing |
| `validation.checkMessages` | Enable Anthropic `/v1/messages` inference check (default: `false`) |
| `validation.checkResponses` | Enable OpenAI `/v1/responses` inference check (default: `false`) |
| `cleanup` | Delete the LLMInferenceService after tests complete |

### Cleanup order for multi-document manifests

`test_99_cleanup` deletes what the manifest applied in dependency-safe order:

1. companion documents that reference the service (MaaS model refs and policies,
   side stores such as PostgreSQL/SeaweedFS), in reverse manifest order;
2. the `LLMInferenceService`s;
3. documents a service depends on through `spec.baseRefs` — for example an
   `LLMInferenceServiceConfig` — deleted **last**, because KServe rejects their
   deletion while a service still references them (`DeletionBlocked: ...
   referenced by 1 LLMInferenceService(s)`) and leaves them Terminating.

`SERVICE_DEPENDENCY_KINDS` in `src/conformance/deployer.py` holds that third
group. Every delete carries `--timeout`, so a document stuck behind a finalizer
that cannot clear is reported (and retried at session teardown) instead of
hanging the phase.

### Adding metrics checks

Enable metrics validation by adding a `metricsCheck` section under `validation`:

```yaml
validation:
  # ... other fields ...
  metricsCheck:
    enabled: true
    checkVLLM: true          # test_10: vllm:request_success_total > 0
    checkPrefixCache: true   # test_11: prefix_queries > 0, hit rate
    checkScheduler: true     # test_13: scheduler_e2e_count > 0, ready_pods > 0
    checkFlowControl: true   # test_14: dispatch_cycle_count > 0
    checkPD: true            # test_12: P/D disaggregation metrics
    checkNIXL: true          # (future) NIXL KV transfer metrics
```

Only enable the checks that match your deployment topology. For example, a basic single-GPU test only needs `checkVLLM: true` and `checkScheduler: true`.

### P/D disaggregation config

For prefill/decode split deployments, add a `prefill` section under `deployment`:

```yaml
deployment:
  manifestPath: pd.yaml
  replicas: 1
  prefill:
    replicas: 1
    resources:
      cpu: "4"
      memory: 16Gi
      gpus: 1
```

### Adding a benchmark

To run a GuideLLM performance benchmark after inference:

```yaml
validation:
  benchmark:
    enabled: true
    image: ghcr.io/vllm-project/guidellm:v0.6.0
    rate: 200
    maxSeconds: 240
    data: "prompt_tokens=8000,output_tokens=800"
    backendType: openai_http
    requestType: text_completions
    timeout: 30m
    thresholds:
      minOutputTokensPerSecond: 8000.0
      maxTtftMedianMs: 2000.0
      maxItlMedianMs: 50.0
```

## Step 3: Add to a Profile (optional)

Profiles group test cases for batch runs. Add your test case name to a profile in `configs/profiles/`:

```yaml
# configs/profiles/smoke.yaml
name: smoke
description: "Quick smoke test"
testCases:
  - single-gpu-smoke
  - my-test-case        # ← add here
```

Or create a new profile:

```yaml
# configs/profiles/my-profile.yaml
name: my-profile
description: "My custom test suite"
testCases:
  - my-test-case
```

## Step 4: Pull Manifests and Verify

```bash
# Pull the latest manifests (replace 'main' with your branch)
uv run llm-d-e2e --setup main

# Or pull from your own fork/branch before it merges upstream
uv run llm-d-e2e --setup my-branch \
  --manifest-repo https://github.com/<you>/llm-d-conformance-manifests.git

# Verify your test case shows up
uv run llm-d-e2e --list-testcases
#   ✓ my-test-case  → my-test-case.yaml

# Verify your profile (if added)
uv run llm-d-e2e --list-profiles
```

If your test case shows `✗ (missing)`, the manifest file isn't in the branch you pulled.
`--setup` clones from `opendatahub-io/llm-d-conformance-manifests` by default; add
`--manifest-repo <URL>` to pull from a different repo (e.g. your fork).

## Step 5: Run the Test

```bash
# Mock mode (no GPU, uses llm-d-inference-sim)
KUBECONFIG=~/.kube/my-cluster uv run llm-d-e2e -t my-test-case --mock -v

# Real GPU
KUBECONFIG=~/.kube/my-cluster uv run llm-d-e2e -t my-test-case -v

# With HTML report
KUBECONFIG=~/.kube/my-cluster uv run llm-d-e2e -t my-test-case --mock --html reports/my-test.html -v

# Keep resources after test (for debugging)
KUBECONFIG=~/.kube/my-cluster uv run llm-d-e2e -t my-test-case --mock --nocleanup -v
```

## Step 6: Run Unit Tests

Make sure existing tests still pass:

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run pytest tests/test_smoke.py -v
```

## What Happens at Runtime (mock mode)

When `--mock` is used, the framework patches your manifest before applying:

1. Swaps the `main` container image with `ghcr.io/llm-d/llm-d-inference-sim:latest`
2. Sets simulator args: `--mode random --self-signed-certs --enable-kvcache true`
3. Strips `nvidia.com/gpu` resource requests
4. The EPP/scheduler config from the manifest is **not changed** — it runs as-is

This means your manifest's scheduler/EPP config must be compatible with the RHAII version installed on the cluster, even in mock mode.

## GPU Gate

Some test cases need a real GPU to run at all. For example `kv-offloading-cpu` offloads the KV cache *to* CPU memory, but the model still runs on a GPU. To keep these out of CPU-only clusters, a test case marks its need with `requiresGpu: true`. Such test cases are **skipped by default** and run only when you pass **`--need-gpu`**.

### Mark a test case as needing a GPU

```yaml
deployment:
  manifestPath: kv-offloading-cpu.yaml
  requiresGpu: true       #  skipped unless --need-gpu is passed
```

### How it behaves

| Situation | Result |
|---|---|
| `requiresGpu: false` (default) | Never gated — runs everywhere |
| `requiresGpu: true`, no `--need-gpu` | **SKIPPED** — "requires a GPU — pass --need-gpu to run it" |
| `requiresGpu: true`, `--need-gpu`, cluster has GPUs | Runs |
| `requiresGpu: true`, `--need-gpu`, cluster has **no** GPU | **SKIPPED** (safety net — avoids pods stuck Pending) |
| `requiresGpu: true`, `--mock` | Runs — mock strips GPU requests, so no GPU is needed |

The check runs in the earliest phases (`test_01_prereq` / `test_02_deploy`), so when
it skips, all downstream phases skip too. When `--need-gpu` is set, GPU capacity is
queried once via `kubectl get nodes` (allocatable `nvidia.com/gpu` summed across
nodes) and cached, as a safety net.

### Run GPU-only test cases

On a GPU cluster, opt in explicitly:

```bash
e2e -t kv-offloading-cpu --need-gpu -v
```

## Version Compatibility

Manifests are version-specific. If your test case uses EPP features from a specific RHAII version, put the manifest in the right branch:

| RHAII Version | Manifest branch | EPP features available |
|---|---|---|
| 3.4 | `3.4-GA` | `precise-prefix-cache-scorer`, basic flow control |
| 3.5+ | `main` | `precise-prefix-cache-producer`, `concurrency-detector`, token-based saturation |

If the manifest uses 3.5 EPP plugins but runs on a 3.4 cluster, the EPP will CrashLoopBackOff.

## Adding a New Conformance Phase

If you need to validate something new (not just a new test case), you add a test method to `tests/test_conformance.py`. Phases run in order by their numeric prefix.

### Current phase map

```
test_01_prereq          — CRD exists, manifest exists
test_02_deploy          — kubectl apply the manifest
test_03_service         — wait for Service creation
test_04_gateway         — wait for Gateway programmed
test_05_pods            — wait for pods Running
test_06_ready           — wait for Ready=True
test_07_health          — GET /health (direct pod)
test_08_models          — GET /v1/models (direct pod)
test_09a_inference      — POST /v1/chat/completions + /v1/completions (via gateway)
test_09b_messages_responses — Anthropic /v1/messages + OpenAI /v1/responses
test_09c_tool_calling   — tool-calling: validates tool_calls in response
test_10_metrics_vllm    — vLLM request_success > 0
test_11_metrics_cache   — prefix cache queries/hits
test_12_metrics_pd      — P/D disaggregation metrics
test_13_metrics_scheduler — EPP scheduler latency and ready endpoints > 0
test_14_metrics_flow_control — flow control dispatch/saturation
test_20_benchmark       — GuideLLM performance benchmark
test_21_metrics_post_benchmark — metrics after benchmark
test_99_cleanup         — delete LLMInferenceService
```

### How to add a new phase

1. Pick a number between existing phases (e.g. `test_18` for a new phase after KV-cache checks).

2. Add the method to `TestConformance` in `tests/test_conformance.py`:

```python
def test_15_metrics_nixl(self, deployer: Deployer, scraper: Scraper, tc: TestCase, test_mode: str):
    """NIXL KV transfer metrics should show transfers."""
    _require_deployed(deployer, tc, test_mode)
    mc = tc.validation.metrics_check
    if not mc.enabled or not mc.check_nixl:
        pytest.skip("NIXL metrics check disabled")
    _log("Scraping NIXL metrics...")
    results = scraper.scrape_vllm(tc.name)
    # Add your validation logic here
    assert results, "No NIXL metrics scraped"
```

### Key patterns to follow

**Guard with `_require_deployed()`** — skips the phase if deploy failed earlier:
```python
_require_deployed(deployer, tc, test_mode)
```

**Guard with config flags** — skip if the test case doesn't enable this check:
```python
if not mc.enabled or not mc.check_nixl:
    pytest.skip("NIXL metrics check disabled")
```

**Use `_log()` for output** — prints with `→` prefix for consistent formatting:
```python
_log(f"Scraped {len(results)} pod(s)")
```

### Available fixtures (request by parameter name)

| Fixture | Scope | What it provides |
|---|---|---|
| `deployer` | session | kubectl wrapper, deploy/cleanup, port-forwarding |
| `tc` | parametrized | `TestCase` dataclass for the current test case |
| `client` | class | `LLMClient` pointing at gateway (for inference) |
| `pod_client` | class | `LLMClient` pointing at pod directly (for health/models) |
| `endpoint` | class | Gateway port-forward URL |
| `pod_endpoint` | class | Pod port-forward URL |
| `scraper` | class | `Scraper` for Prometheus metrics |
| `test_mode` | session | `"deploy"` or `"discover"` |
| `mock_mode` | session | `True` if `--mock` was passed |
| `no_cleanup` | session | `True` if `--nocleanup` was passed |
| `guidellm_image` | session | GuideLLM container image |

### Adding a new config field

If your phase needs a new config flag (like `check_nixl`):

1. Add the field to the dataclass in `src/conformance/config.py`:
```python
@dataclass
class MetricsCheck:
    # ... existing fields ...
    check_nixl: bool = False   # ← add here (snake_case)
```

2. Use `checkNIXL` (camelCase) in the YAML config:
```yaml
metricsCheck:
  checkNIXL: true
```

The `_snake()` function in `config.py` converts camelCase to snake_case automatically.

### Adding a new metrics validator

If your phase validates Prometheus metrics, add a validator function to `src/conformance/metrics.py`:

```python
def validate_nixl(vllm_results: list[dict]) -> list[CheckResult]:
    checks = []
    for pod_metrics in vllm_results:
        count = pod_metrics.get("nixl:kv_transfer_count_total", [{}])[0].get("value", 0)
        checks.append(CheckResult(
            name="nixl_transfer_count",
            passed=count > 0,
            message=f"nixl_transfer_count={count}",
        ))
    return checks
```

Then import and call it from your test phase in `test_conformance.py`.

### Adding a new CLI flag

If your phase needs a new CLI option:

1. Add to `cli.py:main()` in the `flag_map` dict (or as a boolean flag):
```python
flag_map["--my-flag"] = args.my_flag
```

2. Add the matching `pytest_addoption` entry in `tests/conftest.py`:
```python
parser.addoption("--my-flag", default="", help="Description")
```

3. Add a fixture in `conftest.py` to expose it to tests.

## Quick Start: Generate Boilerplate

Instead of writing YAML from scratch, use the generator script. It creates both files with all required fields pre-filled.

### 1. Run the script

```bash
./scripts/new-testcase.sh my-new-test
```

This creates two files:

```
configs/testcases/my-new-test.yaml    ← test case config (edit in this repo)
deploy/manifests/my-new-test.yaml     ← manifest template (move to conformance-manifests repo)
```

The script refuses to overwrite existing files — safe to re-run.

### 2. Edit the generated config

Open `configs/testcases/my-new-test.yaml` and update:

```yaml
# Change the description
description: "TODO: describe what this test validates"  # ← update this

# Change model if needed (default: Qwen/Qwen3-0.6B)
model:
  name: Qwen/Qwen3-0.6B

# Adjust resources for your topology
deployment:
  replicas: 2            # ← change replica count
  resources:
    gpus: 2              # ← change GPU count

# Enable the right metrics checks for your topology
validation:
  metricsCheck:
    checkVLLM: true
    checkScheduler: true
    # checkPrefixCache: true    # ← uncomment for cache-aware
    # checkFlowControl: true    # ← uncomment for flow control
    # checkPD: true             # ← uncomment for P/D
```

Each flag controls a specific conformance phase and validates different Prometheus metrics:

| Flag | Phase | What it validates | When to enable |
|---|---|---|---|
| `checkVLLM` | `test_10` | `vllm:request_success_total > 0` | Always (basic sanity) |
| `checkPrefixCache` | `test_11` | `vllm:prefix_cache_queries` or `vllm:prefix_cache_queries_total` > 0, `vllm:prefix_cache_hits` or `vllm:prefix_cache_hits_total` > 0, hit rate > 0% | Cache-aware routing with `precise-prefix-cache-producer` + `prefix-cache-scorer` |
| `checkScheduler` | `test_13` | `llm_d_epp_scheduler_e2e_duration_seconds_count` (fallback `inference_extension_scheduler_e2e_duration_seconds_count`) > 0, `llm_d_epp_ready_endpoints` (fallback `inference_pool_ready_pods`) > 0 | Any topology with an EPP/scheduler |
| `checkFlowControl` | `test_14` | `llm_d_epp_flow_control_dispatch_cycle_duration_seconds_count` (fallback `inference_extension_flow_control_dispatch_cycle_duration_seconds_count`) > 0, `llm_d_epp_flow_control_request_enqueue_duration_seconds_count` (fallback `inference_extension_flow_control_request_enqueue_duration_seconds_count`) > 0 | Flow control with `flowControl.saturationDetector` |
| `checkPD` | `test_12` | P/D disaggregation metrics (prefill/decode split) | Prefill/decode topology |
| `checkNIXL` | (future) | NIXL KV transfer count | NIXL-enabled KV cache transfer |

**Pick based on your deployment topology:**

- Basic single-GPU: `checkVLLM` + `checkScheduler`
- Cache-aware routing: add `checkPrefixCache`
- Flow control: add `checkFlowControl`
- P/D disaggregation: add `checkPD`
- Combination (e.g. cache-aware + flow control): enable all relevant flags

**To add a new metrics check** (full example: adding NIXL KV transfer validation):

You need to touch 4 files. Here's every step with the actual code.

**File 1: `src/conformance/config.py`** — add the config flag

```python
@dataclass
class MetricsCheck:
    enabled: bool = False
    check_vllm: bool = False
    check_prefix_cache: bool = False
    check_pd: bool = False
    check_scheduler: bool = False
    check_flow_control: bool = False
    check_nixl: bool = False        # ← add here (snake_case, default False)
```

**File 2: `src/conformance/metrics.py`** — add the validator function

Every validator takes scraped `ScrapeResult` list(s), checks specific Prometheus metrics, and returns `CheckResult` objects. Follow the existing pattern:

```python
# At the top, define the metric name constants
NIXL_KV_TRANSFER = "nixl:kv_transfer_count_total"

def validate_nixl(vllm: list[ScrapeResult]) -> list[CheckResult]:
    """Validate NIXL KV transfer metrics from workload pods."""
    checks = []
    for r in vllm:
        count = r.get(NIXL_KV_TRANSFER)
        checks.append(
            CheckResult(
                name="nixl_kv_transfer",
                metric=NIXL_KV_TRANSFER,
                source=r.source,
                value=count or 0,
                passed=count is not None and count > 0,
                message=f"nixl_kv_transfer_count={count}",
            )
        )
    return checks
```

Key parts:
- `r.get(METRIC_NAME)` returns the metric value from a pod's scraped metrics (or `None` if not found)
- `CheckResult` has: `name` (display label), `metric` (Prometheus name), `source` (pod name), `value`, `passed` (bool), `message`
- Return a list — one `CheckResult` per pod per metric checked

**File 3: `tests/test_conformance.py`** — add the test phase

Pick a number between existing phases (e.g. `test_15` after flow control). Import your validator and wire it up:

```python
# At the top, add to the imports
from conformance.metrics import (
    ...
    validate_nixl,       # ← add import
)

# In class TestConformance, add the method:
def test_15_metrics_nixl(self, deployer: Deployer, scraper: Scraper, tc: TestCase, test_mode: str):
    """NIXL KV transfer metrics should show transfers."""
    _require_deployed(deployer, tc, test_mode)
    mc = tc.validation.metrics_check
    if not mc.enabled or not mc.check_nixl:
        pytest.skip("NIXL metrics check disabled")
    _log("Scraping NIXL metrics...")
    results = scraper.scrape_vllm(tc.name)
    _log(f"Scraped {len(results)} pod(s)")
    assert results, "No NIXL metrics scraped"
    checks = validate_nixl(results)
    for c in checks:
        _log(f"  {c.name}: {'PASS' if c.passed else 'FAIL'} — {c.message}")
    failed = [c for c in checks if not c.passed]
    assert not failed, f"NIXL metric checks failed: {[c.message for c in failed]}"
```

The boilerplate pattern is always:
1. `_require_deployed()` — skip if deploy failed
2. Check the config flag — skip if disabled
3. Scrape metrics — `scraper.scrape_vllm()` for workload pods, `scraper.scrape_epp()` for EPP pods
4. Run validator — returns `CheckResult` list
5. Log each result
6. Assert no failures

**File 4: `configs/testcases/<name>.yaml`** — enable the flag

```yaml
validation:
  metricsCheck:
    enabled: true
    checkNIXL: true       # ← camelCase in YAML
```

**That's it — 4 files, and the new check runs automatically for any test case that enables it.**

### 3. Edit the generated manifest

Open `deploy/manifests/my-new-test.yaml` and add your EPP/scheduler config:

```yaml
spec:
  router:
    scheduler:
      config:
        inline:
          # ← add your EndpointPickerConfig here
```

The generated manifest is a minimal single-GPU template. For cache-aware, flow-control, or P/D topologies, copy the scheduler section from an existing manifest like `cache-aware.yaml` or `flow-control.yaml`.

### 4. Move the manifest to the conformance-manifests repo

```bash
# Copy to the conformance-manifests repo
cp deploy/manifests/my-new-test.yaml \
   /path/to/llm-d-conformance-manifests/my-new-test.yaml

# Always add to main first — main is the latest/greatest and must have all manifests
cd /path/to/llm-d-conformance-manifests
git checkout main
git add my-new-test.yaml
git commit -m "add my-new-test manifest"
git push

# If the manifest is also compatible with 3.4, cherry-pick to 3.4-GA
git checkout 3.4-GA
git cherry-pick main
git push
```

**Branch rules:**
- `main` = latest/greatest, targets the newest RHAII version — **always add here first**
- `3.4-GA`, `3.5-GA`, etc. = version-specific branches with compatible EPP configs
- If a manifest is compatible with older versions, cherry-pick from main to the stable branch
- If a manifest is only valid for a specific version (e.g. uses 3.4-only plugins), add it only to that branch — do NOT add it to main
- When a new release branch is created (e.g. `3.6-stable`), sync it from main so it starts with all latest manifests
- Keep main and the latest release branch in sync — divergence causes the stale-manifest failures we saw with the nightly CI

### 5. Pull and run

```bash
# Pull the updated manifests
uv run llm-d-e2e --setup main

# Or, before merging upstream, pull from your fork/branch:
uv run llm-d-e2e --setup my-branch \
  --manifest-repo https://github.com/<my-org>/<my-repo>.git

# Verify it shows up
uv run llm-d-e2e --list-testcases
#   ✓ my-new-test  → my-new-test.yaml

# Run in mock mode
KUBECONFIG=~/.kube/my-cluster uv run llm-d-e2e -t my-new-test --mock -v
```

### What the generated files contain

The **config YAML** comes pre-filled with:
- Qwen3-0.6B model (smallest, works in mock mode)
- 1 replica, 1 GPU, 10Gi cache PVC
- Health check on port 8000 (HTTPS)
- Single test prompt ("What is 2+2?")
- vLLM + scheduler metrics enabled
- Cleanup enabled

The **manifest YAML** comes pre-filled with:
- `LLMInferenceService` with correct apiVersion
- Pull secret references (`rhai-pull-secret`)
- Scheduler with the `main` container
- Resource limits with GPU request
- Liveness probe on `/health:8000`

Both use `name: my-new-test` matching the filename, so `manifestPath` resolves correctly.

## Checklist

- [ ] Manifest created in conformance-manifests repo (correct branch)
- [ ] Test case config created in `configs/testcases/<name>.yaml`
- [ ] `deployment.manifestPath` matches the manifest filename
- [ ] `metricsCheck` flags match the deployment topology
- [ ] `deployment.requiresGpu: true` set if the test case needs a real GPU (see [GPU Gate](#gpu-gate))
- [ ] Added to relevant profile(s) in `configs/profiles/`
- [ ] `--list-testcases` shows `✓` for the new test case
- [ ] Test passes in mock mode (`--mock`)
- [ ] Test passes with real GPU (if applicable)
- [ ] Unit tests pass (`pytest tests/test_smoke.py`)
- [ ] Lint passes (`ruff check src/ tests/`)
