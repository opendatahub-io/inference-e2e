# llm-d-e2e

End-to-end conformance tests for [llm-d](https://github.com/llm-d) / KServe `LLMInferenceService` deployments on Kubernetes.

**Guides:** [Adding a Test Case](docs/adding-a-test-case.md)

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) — fast Python package manager
- `kubectl` configured with cluster access
- Cluster with `LLMInferenceService` CRD installed (RHAI or KServe)

### xKS cluster requirements

The test suite automatically labels the `llm-conformance-test` namespace with `inference-gateway-access=true` so HTTPRoutes are accepted by the inference gateway. The gateway must be configured with a namespace label selector that matches this label — refer to your platform's installation guide for the correct helm values.

Install uv if you don't have it:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/opendatahub-io/llm-d-e2e.git
cd llm-d-e2e

# 2. Install dependencies
uv sync

# 3. Clone test manifests — interactive (lists branches)
uv run llm-d-e2e --setup

# 3. Or clone a specific branch directly
uv run llm-d-e2e --setup main          # latest
uv run llm-d-e2e --setup 3.5-GA        # 3.5 GA manifests
uv run llm-d-e2e --setup 3.4-stable    # 3.4 stable manifests

# 3. Or pull from a different repo/branch (e.g. your fork, for local testing)
uv run llm-d-e2e --setup my-branch \
  --manifest-repo https://github.com/<my-org>/<my-repo>.git

# 4. (Optional) Set up a shortcut
alias e2e='uv run llm-d-e2e'
```

By default `--setup` clones manifests from the upstream repo
(`https://github.com/opendatahub-io/llm-d-conformance-manifests.git`). Use
`--manifest-repo <URL>` to pull from any other repo — combined with a branch name,
this lets you test manifests from your own fork before they merge upstream.

You can also use `make setup` which provides the same interactive branch selection:
```bash
make setup                      # interactive — lists branches, pick one
make setup BRANCH=3.5-GA        # direct — specific branch
make setup BRANCH=my-branch MANIFEST_REPO=https://github.com/<my-org>/<my-repo>.git
```

## Quick Start

```bash
# List available tests and profiles
e2e --list-testcases
e2e --list-profiles

# Run smoke test
e2e -t single-gpu-smoke

# Run all conformance tests
e2e -p configs/profiles/all.yaml
```

## Usage

```bash
# Single test case
e2e -t single-gpu

# Multiple test cases
e2e -t single-gpu,cache-aware

# Setup manifests and run tests in one command
e2e --setup 3.5-GA -t single-gpu,cache-aware

# Keep resources after test (for debugging)
e2e -t single-gpu --nocleanup

# Simulate vLLM with llm-d-inference-sim (no GPU needed)
e2e -t single-gpu --mock

# Exercise an auth-protected RHOAI inference endpoint with the simulator
e2e -t single-gpu-smoke --mock

# Run full 3.5 profile (auto-skips tests needing more GPUs than available)
e2e -p configs/profiles/3.5.yaml

# Run 3.5 profile in mock mode (no GPU needed, requiresGpu tests skipped)
e2e -p configs/profiles/3.5.yaml --mock -v

# Verbose output, stop on first failure
e2e -t single-gpu -v -x

# Generate HTML report
e2e -t single-gpu --html report.html

# Target a specific cluster
e2e -t single-gpu --kubeconfig ~/.kube/config --namespace my-test-ns

# Validate an existing deployment (no deploy/cleanup)
e2e -t single-gpu --mode discover --endpoint http://my-service:8000
```

Authentication is enabled by default. Supply `--bearer-token` to use an existing
identity; otherwise the suite creates a dedicated service account and
namespace-scoped Role/RoleBinding for each testcase. The Role grants `get` access
only to that testcase's `LLMInferenceService` resources. A one-hour token is passed
directly to the conformance HTTP clients. The cluster credential must be allowed
to create these resources and request service-account tokens. Credentials are
removed at fixture teardown unless `--nocleanup` or the testcase's `cleanup: false`
retains them. `--disable-auth` requests anonymous access on the deployed service
and skips service-account creation. MaaS API-key authentication remains a separate
flow.

## Testing an Existing Deployment

If you already have an LLMInferenceService running and just want to validate it (health, inference, metrics) without deploying or cleaning up:

```bash
# Get the service URL
kubectl get llminferenceservice -n my-namespace
# NAME         URL                                    READY
# my-model     http://gateway.example.com/ns/model    True

# Run validation against it
e2e -t single-gpu --mode discover --namespace my-namespace

# Or specify the endpoint directly
e2e -t single-gpu --mode discover --endpoint http://gateway.example.com/ns/model
```

This skips the deploy and cleanup phases — only runs health, models, inference, and metrics checks.

## Container Image

The test suite is available as a container image at `ghcr.io/opendatahub-io/llm-d-e2e`.

### Build

```bash
# Default (main manifests baked in)
docker build -t llm-d-e2e .

# Specific manifest branch baked in
docker build --build-arg MANIFEST_REF=3.5-GA -t llm-d-e2e:3.5 .
```

### Run tests

```bash
# Run with baked-in manifests
docker run --rm \
  -v ~/.kube:/root/.kube:z \
  ghcr.io/opendatahub-io/llm-d-e2e \
  -t single-gpu-smoke --mock -v

# Use a non-default kubeconfig
docker run --rm \
  -e KUBECONFIG=/root/.kube/my-cluster \
  -v ~/.kube:/root/.kube:z \
  ghcr.io/opendatahub-io/llm-d-e2e \
  -t single-gpu-smoke,single-gpu,cache-aware --mock -v

# Setup different manifests and run tests in one command
docker run --rm \
  -e KUBECONFIG=/root/.kube/my-cluster \
  -v ~/.kube:/root/.kube:z \
  ghcr.io/opendatahub-io/llm-d-e2e \
  --setup 3.5-GA \
  -t cache-aware,flow-control,flow-control-tokens --mock -v

# Generate HTML report (mount reports directory)
docker run --rm \
  -e KUBECONFIG=/root/.kube/my-cluster \
  -v ~/.kube:/root/.kube:z \
  -v $(pwd)/reports:/app/reports:z \
  ghcr.io/opendatahub-io/llm-d-e2e \
  -t single-gpu-smoke --mock --html reports/mock-ci.html -v
```

### Switch manifest branch

```bash
# List available branches and pick one (requires -it for interactive prompt)
docker run --rm -it ghcr.io/opendatahub-io/llm-d-e2e --setup

# Direct — no prompt
docker run --rm ghcr.io/opendatahub-io/llm-d-e2e --setup 3.5-GA
docker run --rm ghcr.io/opendatahub-io/llm-d-e2e --setup 3.4-stable
```

### Interactive mode

```bash
# Interactive shell — switch branches, run multiple tests
docker run --rm -it \
  -e KUBECONFIG=/root/.kube/my-cluster \
  -v ~/.kube:/root/.kube:z \
  -v $(pwd)/reports:/app/reports:z \
  --entrypoint bash ghcr.io/opendatahub-io/llm-d-e2e

# Inside the container:
uv run llm-d-e2e --setup              # interactive branch selection
uv run llm-d-e2e --setup 3.5-GA       # or direct
uv run llm-d-e2e -t single-gpu --mock -v
uv run llm-d-e2e -t cache-aware --mock --html reports/cache.html -v
```

### Utility commands

```bash
# List test cases
docker run --rm ghcr.io/opendatahub-io/llm-d-e2e --list-testcases

# List profiles
docker run --rm ghcr.io/opendatahub-io/llm-d-e2e --list-profiles

# Setup only (clone manifests, show test case mapping)
docker run --rm ghcr.io/opendatahub-io/llm-d-e2e --setup 3.5-GA
```

### How manifests work in the container

- **Build time**: `main` branch manifests are baked into the image (configurable via `--build-arg MANIFEST_REF=`)
- **Runtime `--setup <branch>`**: clones the specified branch from GitHub, replaces baked-in manifests
- **Runtime `--setup`** (interactive, requires `-it`): lists all branches from GitHub, prompts to pick
- **Runtime `--manifest-repo <URL>`**: clone from a different repo instead of the default (`opendatahub-io/llm-d-conformance-manifests`) — works with both direct and interactive `--setup`, and with a branch name lets you test a fork before it merges upstream
- Network access to GitHub is required at runtime for `--setup`; without it, the baked-in manifests are used

## Test Cases

| Name | GPUs | What it tests |
|------|------|---------------|
| single-gpu-smoke | 1 | Fast baseline (no metrics) |
| single-gpu | 1 | Scheduler + metrics |
| single-gpu-no-scheduler | 3 | K8s native round-robin |
| cache-aware | 2 | Prefix KV cache routing |
| pd | 3 | Prefill/Decode disaggregation |
| moe | 8 | MoE, RDMA, expert parallelism |
| multi-pool | 2 | Multiple InferencePools |
| flow-control | 1 | Flow control with utilization-based saturation detector |
| flow-control-tokens | 1 | Flow control with token-based concurrency detector |
| kv-offloading-cpu | 1 | KV cache offloading to CPU memory |
| kv-offloading-tiered | 1 | KV cache offloading to CPU + filesystem tiers |
| pd-performance | 16 | P/D benchmark with GuideLLM (4 prefill + 2 decode, NIXL, RDMA) |
| maas-single-gpu | 1 | single-gpu served through MaaS: auth, API key, token rate limit (needs the MaaS stack) |

## Test Phases

Each test case runs through ordered phases:

1. **Prereq** — CRD exists
2. **Deploy** — apply LLMInferenceService manifest
3. **Service** — wait for Service
4. **Gateway** — wait for Gateway programmed
5. **Pods** — wait for pods Running
6. **Ready** — wait for Ready=True
7. **Health** — GET /health
8. **Models** — GET /v1/models
9. **Inference** — POST /v1/chat/completions
10. **Metrics** — scrape and validate Prometheus metrics
11. **MaaS** — for manifests with a `MaaSModelRef`: unauthenticated 401, API key, authenticated inference, rate limit
12. **Cleanup** — delete resources

If deploy fails (e.g., manifest missing for the selected branch), all subsequent phases for that test case are automatically skipped. CrashLoopBackOff is detected within ~45 seconds instead of waiting the full timeout.

## P/D Performance Benchmark

The `pd-performance` test case runs a [GuideLLM](https://github.com/vllm-project/guidellm) benchmark against a P/D disaggregated deployment (gpt-oss-120b) and validates performance thresholds and NIXL transfer metrics.

### Requirements

- 16 GPUs: 2 decode nodes (4 GPU each, TP=4) + 4 prefill nodes (2 GPU each, TP=2)
- RDMA/InfiniBand networking (`rdma/ib` resource)
- Pre-created PVC `model-cache-pvc` with the model downloaded (500Gi)

### Pre-cache the model

```bash
e2e -t pd-performance --mode cache
```

This creates the `model-cache-pvc` PVC and downloads `openai/gpt-oss-120b` from HuggingFace.

### Run the benchmark

```bash
# Full run: deploy → conformance phases → benchmark → post-benchmark metrics → cleanup
e2e -t pd-performance

# With node placement control
e2e -t pd-performance \
  --decode-node-selector gpu-type=a100-80g \
  --prefill-node-selector gpu-type=a100-40g

# Override the GuideLLM image
e2e -t pd-performance --guidellm-image ghcr.io/vllm-project/guidellm:v0.7.0

# Keep resources for debugging
e2e -t pd-performance --nocleanup
```

### What the benchmark validates

| Check | Threshold |
|-------|-----------|
| Output tokens/s | >= 8000 |
| TTFT median | <= 2000 ms |
| TTFT p95 | <= 5000 ms |
| ITL median | <= 50 ms |
| ITL p95 | <= 100 ms |
| Failed request ratio | <= 5% |
| NIXL transfers | > 0 (KV transfer happened) |
| NIXL failed transfers | == 0 |
| Decode KV transfer > local compute | P/D topology is working |

Thresholds are configurable in `configs/testcases/pd-performance.yaml` under `validation.benchmark.thresholds`.

### Benchmark phases

The benchmark adds three phases after the standard conformance checks:

- **test_12** — Pre-benchmark P/D metrics (raw metric dump for baseline)
- **test_20** — GuideLLM benchmark (warmup + main run + threshold assertions)
- **test_21** — Post-benchmark P/D metrics (validates NIXL transfers after load)

## MaaS Tests

`maas-single-gpu` deploys `single-gpu` through Models-as-a-Service and, after the standard llm-d phases, checks the governed route: unauthenticated request → 401, API key creation → 201, authenticated inference → 200, subscription token rate limit → 429. Its manifest (`maas-single-gpu.yaml`) holds the LLMInferenceService bound to both `inference-gateway` and `maas-default-gateway` — the MaaS controller only governs routes on the MaaS gateway — plus its `MaaSModelRef`, `MaaSAuthPolicy`, and `MaaSSubscription`. MaaS phases live in `tests/maas/` and are deselected for test cases whose manifest declares no `MaaSModelRef`.

MaaS suites are versioned like the llm-d ones (`maas-3.5`, `maas-3.6`); run them against the same manifest branch as the llm-d profile, on a cluster with the MaaS stack:

```bash
uv run llm-d-e2e -p configs/profiles/3.6.yaml      --setup 3.6-ea2 --platform aks --html reports/report.html      --mock -v
uv run llm-d-e2e -p configs/profiles/maas-3.6.yaml --setup 3.6-ea2 --platform aks --html reports/maas-report.html --mock -v
```

Prerequisites: the MaaS stack installed — RHCL (Kuadrant: Authorino, Limitador), the MaaS Postgres database and `maas-db-config` Secret, and the RHAII chart with `components.aigateway.modelsAsAService.enabled=true`, which creates both gateways. On xKS, `validation.maas.endpointScheme: http` is set because the MaaS gateway's port 443 is not reachable externally and its certificate only covers in-cluster names.

Known issues:

- Kuadrant policies stay Pending with "Gateway API provider (istio / envoy gateway) is not installed" when the Kuadrant operator started before Istio: restart it (`kubectl delete pod -n kuadrant-operators -l app.kubernetes.io/name=kuadrant-operator`).
- On xKS the `maas-api-key-cleanup` CronJob fails with `CreateContainerConfigError` (runAsNonRoot with a root image) until [models-as-a-service#1627](https://github.com/opendatahub-io/models-as-a-service/pull/1627) ships; it does not affect these tests.

### Sizing a mock cluster (e.g. minikube)

`--mock` gives each workload container fixed resources (requests `500m`/`1Gi`, limits `1`/`2Gi`), so the platform dominates. Requests measured on AKS with RHAII 3.5 + MaaS: ~2 CPU / ~4 GiB across 22 platform pods (Istio ~0.5 CPU / 2.1 GiB, Kuadrant ~0.9 CPU / 0.7 GiB, KServe + MaaS controllers and gateways ~0.6 CPU / 1.2 GiB, maas-api + Postgres 0.1 CPU / 128 MiB). With system pods and one mock test case that is ~4 CPU / ~7 GiB of requests:

```bash
minikube start --cpus=8 --memory=16g --disk-size=60g   # minimum ~6 CPU / 12 GiB
minikube tunnel   # keep running: gateways are LoadBalancer Services, and MaaS phases call the gateway address from the host
```

Disk covers the platform images and the Postgres PVC; the manifest's `hf://` model URI is kept in mock mode, so the node also needs internet access to Hugging Face. This sizing is derived from AKS measurements and has not been validated on minikube.

## Development

```bash
# Run unit tests (no cluster needed)
uv run pytest tests/test_smoke.py -v

# Lint
uv run ruff check src/ tests/

# Format
uv run ruff format src/ tests/
```

CI runs lint, format, and smoke tests automatically on every PR via GitHub Actions.
