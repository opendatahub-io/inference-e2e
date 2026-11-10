# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

End-to-end conformance test suite for [llm-d](https://github.com/llm-d) / KServe `LLMInferenceService` deployments on Kubernetes. Python + pytest framework that deploys LLMInferenceService resources and validates them through ordered phases (CRD check, deploy, service/gateway/pod readiness, health, inference, metrics, cleanup).

## Docs — read these first

- `docs/quick-start.md` — running the suite (prerequisites, run modes, mock mode, container)
- `docs/test-cases.md` — what every test case deploys and verifies, with a generated summary table
- `docs/developing.md` — contributor workflows: adding test cases, phases, config fields, CLI flags, metrics validators
- `docs/troubleshooting.md` — symptom-first failure guide

This file is the code-level architecture map. Contributor how-to content lives in `docs/developing.md` — do not duplicate it here.

## Prerequisites

- Python 3.11+, [uv](https://docs.astral.sh/uv/) package manager
- `kubectl` configured with cluster access (for conformance tests, not unit tests)
- Cluster with `LLMInferenceService` CRD installed (RHAI or KServe)
- Manifests from [llm-d-conformance-manifests](https://github.com/opendatahub-io/llm-d-conformance-manifests) (cloned via `--setup`)

## Common Commands

```bash
uv sync                                              # install dependencies
uv run llm-d-e2e --setup main                        # clone test manifests (latest)
uv run llm-d-e2e --setup 3.5-GA                      # clone manifests (specific branch)

uv run pytest tests/ -v --ignore=tests/test_conformance.py  # unit tests (no cluster needed)
uv run ruff check src/ tests/                         # lint
uv run ruff format src/ tests/                        # format

uv run llm-d-e2e -t single-gpu-smoke                  # run single conformance test case
uv run llm-d-e2e -t single-gpu,cache-aware            # run multiple test cases
uv run llm-d-e2e -t single-gpu --mock                 # simulate vLLM (no GPU)
uv run llm-d-e2e -t single-gpu --mode discover --endpoint http://svc:8000  # validate existing deployment
uv run llm-d-e2e -p configs/profiles/smoke.yaml       # run a profile
uv run llm-d-e2e -p configs/profiles/3.5.yaml         # run version-specific conformance profile
uv run llm-d-e2e -t single-gpu --nocleanup            # keep resources after test
uv run llm-d-e2e -t single-gpu --html report.html     # generate HTML report
uv run llm-d-e2e -t single-gpu -x                     # stop on first failure
uv run llm-d-e2e --list-testcases                     # list available test cases
uv run llm-d-e2e --list-profiles                      # list available profiles

# Run a single conformance phase (by method name prefix)
uv run pytest tests/test_conformance.py -k "test_09a_inference" --testcase single-gpu
```

Makefile targets mirror CLI: `make test TESTCASE=single-gpu`, `make unittest`, `make lint`, `make format`, `make setup`.

## Architecture

### CLI → pytest delegation

`cli.py:main()` parses user flags and translates them to pytest options, then runs `pytest tests/test_conformance.py` as a subprocess. **Every CLI flag in `cli.py` must have a matching `conftest.py:pytest_addoption()` entry** — when adding a new flag, update both files and the `flag_map` dict in `cli.py:main()`. Boolean flags (`--nocleanup`, `--disable-auth`) are handled separately after `flag_map` iteration since they use `store_true` rather than values.

### Test case parametrization

`conftest.py:pytest_generate_tests()` resolves which test cases to run (from `--testcase` names or `--profile` YAML) and parametrizes the `tc` fixture. Each `tc` is a `TestCase` dataclass loaded from `configs/testcases/*.yaml`. `TestConformance` methods run once per test case.

### Run modes

The `--mode` flag controls which phases execute:
- **`deploy`** (default) — full lifecycle: deploy → validate → cleanup.
- **`discover`** — skip deploy/cleanup, validate an existing deployment (requires `--endpoint` or auto-detected). Phases call `_require_deployed()` which returns early in discover mode. Pod and readiness phases check the testcase-named primary service plus only those secondary manifest services that exist in the cluster (`Deployer.existing_service_names()`).

### Ordered conformance phases

`test_conformance.py:TestConformance` composes three phase groups: `LLMDPhases` (01–21, in `test_conformance.py`), `MaaSPhases` (30a–30e, in `tests/maas/phases.py`), and `CleanupPhase` (99). pytest collects inherited test methods in reverse-MRO order, so `class TestConformance(CleanupPhase, MaaSPhases, LLMDPhases)` runs the groups in that order for each test case; within a group, methods run in definition order.

The numbered phase table (01 prereq → 99 cleanup, including MaaS and benchmark phases) lives in `docs/developing.md` — keep it in sync when adding a phase.

Phases a test case's config turns off are deselected at collection, not skipped: each such phase carries `@pytest.mark.applies_when.with_args(predicate)`, and `conftest.py:pytest_collection_modifyitems` drops items whose `predicate(tc)` is false (the summary reports them as `deselected`). Use `.with_args(...)` — a mark called with a lone callable treats it as the decorated function. Runtime conditions (missing manifest, failed deploy, too few GPUs, `--nocleanup`, `--mode discover`) still `pytest.skip()` and show as SKIPPED.

### MaaS testcases

A MaaS testcase is a testcase config plus a same-named manifest (e.g. `maas-single-gpu.yaml`) holding the LLMInferenceService (bound to `inference-gateway` and `maas-default-gateway`) and its `MaaSModelRef`, `MaaSAuthPolicy`, and `MaaSSubscription`. The llm-d phases validate the model as usual; the MaaS phases are deselected unless the manifest declares a `MaaSModelRef`. MaaS code is owned separately: `tests/maas/` (phases, unit tests) and `src/conformance/maas.py` (client, readiness, manifest ref patching).

Deploy applies documents that declare another namespace (MaaS policies live in `models-as-a-service`) in a separate `kubectl apply -n <namespace>`; manifests with no declared namespaces still apply in one call. `maas.patch_maas_refs()` points the `MaaSModelRef` at the renamed primary service and the policy/subscription `modelRefs` at the test namespace. Cleanup deletes non-LLMInferenceService documents in reverse manifest order before the services. `validation.maas.endpointScheme` overrides the MaaSModelRef endpoint scheme (`http` on AKS, where the gateway's port 443 is blocked).

### Skip propagation

Two helpers control cascading skips across phases:
- `_require_manifest(tc)` — skips if the manifest file doesn't exist for the current branch.
- `_require_deployed(deployer, tc, test_mode)` — skips if deploy failed or was skipped. In discover mode, this check is bypassed.

`test_99_cleanup` is gated separately by `Deployer.needs_cleanup(tc.name)`, which becomes true as soon as `kubectl apply` is attempted. A multi-document manifest can partially apply before a later document is rejected, so a failed deploy still gets cleaned up.

### Leftover resources and GPUs

Leftover LLMInferenceServices hold GPUs, so later test cases stay Pending. The suite guards against this in three places:
- **Ctrl-C** — `cli.py:_run_pytest` waits for pytest to finish session teardown instead of killing it. A second Ctrl-C force-quits.
- **Session teardown** — the `deployer` fixture calls `Deployer.cleanup_pending()`, which deletes every applied test case that phase 99 did not clean up (covers Ctrl-C and `-x`). It runs even if stopping port-forwards fails, and honors `tc.cleanup`.
- **Run start** — in `--mode deploy`, the `deployer` fixture calls `Deployer.remove_stale_services()`, which deletes LLMInferenceServices left by a run that was killed outright (SIGKILL, closed terminal). It selects only services labeled `llm-d-e2e.opendatahub.io/managed=true` (`MANAGED_LABEL`, added by `_patch_manifest()`), so services the suite did not deploy are never touched.

`--nocleanup` disables both the teardown cleanup and the start-of-run sweep.

### Fast-fail behaviors

- **CrashLoopBackOff**: `wait_for_pods()` and `wait_for_ready()` poll every 15s. After 3 consecutive detections (~45s), the deploy fails immediately instead of waiting the full timeout (30+ minutes).
- **Persistent controller error**: `wait_for_ready()` polls the `Ready` condition's `reason` and `message`. If the same reason+message pair persists across 3 consecutive polls, it raises immediately. Changing reason/message indicates controller progress and polling continues.
- **Operator image pull detection**: Both `test_01_prereq` and `wait_for_ready()` call `_check_operator_image_issues()`, scanning pods in `OPERATOR_NAMESPACES` for `ImagePullBackOff`/`ErrImagePull` states.

### Webhook and CRD transient error retry

`deployer.py:_apply_with_webhook_retry()` wraps `kubectl apply` with retry logic for transient post-upgrade errors (webhook not ready, CRD not registered). Non-transient errors (bad fields, RBAC) are re-raised immediately.

### Endpoint routing (gateway vs pod)

Health and models endpoints return 503 through the Gateway API + EPP because EPP only handles inference. The suite uses two separate port-forwards:
- **Gateway** (`client` fixture): the Service of the Gateway the LLMInferenceService is routed through, on that Gateway's HTTP listener port — for `/v1/chat/completions`, `/v1/messages`, `/v1/responses`.
- **Pod** (`pod_client` fixture): `workload-pod:8000` in the test namespace — for `/health` and `/v1/models`. HTTPS (self-signed).

`Deployer.service_gateway()` reads the first Gateway `parentRef` of the service's `<name>-kserve-route` HTTPRoute, which covers both explicit `spec.router.gateway.refs` and the platform default (`openshift-ingress/openshift-ai-inference` on OpenShift, `redhat-ods-applications/inference-gateway` on xKS). Phase 04 (`wait_for_gateway()`) waits for that Gateway to be `Programmed` with an address; other Gateways in the cluster are ignored. The port-forward targets the Service labeled `gateway.networking.k8s.io/gateway-name=<gateway>` in the Gateway's namespace.

### Fixture scoping

- **Session-scoped**: `deployer`, `report`, `no_cleanup`, `test_mode`, `mock_mode`, `guidellm_image`
- **Class-scoped**: `endpoint`, `client`, `pod_endpoint`, `pod_client`, `scraper`

### Source modules (`src/conformance/`)

- **config.py** — Dataclass config types and YAML loaders. YAML keys are camelCase, Python fields are snake_case; `_build()` handles recursive conversion.
- **deployer.py** — `Deployer`: manages LLMInferenceService lifecycle via `kubectl` subprocess calls. Handles deploy, wait-for-ready, port-forwarding, manifest patching (mock image, pull secrets, auth disable, LoRA spec injection, node selectors, env overrides), EPP metrics RBAC, pull secret propagation, namespace labeling for gateway access, and cleanup.
- **client.py** — `LLMClient`: HTTP client (httpx) for `/health`, `/v1/models`, `/v1/completions`, `/v1/chat/completions`, `/v1/messages`, `/v1/responses`.
- **metrics.py** — `Scraper`: scrapes Prometheus metrics from pods via `kubectl exec` (python3/wget), falling back to port-forward + httpx for minimal containers. Per-topology validators: `validate_vllm_basic`, `validate_cache_aware`, `validate_pd`, `validate_scheduler`, `validate_flow_control`, `validate_lora`. `parse_prometheus()` parses text exposition format.
- **model.py** — `ModelDownloader`: creates PVCs and download Jobs for pre-caching models from HuggingFace.
- **report.py** — JSON report generation with pass/fail/skip summary.
- **benchmark.py** — `run_benchmark()`: creates a GuideLLM K8s Job, parses JSON results delimited by `---GUIDELLM_JSON_START---` marker.

### Metrics validation topology

| Validator | Scrape target | Phase | Topology | Gate flag |
|-----------|--------------|-------|----------|-----------|
| `validate_vllm_basic` | workload pods | test_10 | All | `checkVLLM` |
| `validate_cache_aware` | workload + EPP | test_11 | Prefix KV cache | `checkPrefixCache` |
| `validate_pd` | workload + prefill | test_12 | P/D disaggregation | `checkPD` |
| `validate_scheduler` | EPP pods | test_13 | Scheduler/EPP | `checkScheduler` |
| `validate_flow_control` | EPP pods | test_14 | Flow control | `checkFlowControl` |
| `validate_lora` | workload pods | test_15 | LoRA adapters | `checkLora` |

`MetricsCheck.check_nixl` exists in the dataclass but has no validator method yet.

EPP pod discovery tries multiple label patterns (`EPP_LABELS` list in `metrics.py`) because the component label varies across llm-d versions.

### EPP metrics auth

The EPP's `--metrics-endpoint-auth=true` flag (default in RHOAI 3.5+) requires bearer token auth to scrape `/metrics` on port 9090. `Deployer.ensure_metrics_rbac()` creates a `ClusterRoleBinding` granting the EPP's service account access to `kserve-metrics-reader-cluster-role`. The scraper generates a token via `kubectl create token`.

## Contributor workflows

Adding a CLI flag, test case, config field, conformance phase, or metrics validator is documented step-by-step in `docs/developing.md` — the single copy. Keep that file in sync when these code paths change.

## Key Design Decisions

- All cluster interaction goes through `kubectl` subprocess calls (no Python K8s client library).
- Test cases are data-driven via YAML configs, not hardcoded in test files.
- The `--mock` flag swaps the vLLM container with llm-d-inference-sim, injects simulator args, and strips GPU resource requests, enabling full e2e flow without GPUs.
- camelCase in YAML, snake_case in Python — `_snake()` and `_build()` in `config.py` bridge the two.
- Health/models go directly to pods; inference goes through the gateway — the EPP only routes inference requests.
- Metrics scraping tries `kubectl exec` first (python3, wget), falls back to port-forward + httpx for minimal container images.
- Global pytest timeout is 21600s (6 hours) to accommodate slow model downloads and pod startup.

### Gateway namespace access (xKS / rhoai-3.5+)

As of [odh-gitops PR#156](https://github.com/opendatahub-io/odh-gitops/pull/156), the inference gateway uses a label selector instead of `allowedRoutes.namespaces.from: All`. The helm chart install must include:

```bash
--set-json 'components.kserve.gateway.allowedRoutes.namespaces={"from":"Selector","selector":{"matchLabels":{"inference-gateway-access":"true"}}}'
```

`ensure_namespace()` labels the test namespace with `inference-gateway-access=true` (idempotent, applied on every run) so HTTPRoutes from `llm-conformance-test` are accepted by the gateway. No cluster-admin gateway patching is required or performed.

## Config files

- **configs/testcases/*.yaml** — one `TestCase` per file; anatomy in `configs/testcases/README.md`, per-case documentation in `docs/test-cases.md`
- **configs/profiles/*.yaml** — named groups of case names; anatomy in `configs/profiles/README.md`
- **deploy/manifests/** — LLMInferenceService manifests, cloned via `--setup` (gitignored); `.manifest-ref` tracks the active branch/commit

## CI

GitHub Actions (`.github/workflows/ci.yaml`) runs on push/PR to `main`:
1. **lint-and-format** — `ruff check` + `ruff format --check`
2. **smoke-tests** — clones manifests, runs unit tests (`pytest tests/ --ignore=tests/test_conformance.py`)

No cluster integration tests run in CI.

## Code Style

- Ruff for linting and formatting, line length 120, target Python 3.11.
- Uses `from __future__ import annotations` throughout.
- Config types are plain dataclasses (no Pydantic).
