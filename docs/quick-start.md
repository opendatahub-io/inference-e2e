# Quick start

From clone to a passing conformance run in a few commands, natively or in a
container. For what each test case does see `docs/test-cases.md`; for failures
see `docs/troubleshooting.md`.

## What you need

**Where you run:**

- Python 3.11+ and [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- kubectl or oc — only for the checks below; the suite itself is self-contained

**Cluster** (llm-d / RHOAI with KServe):

- the `LLMInferenceService` CRD (`llminferenceservices.serving.kserve.io`)
- a Gateway API `Gateway` — the suite finds the one each service routes
  through automatically; nothing to configure
- free GPUs: 1 for the single-GPU cases, 3 for `pd`, more for the rest — cases
  that don't fit are skipped, never left Pending
- egress to huggingface.co from the workload nodes (vLLM downloads the model
  on first run)
- a kubeconfig whose user can create namespaces, SAs/roles/rolebindings,
  clusterrolebindings, LLMInferenceServices, jobs and PVCs

Read-only sanity check:

```bash
export KUBECONFIG=/path/to/kubeconfig
kubectl cluster-info
kubectl get crd llminferenceservices.serving.kserve.io
kubectl get gateway -A                                     # at least one PROGRAMMED True
```

## Setup

```bash
git clone https://github.com/opendatahub-io/inference-e2e.git && cd inference-e2e
uv sync
uv run llm-d-e2e --setup main     # clone test manifests (or a branch: 3.5-GA, 3.6-ea2, ...)
```

## First run

```bash
uv run llm-d-e2e -t single-gpu-smoke -v
```

This deploys one Qwen3-0.6B replica and walks the phases — deploy, service,
gateway, pods, ready, health, models, one inference, cleanup — then deletes
everything it created. First runs are slow (vLLM image pull + model download);
later runs take a few minutes. A JSON summary lands in `reports/`.

Select cases with `-t name` or `-p configs/profiles/<name>.yaml`; list them
with `--list-testcases` and `--list-profiles`.

## Run modes

| Mode | What it does |
|---|---|
| `deploy` (default) | Full lifecycle: deploy → validate → cleanup |
| `discover` | Validate an existing deployment — no deploy, no cleanup |

```bash
# Validate a service that is already running
uv run llm-d-e2e -t single-gpu --mode discover --namespace my-ns
```

## Mock mode (no GPUs)

`--mock` swaps vLLM for the
[llm-d-inference-sim](https://github.com/llm-d/llm-d-inference-sim) simulator
and strips GPU requests, exercising the whole platform (deploy, routing,
inference path) without GPUs. Cases with `requiresGpu` — `pd`, `moe`,
`kv-offloading-*`, `agentic-serving`, `pd-performance` — are skipped: they
exist to validate real GPU behavior.

```bash
uv run llm-d-e2e -t single-gpu-smoke --mock -v
```

## Authentication

By default the suite creates a short-lived service account per test case,
scoped to `get` that case's `LLMInferenceService`, and sends its token with
inference requests. Alternatives: `--bearer-token <token>` to use an existing
identity, or `--disable-auth` for anonymous access.

## Pull secrets

The manifests reference `rhai-pull-secret`, which the suite copies into the
test namespace from `rhaii`, `redhat-ods-applications`, or `default` if it
exists there. On clusters whose nodes pull images without pod-level auth
(registry mirror, cluster-wide pull secret), a placeholder in `default` is
enough — the name just has to resolve:

```bash
kubectl -n default create secret docker-registry rhai-pull-secret \
  --docker-server=dummy.invalid --docker-username=x --docker-password=x
```

## Reports

Every run writes `reports/report.json`; add `--html reports/run.html` for an
HTML report. Rerun a failure with `--nocleanup` to inspect what's deployed.

## Running in a container

The suite can be run using the container image:

```bash
podman build -t llm-d-e2e:local .        # bake manifests from main; --build-arg MANIFEST_REF=3.6-ea2 for a branch

mkdir -p reports
podman run --rm -it \
  -v "$(dirname "$KUBECONFIG")":/kube:ro,z \
  -e KUBECONFIG="/kube/$(basename "$KUBECONFIG")" \
  -v "$PWD/reports":/app/reports:z \
  llm-d-e2e:local -t single-gpu-smoke -v
```

Everything after the image name goes to the `llm-d-e2e` CLI — the same flags
as the native commands above. To use different manifests at runtime, add
`--setup <branch>` (clones from GitHub inside the container); without it, the
baked-in manifests are used.
