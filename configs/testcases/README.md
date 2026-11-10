# Test cases

Each YAML file in this directory is one **conformance test case**: the spec
for deploying an `LLMInferenceService` and validating it through the
ordered phases in `tests/test_conformance.py`.

Cases are loaded into the `TestCase` dataclass (`src/conformance/config.py`).
YAML keys are **camelCase**; Python fields are **snake_case**.

## How they are selected

```bash
uv run llm-d-e2e -t single-gpu-smoke              # one case (by name)
uv run llm-d-e2e -t single-gpu,cache-aware        # several
uv run llm-d-e2e -p configs/profiles/smoke.yaml   # profile → list of names
uv run llm-d-e2e --list-testcases                 # names + GPU counts + descriptions
```

Profiles under `configs/profiles/` only list test case **names** from this
folder. Manifests live in `deploy/manifests/` (cloned via `--setup`); each case
points at one file with `deployment.manifestPath`.

## Anatomy

| Section | Purpose |
|---------|---------|
| `name` / `description` | Identity; `description` feeds the docs summary table |
| `model` | HF name/URI, optional PVC `cache`, optional `lora` adapters |
| `deployment` | Manifest path, `requiresGpu`, timeouts, resources, optional `prefill` |
| `validation` | Health, prompts / `chatPrompts`, `metricsCheck` flags, optional `benchmark` |
| `cleanup` | Whether phase 99 deletes the deployment |

Duration strings (`15m`, `2h`, `300s`) are accepted on timeout fields.

What every case deploys and verifies — including which metrics flags to enable
for which topology — is documented in `docs/test-cases.md` and
`docs/developing.md`. Keep that file's summary table in sync when you add or
change a case, and don't duplicate case lists anywhere else.

## Adding a case

```bash
scripts/new-testcase.sh <name>
```

Then follow `docs/developing.md`.
