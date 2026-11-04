# Test cases

Each YAML file in this directory is one **conformance test case**: a data-driven
spec for deploying an `LLMInferenceService` and validating it through the
ordered phases in `tests/test_conformance.py`.

Cases are loaded into the `TestCase` dataclass (`src/conformance/config.py`).
YAML keys are **camelCase**; Python fields are **snake_case**.

## How they are selected

```bash
uv run llm-d-e2e -t single-gpu-smoke          # one case (by name)
uv run llm-d-e2e -t single-gpu,cache-aware    # several
uv run llm-d-e2e -p configs/profiles/smoke.yaml   # profile → list of names
uv run llm-d-e2e --list-testcases             # names + descriptions
```

With neither `-t` nor `-p`, every case runs except those with `optIn: true`,
which run only when named or listed in a profile.

Profiles under `configs/profiles/` only list test case **names** from this
folder. Manifests live in `deploy/manifests/` (cloned via `--setup`); each case
points at one file with `deployment.manifestPath`.

## Anatomy

| Section | Purpose |
|---------|---------|
| `name` / `description` | Identity; `description` is printed by `--list-testcases` |
| `model` | HF name/URI, optional PVC `cache`, optional `lora` adapters |
| `deployment` | Manifest path, replicas, resources, timeouts, optional `prefill` / `envOverrides` |
| `validation` | Health, prompts / `chatPrompts`, `metricsCheck` flags, optional `benchmark` |
| `cleanup` | Whether phase 99 deletes the deployment |
| `optIn` | Run only when named or listed in a profile, never in a run that selects no cases |

Duration strings (`15m`, `2h`, `300s`) are accepted on timeout fields.

### Metrics flags (`validation.metricsCheck`)

Enable only what the topology proves:

| Flag | Phase | Topology |
|------|-------|----------|
| `checkVLLM` | 10 | Any workload |
| `checkPrefixCache` | 11 | Prefix KV cache / cache-aware routing |
| `checkPD` | 12, 21 | Prefill/decode disaggregation |
| `checkScheduler` | 13 | EPP / scheduler present |
| `checkFlowControl` | 14 | Flow-control EPP |
| `checkLora` | 15 | LoRA adapters on vLLM |
| `checkNIXL` | — | Reserved (no validator yet) |

## Available cases

| Name | Focus |
|------|--------|
| `single-gpu-smoke` | Fast 1-GPU smoke |
| `single-gpu` | 1 GPU with scheduler |
| `single-gpu-no-scheduler` | 1 GPU, K8s-native routing (no EPP) |
| `cache-aware` | Multi-replica prefix KV cache-aware routing |
| `pd` | Prefill/decode disaggregation |
| `pd-cache-aware` | P/D + cache-aware routing |
| `pd-performance` | GuideLLM P/D performance thresholds |
| `flow-control` / `flow-control-tokens` | EPP flow control |
| `lora-single` / `lora-multi` | LoRA adapter registration + inference |
| `moe` | MoE with DP/EP |
| `multi-pool` | Multiple InferencePools |
| `maas-single-gpu` | single-gpu through MaaS (auth, API key, rate limit); `validation.maas.endpointScheme` |
| `modelexpress-p2p` | ModelExpress native loading: P2P scale-out, server TLS, token auth, NetworkPolicy |
| `modelexpress-p2p-xks` | `modelexpress-p2p` for xKS without RDMA: P2P over TCP, plaintext server, token auth; `resources.rdma: false` |

Exact descriptions are in each file’s `description` field.

## Adding a case

```bash
scripts/new-testcase.sh <name>
```

That scaffolds a YAML here and a manifest stub. Then:

1. Set `deployment.manifestPath` to a real file in the manifests repo / `deploy/manifests/`.
2. Turn on the right `metricsCheck` flags for the topology.
3. Add the name to relevant profiles under `configs/profiles/`.

See the project root `CLAUDE.md` / README for full CLI flags (`--mock`, `--mode`, PVC cache, etc.).
