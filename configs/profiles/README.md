# Profiles

Each YAML file in this directory is a **named group of test cases**. A profile
lists case names from `configs/testcases/`; it does not redefine models or
manifests. Loaded as `TestProfile` (`src/conformance/config.py`).

## How to run

```bash
uv run llm-d-e2e -p configs/profiles/smoke.yaml
uv run llm-d-e2e --list-profiles
```

`resolve_profile()` loads every YAML under `--testcase-dir`, then keeps only
the names listed in `testCases` (unknown names are dropped silently).

Prefer `-p` for CI / release suites; use `-t` when you need one or two cases.

## Anatomy

| Field | Purpose |
|-------|---------|
| `name` / `description` | Identity; shown by `--list-profiles` |
| `platform` | Hint (`any`, `ocp`, …); not a hard filter today |
| `testCases` | Ordered list of case **names** from `configs/testcases/` |
| `parallel` | Reserved (suite still runs cases sequentially via pytest) |
| `timeout` | Suite budget string (`30m`, `2h`, …) |

## Available profiles

| Name | Cases | Notes |
|------|-------|--------|
| `smoke` | `single-gpu-smoke` | Fastest path; good first check |
| `lora` | `lora-single`, `lora-multi` | LoRA adapter registration + inference |
| `cache-aware` | `cache-aware` | Prefix KV cache-aware routing |
| `flow-control` | `flow-control`, `flow-control-tokens` | EPP flow control variants |
| `pd` | `pd` | Prefill/decode disaggregation |
| `pd-performance` | `pd-performance` | GuideLLM P/D benchmarks |
| `moe` | `moe` | Needs 8 GPUs + RDMA/RoCE |
| `maas-3.5` | `maas-single-gpu` | RHOAI 3.5 MaaS suite; needs the MaaS stack (RHCL/Kuadrant, maas-api, maas-default-gateway) |
| `maas-3.6` | same as `maas-3.5` | RHOAI 3.6 MaaS suite; add 3.6-only MaaS cases here |
| `3.4` | single-gpu (+ no-scheduler), cache-aware | RHOAI 3.4 suite |
| `3.5` | single-gpu, cache-aware, flow-control*, pd, lora*, kv-offloading* | RHOAI 3.5 (auto-skips tests needing more GPUs than available) |
| `3.6` | same as `3.5` | RHOAI 3.6 EA2 has no differences from 3.5 |
| `all` | Broad set (scheduler, P/D, flow-control, lora) | Full conformance sweep |

Exact descriptions and timeouts are in each file.

## Adding a profile

1. Create `configs/profiles/<name>.yaml` with `name`, `description`, and a
   `testCases` list of existing case names.
2. Only reference names that exist under `configs/testcases/`.
3. Keep release profiles (`3.4`, `3.5`, `3.6`) aligned with what that
   product train actually ships.

See `configs/testcases/README.md` for case YAML structure and metrics flags.
