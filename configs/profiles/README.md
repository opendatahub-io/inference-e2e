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

The available profiles are whatever is in this directory — run
`uv run llm-d-e2e --list-profiles` for the current list. Keep release profiles
(`3.4`, `3.5`, `3.6`) aligned with what that product train actually ships.

## Adding a profile

1. Create `configs/profiles/<name>.yaml` with `name`, `description`, and a
   `testCases` list of existing case names.
2. Only reference names that exist under `configs/testcases/`.

See `docs/test-cases.md` for what each case does.
