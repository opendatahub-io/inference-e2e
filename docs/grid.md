# Grid endpoint smoke checks

The Grid suite validates an **existing consumer endpoint**. It sends OpenAI chat
completion requests and checks inference responses plus Grid's site/backend
attribution. It requires no deployment manifests or Kubernetes access from the
runner. Provision the Grid and its inference backends separately.

## Run

```bash
uv sync
uv run llm-d-e2e --suite grid --mode discover \
  --endpoint http://127.0.0.1:8888 \
  --html reports/grid-smoke.html
```

Use the consumer's base URL, without `/v1/chat/completions`. A base path is
supported when the consumer is exposed below a URL prefix. Supply
`--bearer-token` if the consumer authenticates callers; no service accounts or
RBAC resources are created. `--disable-auth` is unnecessary for this suite.
The shared LLMClient currently disables TLS certificate verification, as it does
for other conformance clients; use a trusted test endpoint.

The built-in sample is `configs/grid/testcases/grid-smoke.yaml`. Copy it into a
separate testcase directory and customize the model and allowed route pairs:

```yaml
name: grid-smoke
model:
  name: my-model
validation:
  timeout: 30s
  testPrompts: [Hello]
  grid:
    requestCount: 10
    maxTokens: 16
    expectedRoutes:
      site-a: [provider-a, provider-b]
      site-b: [provider-c]
cleanup: false
```

```bash
uv run llm-d-e2e --suite grid --mode discover \
  --endpoint http://grid.example:8080 \
  --testcase-dir ./my-grid-cases --testcase grid-smoke
```

`--model` overrides the configured served model name. Expected site and backend
values must match Grid's response attribution, not Kubernetes pod names.
Request counts and output-token limits must be positive integers. Every configured
site needs a nonempty backend list.

To use pytest directly:

```bash
uv run pytest tests/grid/test_grid.py --suite grid --mode discover \
  --endpoint http://127.0.0.1:8888 --junitxml=reports/grid-smoke.xml
```

## Assertions and scope

Every request must return HTTP 200 with the requested model, `chat.completion`,
and at least one nonempty assistant text choice. Both `x-grid-site` and
`x-grid-backend` must be present and form an allowed pair in `expectedRoutes`.
For example, listing provider-a under site-a does not permit a response claiming
site-b/provider-a. Header names are case-insensitive.

The test prints counts for the observed routes. It does not require every allowed
backend to be sampled, assert a load-balancing ratio, or retry failed requests.
It does not prove EPP participation, operator configuration publication, failover,
or multi-cluster networking. Those need additional checks in later contributions.

Grid checks do not invoke deployment, pod discovery, auth/RBAC creation or cleanup
fixtures. The CLI requires both `--mode discover` and `--endpoint`, and rejects
`--setup`. Default conformance testcase/profile selection remains unchanged;
Grid cases live in their own directory. The standard offline test command
continues to deselect the live Grid checks.

## Local verification

This suite was run against a Kind deployment using operator-managed routing
through an mTLS provider gateway and two independent llm-d/EPP/inference-sim
backends. All ten requests returned valid attributed inference responses, with
five responses from each backend. That run validates these smoke assertions;
the test itself makes no claim about the internal deployment topology.
