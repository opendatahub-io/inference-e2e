# Adding a Batch Gateway Test Case

How the Batch Gateway conformance case works, what the platform must provide,
and what the manifest in
[llm-d-conformance-manifests](https://github.com/opendatahub-io/llm-d-conformance-manifests)
must contain.

The skeleton validates **one** `LLMInferenceService` plus **one**
`LLMBatchGateway` using **async dispatch over the PostgreSQL (SQL) transport**:
Postgres holds both the batch control-plane state and the async request/result
queues, so no Redis is involved. Redis transport, multi-tenant/MaaS batches,
cancel/expiry/GC coverage, and batch metrics validation are deliberately out of
scope.

## What the suite validates

`batch-gateway-async-sql` keeps every llm-d phase (01-21) and adds eleven batch
phases that run **after** the model is Ready and **before** cleanup. They are
ports of the upstream [`llm-d-batch-gateway`](https://github.com/llm-d/llm-d-batch-gateway)
e2e suite (`test/e2e/*.go`), so a failure here means the platform diverged from
behaviour upstream already guarantees.

| Phase | Method | Ported from | What it validates |
|-------|--------|-------------|-------------------|
| 40a | `test_40a_batch_ready` | operator CRD contract | `LLMBatchGateway` exists and reports `Ready=True` with `APIServerAvailable`, `ProcessorAvailable`, `GCAvailable` (plus `AsyncProcessorAvailable` when async dispatch is configured). Polls for `deployment.readyTimeout` and fails fast when one reason+message persists. |
| 40b | `test_40b_batch_files` | `Files/Lifecycle` | `POST/GET/GET content/DELETE /v1/files`: upload a small JSONL, check `purpose=batch` and `bytes`, read the content back byte-for-byte, delete it (no residue, GET is then 404). |
| 40c | `test_40c_batch_inference` | `Batches/Lifecycle` | Upload a JSONL built from `validation.testPrompts`, create a batch, poll to a terminal status, require `completed` with `request_counts.total == expectOutputs`, then parse the output JSONL: one line per request, matching `custom_id`s, HTTP 200 per response, non-empty assistant content. |
| 40d | `test_40d_batch_files_contract` | `Files/Create/DuplicateFilename`, `Files/List/Pagination`, `Files/List/PurposeFilter` | Under a dedicated tenant: two uploads of one filename get distinct ids and keep the original `filename`; `purpose=batch` filters; `limit` + `after` page in creation order with a correct `has_more`, and `after` also accepts an integer offset. |
| 40e | `test_40e_batch_validation` | `Batches/Validation/*` | Creating a batch with a missing `input_file_id`, an unsupported `endpoint`, or an unknown file id returns **400** with an OpenAI-shaped `error.message`. |
| 40f | `test_40f_batch_ingestion_rejections` | `Batches/Ingestion/DuplicateCustomID`, `StreamingRejected` | Duplicate `custom_id`s and `"stream": true` requests fail **ingestion**, so the batch reaches `failed` (not `completed` with per-request errors). |
| 40g | `test_40g_batch_partial_results` | `Batches/MixedSuccessFailure`, `Batches/Ingestion/AllModelsUnregistered` | Mixed file (2 good, 1 unknown model) completes with `total=3 completed=2 failed=1`, an `output_file_id` **and** an `error_file_id` listing only `bad-1`; an all-unregistered file completes with `failed=2` and an error file containing `model_not_found`. |
| 40h | `test_40h_batch_file_reuse` | `Batches/TrailingNewline`, `Batches/SharedInputFile` | A JSONL without a trailing newline ingests 2/2; one input file can feed two batches, both completing with distinct output files. |
| 40i | `test_40i_batch_cancel` | `Batches/Cancel/{BeforeProcessing,IdempotentRetry,TerminalBatchRejected}` | Cancelling right after submission returns `cancelling` or `cancelled`, a retry is accepted, the batch settles at `cancelled` with nothing completed and no output file; cancelling a `completed` batch is rejected with 400. |
| 40j | `test_40j_batch_multi_tenant` | `MultiTenant/Isolation` | A second tenant cannot read this tenant's file (404) or batch (404), does not see them in its lists, and cannot create a batch over this tenant's file (400). |
| 40k | `test_40k_batch_concurrency` | `Concurrent/MultipleBatches` | `validation.batchGateway.concurrency` batches are submitted back-to-back and all complete 2/2 with distinct output files. Set `concurrency: 0` to deselect this phase. |

All phases are gated with `@pytest.mark.applies_when`, so they are **deselected**
(not skipped) for every test case that does not set
`validation.batchGateway.enabled: true`.

### `--batch-gateway auto|require`

The platform supplies the CRD and operator; the suite never installs them. The
flag decides what happens when that platform support is missing:

| Value | Behaviour |
|-------|-----------|
| `auto` (default) | If the CRD or operator is missing, the test case **skips at deploy (phase 02)** and prints the platform diagnostic, so CPU-only or pre-3.6 clusters can run the rest of the profile. |
| `require` | Same diagnostic, but the test case **fails at deploy (phase 02)** instead of skipping. Use this in a CI job that is supposed to prove batch support. |

The diagnostic (`batch_platform_state()`) reports, in order: whether
`llmbatchgateways.batch.llm-d.ai` exists, whether the operator Deployment is
present in `redhat-ods-applications` / `opendatahub`, and the
`DataScienceCluster` field
`spec.components.aigateway.batchGateway.managementState`. `require` does not
override the test case's own gates: a test case without
`validation.batchGateway.enabled` is still deselected, and a missing manifest
still skips at phase 01 (see [Pending manifest](#pending-manifest)).

The probe runs in **phase 02**, after that phase's own skips (`--mode discover`,
missing manifest, too few GPUs, no `LLMInferenceService` CRD) and before
`kubectl apply`, for every test case with `validation.batchGateway.enabled`.
Without it, `kubectl apply` would fail with `no matches for kind
"LLMBatchGateway"` and leave the `LLMInferenceService` behind, so `auto` would
report a deploy failure instead of a skip. Test cases without batch validation
never probe the platform. Phase 40a re-checks through `require_batch_target()`,
so a `--mode discover` run, where phase 02 is skipped, gates identically.

## Platform prerequisites

The target platform is RHOAI 3.6 / ODH nightly with the `AIGateway` component.
Enabling `batchGateway` there installs:

- CRD `llmbatchgateways.batch.llm-d.ai` (`apiVersion: batch.llm-d.ai/v1alpha1`),
- the `llm-d-batch-gateway-operator` Deployment in `redhat-ods-applications`
  (ODH: `opendatahub`), whose pod carries
  `control-plane=llm-d-batch-gateway-controller-manager`,
- the operands the operator renders for each `LLMBatchGateway` CR: an API server
  (`app.kubernetes.io/name: batch-gateway-apiserver`, Service
  `<cr-name>-apiserver` on port 8000), a processor StatefulSet, an async
  processor, and a GC Deployment.

Two more platform properties decide whether the *dispatch* path works, and the
suite can only report them, not fix them — all verified on a clean 3.6-EA2
install:

- **Pre-GA API drift.** RHOAI **3.6-EA2** also predates two behaviours phases 40d
  and 40g assert: `GET /v1/files?after=<file id>` returns **400** there (the id
  cursor for file pagination is newer), and a batch with a mix of successes and
  failures completes with `request_counts.failed > 0` but no `error_file_id`.
  Everything else in 40a-40k (files lifecycle, batch to `completed`, ingestion
  rejections, cancel, tenant isolation, concurrency) does pass on EA2 when the
  manifest is adapted to its schema — the crudest gaps are that `dispatchMode:
  sync` still needs a Redis exchange (`redis-url` + a Redis instance, otherwise
  the components die with `redis config has empty url`), `globalInferenceGateway`
  and `modelGateways` are mutually exclusive, and `processor.config` has no
  `progressUpdateInterval` (it is `progressTTLSeconds`).
- **Async transport.** The CRD has to accept the transport the manifest declares:
  `spec.processor.asyncConfig.sql`. 3.6-**EA2**'s CRD only carries
  `asyncConfig.redis`, so applying the SQL manifest fails with
  `strict decoding error: unknown field "spec.processor.asyncConfig.sql"`. The
  SQL transport arrives with 3.6 GA / ODH nightly. `require_batch_platform()`
  now checks the CRD schema for it, so `--batch-gateway auto` skips with that
  explanation instead of failing the deploy.
- **Which Gateway the model's route lives on.** KServe names it in
  `inferenceservice-config` → `ingress.kserveIngressGateway`
  (`namespace/gateway-name`). On 3.6-EA2 that is
  `openshift-ingress/openshift-ai-inference`, a Gateway the `AIGateway` component
  does **not** create — it creates `openshift-ingress/data-science-gateway` — so
  unless something creates the configured one, every `LLMInferenceService` stays
  `RefsInvalid: Managed HTTPRoute references non-existent Gateway`. The suite
  derives its dispatch URL from that same ConfigMap entry, because a platform can
  run several gateways and only one carries the model's route.
- **Route admission.** The `LLMInferenceService`'s `HTTPRoute` is created in the
  test namespace, so the gateway's `allowedRoutes` has to accept routes from it
  (the xKS chart does this with a `matchLabels: {inference-gateway-access: "true"}`
  selector that `Deployer.ensure_namespace()` satisfies). `data-science-gateway`
  selects by namespace *name* (`openshift-ingress`, `redhat-ods-applications`)
  and the `ai-gateway-operator` reverts any patch within seconds, so the route is
  rejected with `NotAllowedByListeners`.
- **Authentication.** `data-science-gateway` sits behind `oauth2-proxy` and
  answers the async processor's unauthenticated dispatch with `302` to OpenShift
  OAuth; `openshift-ai-inference` is plain HTTP on 80. Either the route is
  auth-exempt (`--disable-auth`, plus the `AuthPolicy` CRD from the MaaS stack
  being installed) or the processor carries a credential in
  `inference-api-key`.

### Exact DataScienceCluster patch

```bash
# Replace default-dsc with your DataScienceCluster name.
kubectl patch datasciencecluster default-dsc --type=merge -p '{
  "spec": {
    "components": {
      "aigateway": {
        "managementState": "Managed",
        "batchGateway": {
          "managementState": "Managed"
        }
      }
    }
  }
}'
```

Equivalent in the DSC spec:

```yaml
apiVersion: datasciencecluster.opendatahub.io/v2
kind: DataScienceCluster
metadata:
  name: default-dsc
spec:
  components:
    aigateway:
      managementState: Managed
      batchGateway:
        managementState: Managed
```

### Verify the platform before running

```bash
# CRD registered
kubectl get crd llmbatchgateways.batch.llm-d.ai

# Operator running (RHOAI namespace; use opendatahub on ODH)
kubectl get deploy -n redhat-ods-applications llm-d-batch-gateway-operator
kubectl get pods -n redhat-ods-applications \
  -l control-plane=llm-d-batch-gateway-controller-manager

# DSC projection
kubectl get datasciencecluster default-dsc \
  -o jsonpath='{.spec.components.aigateway.batchGateway.managementState}{"\n"}'
```

If any of these are missing, run with the default `--batch-gateway auto` and the
case will skip; with `--batch-gateway require` it fails and shows the diagnostic
above.

The API server may be TLS-terminated depending on `spec.tls` (disabled by
default, which is what the example below assumes). If the manifest enables TLS
with a self-signed certificate, set `validation.batchGateway.verifyTls: false`.

## Test case config

`configs/testcases/batch-gateway-async-sql.yaml` sets the batch-specific knobs
under `validation.batchGateway` (camelCase in YAML, `BatchGatewayCheck` in
`src/conformance/config.py`):

| Key | Default | Purpose |
|-----|---------|---------|
| `enabled` | `false` | Turns phases 40a-40k on for this test case. |
| `endpoint` | `""` | Explicit API server base URL. Empty derives it: port-forward to `svc/<testcase-name>-apiserver` on 8000 in the test namespace, falling back to the gateway route. |
| `dispatchUrl` | `""` | Pins `asyncConfig.inferenceGateway.url` (and every `queuesConfig[].igwBaseURL`) **verbatim** instead of deriving it from the cluster. See [Dispatch URL and TLS](#dispatch-url-and-tls). |
| `transport` | `sql` | Async queue transport the manifest declares, checked against the installed CRD (`asyncConfig.sql`). Older operators only ship `redis`, so `auto` skips and `require` fails with a diagnostic instead of a rejected apply. |
| `tenant` | `default` | Value of the tenant header (`X-MaaS-Username` by default on the API server). |
| `requestTimeout` | `10m` | Budget for each HTTP call, including `wait_for_terminal`. Duration string. |
| `pollInterval` | `15s` | Batch status poll interval. Duration string. |
| `completionWindow` | `24h` | `completion_window` sent on `POST /v1/batches`. |
| `model` | `""` | Model name in the batch request bodies; empty means `model.name` from the test case. |
| `prompts` | `[]` | Prompts to submit; empty means `validation.testPrompts`. |
| `expectOutputs` | `0` | Expected `request_counts.total` and output line count; `0` means `len(prompts)`. |
| `verifyTls` | `true` | Verify the API server certificate (set `false` for self-signed TLS). |
| `concurrency` | `3` | Batches submitted by phase 40k; `0` deselects that phase. |
| `cancelRequests` | `8` | Requests in the batch that phase 40i cancels immediately after submitting. |

`deployment.requiresGpu` stays `false`: the batch gateway components are
CPU-only, and `--mock` strips the model's GPU request so the whole flow runs on
CPU-only CI. On a real cluster the `LLMInferenceService` in the manifest still
needs a GPU unless you run `--mock`.

### Dispatch URL and TLS

The async processor runs in the cluster, so the route it dispatches to must be
cluster-internal — the test runner's port-forward is not reachable for it. At
deploy time `Deployer._patch_batch_dispatch()` therefore rewrites
`spec.processor.asyncConfig.inferenceGateway.url` (and every declared
`spec.processor.asyncConfig.sql.queuesConfig[].igwBaseURL`; a queue without one
inherits the former) to

```text
<scheme>://<gateway-service>.<gateway-ns>.svc.cluster.local:<port>/<test-namespace>/<llmisvc>
```

built from `Deployer._dispatch_gateway_service()`, which resolves, in order: the
Gateway KServe attaches `LLMInferenceService` routes to —
`inferenceservice-config` → `ingress.kserveIngressGateway` (`namespace/name`,
read from `redhat-ods-applications` or `opendatahub`) resolved to its Service via
the `gateway.networking.k8s.io/gateway-name` label; then the first gateway-labelled
Service in the cluster; then `svc/inference-gateway-istio` in
`redhat-ods-applications`. The ConfigMap wins because a platform can run more
than one Gateway — 3.6-EA2 has both the OAuth-protected `data-science-gateway`
and the plain-HTTP `openshift-ai-inference` — and only one carries the model's
route. Port **443** yields
`https://…:443` and also sets `asyncConfig.tls.insecureSkipVerify` to `true`
(the certificate is signed by the OpenShift service CA, which the batch-gateway
Go binaries do not trust); every other port yields `http://…` and leaves the
manifest's `tls` block untouched. Manifests without an `LLMBatchGateway` skip
discovery entirely, so non-batch test cases make no extra `kubectl` call.

`validation.batchGateway.dispatchUrl` supplies the URL verbatim, for routing
that is not reachable at the derived address or an external hostname. It skips
gateway discovery and leaves the manifest's TLS block exactly as written. There
is no CLI flag — set it in the test case YAML. `verifyTls` is unrelated: it
covers the suite's own calls to the API server.

## Manifest ownership

`deploy/manifests/` is cloned (and gitignored) — manifests are owned by the
[llm-d-conformance-manifests](https://github.com/opendatahub-io/llm-d-conformance-manifests)
repo, not by this one. The test case points at `batch-gateway-async-sql.yaml`
there:

```bash
uv run llm-d-e2e --setup main   # or a branch/fork that has the manifest
uv run llm-d-e2e --list-testcases | grep batch-gateway
```

Add the manifest to `main` first, then cherry-pick it to any release branch
where the batch gateway ships. While developing, `--setup my-branch
--manifest-repo <fork URL>` pulls it from your fork.

### Pending manifest

Until the manifest lands on the branch you pulled:

- `uv run llm-d-e2e --list-testcases` still lists `batch-gateway-async-sql`
  (GPU count shows `?` because the manifest cannot be read),
- `test_01_prereq` **skips** with
  `manifest not found for this branch: batch-gateway-async-sql.yaml`, and every
  later phase, including 40a-40k, skips with it,
- `--batch-gateway=require` does **not** turn that skip into a failure: the
  manifest gate runs in phase 01, before the platform/batch checks. Only a
  deployed test case can fail on missing batch platform support.

### Disconnected clusters

The manifest pulls two images the platform does not ship:
`docker.io/library/postgres:16-alpine` (batch control-plane state and SQL
queues) and `docker.io/chrislusf/seaweedfs:4.48` (S3-compatible file storage).
Both have to be mirrored before the case runs — e.g. list them in
`OC_MIRROR_EXTRA_IMAGES` for the QE pipelines — otherwise the Postgres and
SeaweedFS pods stay in `ImagePullBackOff` and every batch phase fails on the
API server being unreachable.

## Required manifest contents

Every object is namespace-scoped and lives in the test namespace (default
`llm-conformance-test`). Documents without `metadata.namespace` are applied with
`kubectl apply -n <test-namespace>`, which is what the example below relies on.
Phase 99 deletes these documents in reverse manifest order before the
`LLMInferenceServices`, so list the dependencies first and the `LLMBatchGateway`
last.

1. **`LLMInferenceService`** — the model under test. Its name becomes the test
   case name, its InferencePool is `<name>-inference-pool`, and its route is what
   the batch gateway dispatches to.
2. **PostgreSQL** — StatefulSet + Service for control-plane state and the SQL
   async queues. A single instance with `POSTGRES_DB=batch` is enough.
3. **File storage** — S3-compatible object storage (SeaweedFS Deployment +
   Service).
4. **`batch-gateway-secrets` Secret** — credentials consumed by the operator:

   | Key | Required | Purpose |
   |-----|----------|---------|
   | `postgresql-url` | yes | `postgresql://user:pass@host:5432/batch?sslmode=disable` |
   | `s3-secret-access-key` | with `fileStorage.s3` | S3 secret key; the access key id is non-sensitive and lives in the CR |
   | `inference-api-key` | optional | Static API key the async processor sends to the inference gateway |
   | `redis-url` | no | Only for the Redis transport, which this skeleton does not cover |

   `spec.secretRef` is immutable after creation — to rotate it, recreate the CR.
5. **`LLMBatchGateway`** — `spec.dbBackend: postgresql`, `spec.fileStorage.s3`,
   `spec.apiServer`, `spec.processor` with
   `processor.modelGateways` and `processor.asyncConfig.sql`, and `spec.gc`.
   The CRD enforces these with CEL rules (there is no admission webhook).

Naming rules the suite relies on:

- the CR is named after the test case, so its API server Service is
  `<testcase-name>-apiserver` (port 8000) — that is what the `batch_client`
  fixture port-forwards to;
- `processor.modelGateways` is keyed by the **served model name** and points at
  the `LLMInferenceService`'s pool (`<name>-inference-pool`);
- `processor.asyncConfig.inferenceGateway.url` is the model's route through the
  inference gateway. The deployer **rewrites** it and the queue URLs from the
  gateway Service it discovers for the run, so the manifest value is only a
  placeholder — see [Dispatch URL and TLS](#dispatch-url-and-tls). A wrong
  `http://…:80` URL left in place on a 3.6 cluster points at a host that does not
  resolve, and batches stay `in_progress`;
- with `asyncConfig.sql.queuesConfig` set, its queue names must match the
  request queue derived for each model gateway
  (`llm-d-async:requests:<inferencePoolName>`); a `workerPoolID` must match a
  `asyncConfig.workerPools[].name`. A queue that omits `igwBaseURL` inherits
  `inferenceGateway.url`, so the suite leaves that key absent;

The async processor dispatches through the gateway like any other client, so the
gateway must accept its credential. With the suite's default service-account
auth, run the case with `--disable-auth` (which annotates the
`LLMInferenceService` for anonymous access) or provide a static token accepted by
your gateway in `inference-api-key`.

## Complete example manifest

Copy this into the manifests repo as `batch-gateway-async-sql.yaml`. It is
self-contained apart from the platform-provided batch CRD/operator.

```yaml
# Model under test. Same shape as single-gpu-smoke.yaml; adapt resources and
# pull secrets to your cluster.
apiVersion: serving.kserve.io/v1alpha2
kind: LLMInferenceService
metadata:
  name: batch-gateway-async-sql
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
            cpu: "2"
            memory: 8Gi
            nvidia.com/gpu: "1"
          requests:
            cpu: "1"
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
---
# PostgreSQL: batch control-plane state + SQL async queues.
# emptyDir keeps the example runnable without a StorageClass; swap in a PVC if
# you need the data to survive the run (the data is disposable for conformance).
apiVersion: v1
kind: Service
metadata:
  name: batch-gateway-postgres
  labels:
    app: batch-gateway-postgres
spec:
  selector:
    app: batch-gateway-postgres
  ports:
    - name: postgres
      port: 5432
      targetPort: 5432
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: batch-gateway-postgres
spec:
  serviceName: batch-gateway-postgres
  replicas: 1
  selector:
    matchLabels:
      app: batch-gateway-postgres
  template:
    metadata:
      labels:
        app: batch-gateway-postgres
    spec:
      containers:
        - name: postgres
          image: docker.io/library/postgres:16-alpine
          env:
            - name: POSTGRES_PASSWORD
              value: postgres
            - name: POSTGRES_DB
              value: batch
          ports:
            - containerPort: 5432
          readinessProbe:
            exec:
              command: ["pg_isready", "-U", "postgres", "-d", "batch"]
            initialDelaySeconds: 5
            periodSeconds: 5
          volumeMounts:
            # Mount the volume at the parent of PGDATA, not at PGDATA itself:
            # OpenShift runs the container as a random UID, and initdb cannot
            # chmod a volume root that kubelet owns, so PostgreSQL would die with
            # "could not change permissions of directory /var/lib/postgresql/data".
            # As a subdirectory, the data dir is created and owned by that UID.
            - name: data
              mountPath: /var/lib/postgresql
      volumes:
        - name: data
          emptyDir: {}
---
# S3-compatible file storage (SeaweedFS).
apiVersion: apps/v1
kind: Deployment
metadata:
  name: batch-gateway-seaweedfs
  labels:
    app: batch-gateway-seaweedfs
spec:
  replicas: 1
  selector:
    matchLabels:
      app: batch-gateway-seaweedfs
  template:
    metadata:
      labels:
        app: batch-gateway-seaweedfs
    spec:
      containers:
        - name: seaweedfs
          image: docker.io/chrislusf/seaweedfs:4.48
          args:
            - server
            - -dir=/data
            - -ip.bind=0.0.0.0
            - -s3
            - -s3.port=8333
          env:
            - name: AWS_ACCESS_KEY_ID
              value: batch-gateway
            - name: AWS_SECRET_ACCESS_KEY
              value: batch-gateway-secret
          ports:
            - name: s3
              containerPort: 8333
          readinessProbe:
            tcpSocket:
              port: s3
            initialDelaySeconds: 5
            periodSeconds: 5
          volumeMounts:
            - name: data
              mountPath: /data
      volumes:
        - name: data
          emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  name: batch-gateway-seaweedfs
  labels:
    app: batch-gateway-seaweedfs
spec:
  selector:
    app: batch-gateway-seaweedfs
  ports:
    - name: s3
      port: 8333
      targetPort: 8333
---
# Credentials the operator mounts into every batch-gateway component.
apiVersion: v1
kind: Secret
metadata:
  name: batch-gateway-secrets
type: Opaque
stringData:
  postgresql-url: postgresql://postgres:postgres@batch-gateway-postgres:5432/batch?sslmode=disable
  s3-secret-access-key: batch-gateway-secret
  inference-api-key: dummy-api-key
---
# Batch gateway. Named after the test case so the API server Service is
# batch-gateway-async-sql-apiserver:8000.
apiVersion: batch.llm-d.ai/v1alpha1
kind: LLMBatchGateway
metadata:
  name: batch-gateway-async-sql
spec:
  secretRef:
    name: batch-gateway-secrets
  dbBackend: postgresql
  fileStorage:
    s3:
      bucket: batch-gateway
      region: us-east-1
      endpoint: http://batch-gateway-seaweedfs:8333
      accessKeyId: batch-gateway
      usePathStyle: true
      autoCreateBucket: true
  apiServer:
    replicas: 1
  processor:
    replicas: 1
    config:
      progressUpdateInterval: 15s
    # Keyed by served model name; the pool name comes from the LLMInferenceService.
    modelGateways:
      "Qwen/Qwen3-0.6B":
        inferencePoolName: batch-gateway-async-sql-inference-pool
    asyncConfig:
      concurrency: 8
      drainTimeout: 2m
      resultPollTimeout: 2m
      # Dispatch TLS. The suite forces insecureSkipVerify: true when it derives
      # an https:// dispatch URL (port 443), because the gateway certificate is
      # signed by the OpenShift service CA, which the batch components' Go
      # binaries do not trust; on an http:// (port 80) gateway it leaves this
      # block alone. This is a conformance fixture with dummy credentials; use
      # caCertKey with a real Secret elsewhere.
      tls:
        insecureSkipVerify: true
      # Route to the model through the inference gateway. The suite rewrites
      # this (and any queuesConfig[].igwBaseURL) at deploy time from the gateway
      # Service it discovers, using https + insecureSkipVerify on port 443 and
      # http on port 80, so this value is only a placeholder for a manual apply.
      inferenceGateway:
        url: https://data-science-gateway-data-science-gateway-class.openshift-ingress.svc.cluster.local:443/llm-conformance-test/batch-gateway-async-sql
      sql:
        pollIntervalMs: 1000
        batchSize: 10
        maxConnections: 8
        # One queue per model gateway; the name must match the queue the
        # processor publishes to (llm-d-async:requests:<inferencePoolName>).
        queuesConfig:
          - name: llm-d-async:requests:batch-gateway-async-sql-inference-pool
            workerPoolID: default
      workerPools:
        - name: default
          workers: 4
  gc:
    replicas: 1
    interval: 30m
```

## Run it

```bash
# Make sure the manifest is present on the branch you pulled (--setup and
# --list-testcases run independently; --list-testcases never clones)
uv run llm-d-e2e --setup main
uv run llm-d-e2e --list-testcases | grep batch-gateway

# Full flow in mock mode (no GPU): deploy, llm-d phases, batch phases, cleanup
uv run llm-d-e2e -t batch-gateway-async-sql --mock --disable-auth -v

# Require the batch platform; fail instead of skipping when it is missing
uv run llm-d-e2e -p configs/profiles/batch-gateway.yaml --batch-gateway require -v

# Just the batch phases against an already deployed test case
uv run pytest tests/test_conformance.py -k test_40 --testcase batch-gateway-async-sql -v
```
