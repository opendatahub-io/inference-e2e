"""Batch-gateway helpers: OpenAI-compatible files/batches API and platform diagnostics.

The ``AIGateway`` component (RHOAI 3.6 / ODH nightly) can install an
``LLMBatchGateway`` CRD and operator. The operator renders an OpenAI-compatible
API server that accepts JSONL batch jobs: files are uploaded with
``POST /v1/files`` (``purpose=batch``), jobs are submitted with
``POST /v1/batches``, and results are fetched from the completed batch's
``output_file_id``.

``BatchGatewayClient`` mirrors ``conformance.client.LLMClient``: a thin httpx
wrapper that raises for HTTP errors. Tenant identity travels in the
``X-MaaS-Username`` header. ``batch_jsonl`` builds the request body, and
``output_lines`` parses the result file.

The CRD, operator, Postgres, and object storage are platform/manifest
preconditions — this module only observes them. ``batch_platform_state`` and
``batch_operator_running`` are best-effort diagnostics for the phases' skip
versus fail decision; they never raise.

``dispatch_target`` / ``patch_batch_dispatch`` are the manifest half: the deployer
uses them to rewrite an ``LLMBatchGateway``'s async dispatch URL to the gateway
Service discovered in the cluster under test (HTTP on 80, HTTPS +
``tls.insecureSkipVerify`` on 443). ``validation.batchGateway.dispatchUrl``
overrides the derived URL verbatim.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable
from datetime import timedelta
from typing import Any

import httpx

BATCH_CRD = "llmbatchgateways.batch.llm-d.ai"
BATCH_API_VERSION = "batch.llm-d.ai/v1alpha1"
BATCH_OPERATOR_DEPLOYMENT = "llm-d-batch-gateway-operator"
# RHOAI installs into redhat-ods-applications; upstream ODH uses opendatahub.
BATCH_OPERATOR_NAMESPACES = ("redhat-ods-applications", "opendatahub")
BATCH_OPERATOR_LABEL = "control-plane=llm-d-batch-gateway-controller-manager"
BATCH_APISERVER_SERVICE_SUFFIX = "-apiserver"
TENANT_HEADER = "X-MaaS-Username"

TERMINAL_BATCH_STATUSES = frozenset({"completed", "failed", "cancelled", "expired"})

BATCH_GATEWAY_KIND = "LLMBatchGateway"
# Fields the async processor dispatches through; both may be set per model gateway.
DISPATCH_URL_PATH = "spec.processor.asyncConfig.inferenceGateway.url"
DISPATCH_QUEUE_URL_PATH = "spec.processor.asyncConfig.sql.queuesConfig[%d].igwBaseURL"


def dispatch_target(
    gateway_service: str,
    gateway_namespace: str,
    port: int,
    namespace: str,
    llmisvc: str,
) -> tuple[str, bool]:
    """``(url, insecure_tls)`` for dispatching to ``llmisvc`` through the inference gateway.

    ``gateway_service`` may carry the ``svc/`` prefix kubectl accepts. The URL is
    cluster-internal on purpose: the batch components run in the cluster, so a
    port-forward like the test runner uses is not reachable for them.

    Port 443 means the platform's gateway is TLS-only (RHOAI 3.6's AIGateway
    exposes ``data-science-gateway-<class>`` in ``openshift-ingress`` with only
    443/15021) and its certificate is signed by the OpenShift service CA, which
    the batch-gateway Go binaries do not trust — hence the second element, which
    asks the caller to set ``asyncConfig.tls.insecureSkipVerify``.
    """
    service = gateway_service.split("/", 1)[-1]
    host = f"{service}.{gateway_namespace}.svc.cluster.local:{port}"
    if port == 443:
        return f"https://{host}/{namespace}/{llmisvc}", True
    return f"http://{host}/{namespace}/{llmisvc}", False


def patch_batch_dispatch(documents: Iterable[object], *, url: str, insecure_tls: bool = False) -> list[str]:
    """Point every ``LLMBatchGateway``'s async dispatch at ``url``.

    Rewrites ``spec.processor.asyncConfig.inferenceGateway.url`` and every
    ``spec.processor.asyncConfig.sql.queuesConfig[].igwBaseURL`` the manifest
    declares — a queue without one inherits the inference gateway URL, so adding
    the key would change semantics. With ``insecure_tls`` the TLS block is also
    forced to ``insecureSkipVerify: true``.

    Returns the paths that were rewritten (empty for manifests with no
    ``LLMBatchGateway``, or with a sync passthrough that has no ``asyncConfig``),
    so the deployer can log the derived target.
    """
    rewritten: list[str] = []
    for document in documents:
        if not isinstance(document, dict) or document.get("kind") != BATCH_GATEWAY_KIND:
            continue
        spec = document.get("spec") if isinstance(document.get("spec"), dict) else {}
        processor = spec.get("processor") if isinstance(spec.get("processor"), dict) else {}
        async_config = processor.get("asyncConfig") if isinstance(processor.get("asyncConfig"), dict) else {}
        if not async_config:
            continue

        touched = 0
        inference_gateway = async_config.get("inferenceGateway")
        if isinstance(inference_gateway, dict):
            inference_gateway["url"] = url
            rewritten.append(DISPATCH_URL_PATH)
            touched += 1

        sql = async_config.get("sql") if isinstance(async_config.get("sql"), dict) else {}
        queues = sql.get("queuesConfig") if isinstance(sql.get("queuesConfig"), list) else []
        for index, queue in enumerate(queues):
            if isinstance(queue, dict) and "igwBaseURL" in queue:
                queue["igwBaseURL"] = url
                rewritten.append(DISPATCH_QUEUE_URL_PATH % index)
                touched += 1

        if insecure_tls and touched:
            tls = async_config.get("tls") if isinstance(async_config.get("tls"), dict) else {}
            async_config["tls"] = tls
            tls["insecureSkipVerify"] = True

    return rewritten


def _seconds(value: float | timedelta) -> float:
    """Accept a duration in seconds or as a ``timedelta``."""
    return value.total_seconds() if isinstance(value, timedelta) else float(value)


def _page_params(limit: int, after: str) -> dict[str, str]:
    """Query params for a list call; ``0``/empty means "server default"."""
    params: dict[str, str] = {}
    if limit:
        params["limit"] = str(limit)
    if after:
        params["after"] = after
    return params


def error_message(response: httpx.Response) -> str:
    """Extract the OpenAI-style ``error.message`` from a failed response (``""`` when absent)."""
    try:
        error = (response.json() or {}).get("error") or {}
    except ValueError:
        return ""
    if isinstance(error, dict):
        return str(error.get("message") or "")
    return str(error)


def batch_jsonl(
    requests: Iterable[tuple],
    model: str = "",
    endpoint: str = "/v1/chat/completions",
) -> str:
    """Build a batch input file (JSONL) from request tuples.

    Each request is ``(custom_id, messages)`` or ``(custom_id, messages, model)``.
    ``messages`` may be a prompt string (wrapped as a user message), a list of
    OpenAI message dicts, or a complete request-body dict. The optional third
    tuple element overrides the ``model`` keyword for that request.

    Returns the JSONL text (one JSON object per line, trailing newline when
    non-empty).
    """
    lines = []
    for request in requests:
        custom_id, payload, *rest = request
        request_model = rest[0] if rest else model
        if isinstance(payload, str):
            body: dict[str, Any] = {"messages": [{"role": "user", "content": payload}]}
        elif isinstance(payload, list):
            body = {"messages": payload}
        elif isinstance(payload, dict):
            body = dict(payload)
        else:
            raise TypeError(f"batch request {custom_id!r} must carry a prompt, messages list, or body dict")
        if request_model:
            body.setdefault("model", request_model)
        lines.append(
            json.dumps(
                {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": endpoint,
                    "body": body,
                }
            )
        )
    return "\n".join(lines) + ("\n" if lines else "")


class BatchGatewayClient:
    """Files and batches against a batch-gateway API server (OpenAI-compatible)."""

    def __init__(
        self,
        base_url: str,
        bearer_token: str = "",
        tenant: str = "default",
        verify_tls: bool = True,
        timeout: float = 120,
        transport: httpx.BaseTransport | None = None,
    ):
        # Kept so ``for_tenant`` can clone the client for a second tenant.
        self._base_url = base_url
        self._bearer_token = bearer_token
        self._tenant = tenant
        self._verify_tls = verify_tls
        self._timeout = timeout
        self._transport = transport
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers=self._headers(),
            verify=verify_tls,
            timeout=timeout,
            transport=transport,
        )

    def _headers(self, tenant: str | None = None) -> dict[str, str]:
        headers = {}
        if self._bearer_token:
            headers["Authorization"] = f"Bearer {self._bearer_token}"
        effective_tenant = self._tenant if tenant is None else tenant
        if effective_tenant:
            headers[TENANT_HEADER] = effective_tenant
        return headers

    def close(self) -> None:
        self._client.close()

    def for_tenant(self, tenant: str) -> BatchGatewayClient:
        """A second client against the same server acting as another tenant.

        Tenant identity is only the ``X-MaaS-Username`` header, so isolation
        checks need a client per tenant. The caller closes it.
        """
        return BatchGatewayClient(
            base_url=self._base_url,
            bearer_token=self._bearer_token,
            tenant=tenant,
            verify_tls=self._verify_tls,
            timeout=self._timeout,
            transport=self._transport,
        )

    def __enter__(self) -> BatchGatewayClient:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def create_file(self, filename: str, content: bytes) -> dict:
        """Upload a batch input file; the file object's ``purpose`` is ``batch``."""
        response = self._client.post(
            "/v1/files",
            files={"file": (filename, content, "application/octet-stream")},
            data={"purpose": "batch"},
        )
        response.raise_for_status()
        return response.json()

    def get_file(self, file_id: str) -> dict:
        response = self._client.get(f"/v1/files/{file_id}")
        response.raise_for_status()
        return response.json()

    def file_content(self, file_id: str) -> bytes:
        """Raw uploaded/result file bytes."""
        response = self._client.get(f"/v1/files/{file_id}/content")
        response.raise_for_status()
        return response.content

    def delete_file(self, file_id: str) -> dict:
        response = self._client.delete(f"/v1/files/{file_id}")
        response.raise_for_status()
        return response.json() if response.content else {}

    def list_files(self, limit: int = 0, after: str = "", purpose: str = "") -> dict:
        """``GET /v1/files``; omits unset query params.

        ``after`` is either an object id (``file-…``) or a non-negative integer
        offset — the server accepts both. Returns ``{"data": [...], "has_more": bool}``.
        """
        params = _page_params(limit, after)
        if purpose:
            params["purpose"] = purpose
        response = self._client.get("/v1/files", params=params)
        response.raise_for_status()
        return response.json()

    def create_batch(
        self,
        input_file_id: str,
        endpoint: str = "/v1/chat/completions",
        completion_window: str = "24h",
        metadata: dict | None = None,
    ) -> dict:
        body: dict[str, Any] = {
            "input_file_id": input_file_id,
            "endpoint": endpoint,
            "completion_window": completion_window,
        }
        if metadata is not None:
            body["metadata"] = metadata
        response = self._client.post("/v1/batches", json=body)
        response.raise_for_status()
        return response.json()

    def get_batch(self, batch_id: str) -> dict:
        response = self._client.get(f"/v1/batches/{batch_id}")
        response.raise_for_status()
        return response.json()

    def cancel_batch(self, batch_id: str) -> dict:
        response = self._client.post(f"/v1/batches/{batch_id}/cancel")
        response.raise_for_status()
        return response.json()

    def list_batches(self, limit: int = 0, after: str = "") -> dict:
        """``GET /v1/batches``; ``after`` accepts an id (``batch-…``) or an offset."""
        response = self._client.get("/v1/batches", params=_page_params(limit, after))
        response.raise_for_status()
        return response.json()

    def wait_for_terminal(
        self,
        batch_id: str,
        timeout: float | timedelta,
        interval: float | timedelta,
        on_poll: Callable[[dict], None] | None = None,
    ) -> dict:
        """Poll until the batch reaches a terminal status; return the batch object.

        ``on_poll`` receives every polled batch object (for phase logging).
        Raises ``TimeoutError`` when ``timeout`` elapses first.
        """
        timeout_seconds = _seconds(timeout)
        interval_seconds = _seconds(interval)
        deadline = time.monotonic() + timeout_seconds
        while True:
            batch = self.get_batch(batch_id)
            if on_poll:
                on_poll(batch)
            status = batch.get("status", "")
            if status in TERMINAL_BATCH_STATUSES:
                return batch
            if time.monotonic() >= deadline:
                counts = batch.get("request_counts") or {}
                raise TimeoutError(
                    f"batch {batch_id} did not reach a terminal status after {timeout_seconds:g}s "
                    f"(status={status or 'unknown'}, request_counts={counts})"
                )
            time.sleep(interval_seconds)

    def output_lines(self, batch: dict) -> list[dict]:
        """Parse the batch's ``output_file_id`` JSONL into a list of result objects."""
        file_id = batch.get("output_file_id")
        if not file_id:
            raise ValueError(f"batch {batch.get('id', '?')} has no output_file_id (status={batch.get('status', '?')})")
        content = self.file_content(file_id).decode("utf-8")
        return [json.loads(line) for line in content.splitlines() if line.strip()]


def batch_sql_transport_supported(kubectl_fn: Callable[..., str]) -> bool:
    """True when the installed CRD accepts ``spec.processor.asyncConfig.sql``.

    The async SQL transport landed after RHOAI 3.6-EA2: that operator's CRD only
    carries ``asyncConfig.redis``, so a manifest declaring ``asyncConfig.sql``
    is rejected at apply time with ``unknown field "spec.processor.asyncConfig.sql"``.
    Unreadable or unparseable CRDs report True — a wrong guess must not hide a
    deployment the platform may well support.
    """
    try:
        raw = kubectl_fn("get", "crd", BATCH_CRD, "-o", "json")
        versions = json.loads(raw)["spec"]["versions"]
        schema = versions[0]["schema"]["openAPIV3Schema"]
        async_config = schema["properties"]["spec"]["properties"]["processor"]["properties"]["asyncConfig"]
        return "sql" in (async_config.get("properties") or {})
    except Exception:
        return True


def batch_platform_state(kubectl_fn: Callable[..., str]) -> str:
    """Best-effort one-line summary of the platform's batch-gateway enablement.

    Checks, in order: the ``LLMBatchGateway`` CRD, the operator Deployment in the
    known operator namespaces, and the DataScienceCluster's
    ``aigateway.batchGateway.managementState``. Never raises — unanswerable parts
    are reported as such so the caller can show the diagnostic in a skip/failure.
    """
    parts: list[str] = []

    try:
        kubectl_fn("get", "crd", BATCH_CRD)
        parts.append(f"CRD {BATCH_CRD} present")
    except Exception as e:
        detail = str(e).splitlines()[0] if str(e) else "not found"
        parts.append(f"CRD {BATCH_CRD} not found ({detail})")

    operator_found = ""
    for namespace in BATCH_OPERATOR_NAMESPACES:
        try:
            kubectl_fn("get", "deployment", BATCH_OPERATOR_DEPLOYMENT, "-n", namespace)
            operator_found = f"{namespace}/{BATCH_OPERATOR_DEPLOYMENT}"
            break
        except Exception:
            continue
    if operator_found:
        parts.append(f"operator Deployment {operator_found} present")
    else:
        parts.append(
            f"operator Deployment {BATCH_OPERATOR_DEPLOYMENT} not found in " + "/".join(BATCH_OPERATOR_NAMESPACES)
        )

    try:
        state = kubectl_fn(
            "get",
            "datasciencecluster",
            "-o",
            "jsonpath={.items[0].spec.components.aigateway.batchGateway.managementState}",
        ).strip()
        parts.append(f"DataScienceCluster aigateway.batchGateway.managementState={state or 'unset'}")
    except Exception:
        parts.append("DataScienceCluster aigateway.batchGateway.managementState unavailable")

    return "; ".join(parts)


def batch_operator_running(kubectl_fn: Callable[..., str]) -> bool:
    """True when the batch-gateway operator pod is Running in a known operator namespace."""
    for namespace in BATCH_OPERATOR_NAMESPACES:
        try:
            phases = kubectl_fn(
                "get",
                "pods",
                "-n",
                namespace,
                "-l",
                BATCH_OPERATOR_LABEL,
                "-o",
                "jsonpath={.items[*].status.phase}",
            )
        except Exception:
            continue
        if any(phase.strip() == "Running" for phase in phases.split()):
            return True
    return False
