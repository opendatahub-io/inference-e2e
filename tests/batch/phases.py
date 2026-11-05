"""Batch-gateway conformance phases.

``BatchPhases`` is mixed into ``TestConformance`` (tests/test_conformance.py) so
these phases run for each testcase after the llm-d phases and the MaaS phases,
and before cleanup. They are deselected unless the testcase sets
``validation.batchGateway.enabled``.

The batch gateway is a *platform* feature: the ``llmbatchgateways.batch.llm-d.ai``
CRD and its operator come from the platform, so this suite never installs or
deletes them (same rule as every other CRD here). ``--batch-gateway auto``
skips these phases with a diagnostic when the platform lacks that support;
``--batch-gateway require`` fails with the same diagnostic instead. The same
probe gates the phase 02 deploy (``test_conformance._require_batch_platform``),
so the manifest's ``LLMBatchGateway`` document is never applied on a platform
that cannot serve it.

Phases mirror the upstream ``llm-d-batch-gateway`` e2e suite
(``test/e2e/*.go``) rather than inventing new behaviour, so a failure here means
the platform diverged from what upstream already guarantees:

  40a  LLMBatchGateway CR and its component conditions are ready
  40b  files API round-trips a JSONL file and deletes it
  40c  a JSONL batch built from the testcase prompts completes with usable results
  40d  files contract: duplicate filename, purpose filter, pagination
  40e  batch creation rejects missing input_file_id, bad endpoint, unknown file
  40f  ingestion rejects duplicate custom_ids and streaming requests
  40g  partial results: mixed success/failure counts, output and error files
  40h  input file reuse: no trailing newline, one file feeding two batches
  40i  cancel: before processing, idempotent retry, terminal batch rejected
  40j  multi-tenant isolation across files and batches
  40k  several batches in flight concurrently

Not ported (needs infrastructure this suite does not manage): dispatcher/EPP
dispatch and flow control, AIMD shedding, hard-kill/orphan recovery, graceful
shutdown, resumable multi-worker, helm upgrade, GC expiry, pprof and OTel
traces. Those run in the operator repo's ``make dev-deploy && make test-e2e``.

To add a check, add a ``test_40<letter>_batch_<name>`` method here.

Batch metrics validation is deliberately not covered yet: when the gateway
exports Prometheus metrics, add a ``test_40l_batch_metrics`` here next to the
existing ``Scraper`` validators (``src/conformance/metrics.py``).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

from conformance.batch import (
    BATCH_CRD,
    BatchGatewayClient,
    batch_jsonl,
    batch_operator_running,
    batch_platform_state,
    batch_sql_transport_supported,
    error_message,
)
from conformance.config import TestCase
from conformance.deployer import Deployer

# The operator names each operand service after its CR, so the CR is named after
# the testcase: its API server Service is ``<testcase-name>-apiserver`` (port 8000).
BATCH_GATEWAY_KIND = "llmbatchgateway"
# Endpoint every batch request is dispatched to; the API server proxies to the model gateway.
BATCH_ENDPOINT = "/v1/chat/completions"

# Conditions the operator reports on an LLMBatchGateway; async adds the processor one.
BATCH_READY_CONDITION = "Ready"
BATCH_READY_CONDITIONS = ("APIServerAvailable", "ProcessorAvailable", "GCAvailable")
BATCH_ASYNC_CONDITION = "AsyncProcessorAvailable"

BATCH_POLL_INTERVAL_SECONDS = 15
# An identical Ready=False reason+message across this many polls is stuck, not progressing.
_STUCK_POLL_THRESHOLD = 3

_MANIFEST_DIR = Path(__file__).resolve().parents[2] / "deploy" / "manifests"

# (status, reason, message) of one ``status.conditions`` entry.
Condition = tuple[str, str, str]


def _log(msg: str) -> None:
    print(f"  → {msg}")


def _batch_enabled(tc: TestCase) -> bool:
    return tc.validation.batch_gateway.enabled


# Applied per method: a class-level mark would be inherited by every TestConformance phase.
# with_args: a mark called with a lone callable would treat it as the decorated function.
_for_batch_gateways = pytest.mark.applies_when.with_args(_batch_enabled)


def _batch_enabled_with_concurrency(tc: TestCase) -> bool:
    """40k is opt-out: ``concurrency: 0`` deselects the phase instead of skipping it."""
    return _batch_enabled(tc) and tc.validation.batch_gateway.concurrency > 0


_for_batch_gateways_concurrent = pytest.mark.applies_when.with_args(_batch_enabled_with_concurrency)


def batch_gateway_name(tc: TestCase) -> str:
    """Name of the testcase's ``LLMBatchGateway`` CR — the operator derives its Services from it."""
    return tc.name


def parse_conditions(raw: str) -> dict[str, Condition]:
    """Map ``status.conditions`` type → (status, reason, message) from a CR's JSON dump."""
    try:
        resource = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        return {}
    entries = (resource.get("status") or {}).get("conditions") or []
    return {
        entry["type"]: (entry.get("status", ""), entry.get("reason", ""), entry.get("message", ""))
        for entry in entries
        if isinstance(entry, dict) and entry.get("type")
    }


def batch_conditions(deployer: Deployer, name: str) -> dict[str, Condition]:
    """Current conditions of ``LLMBatchGateway/<name>`` (empty when the CR is absent)."""
    raw = deployer.kubectl("get", BATCH_GATEWAY_KIND, name, "-n", deployer.namespace, "-o", "json", check=False)
    return parse_conditions(raw)


def _status(conditions: dict[str, Condition], condition_type: str) -> str:
    return (conditions.get(condition_type) or ("",))[0]


def _format_condition(condition_type: str, conditions: dict[str, Condition]) -> str:
    status, reason, message = conditions.get(condition_type) or ("Missing", "not reported yet", "")
    detail = f"{condition_type}={status or 'Unknown'}"
    if reason:
        detail += f" ({reason}"
        detail += f": {message})" if message else ")"
    return detail


def wait_for_batch_ready(
    deployer: Deployer,
    name: str,
    timeout: float,
    require_async: bool = False,
    print_fn=None,
    interval: float = BATCH_POLL_INTERVAL_SECONDS,
) -> dict[str, Condition]:
    """Poll ``LLMBatchGateway/<name>`` until its component conditions are True.

    Mirrors ``Deployer.wait_for_ready``: a failure reason+message that repeats
    across three consecutive polls raises immediately instead of burning the
    whole timeout, and a plain timeout reports the last observed conditions.
    Returns the conditions so the caller can log them.
    """
    required = (BATCH_READY_CONDITION,) + BATCH_READY_CONDITIONS + ((BATCH_ASYNC_CONDITION,) if require_async else ())
    deadline = time.time() + timeout
    start = time.time()
    conditions: dict[str, Condition] = {}
    stuck_key, stuck_count = "", 0

    while time.time() < deadline:
        conditions = batch_conditions(deployer, name)
        pending = [condition for condition in required if _status(conditions, condition) != "True"]
        elapsed = int(time.time() - start)
        if print_fn and pending:
            print_fn(
                f"[{elapsed}s/{int(timeout)}s] waiting for " + ", ".join(_describe_conditions(pending, conditions))
            )
        elif print_fn:
            print_fn(f"[{elapsed}s/{int(timeout)}s] all batch gateway conditions True")
        if not pending:
            return conditions

        failing = [condition for condition in required if _status(conditions, condition) == "False"]
        stuck = "|".join(f"{condition}:{conditions[condition][1]}:{conditions[condition][2]}" for condition in failing)
        if stuck and stuck == stuck_key:
            stuck_count += 1
            if stuck_count >= _STUCK_POLL_THRESHOLD:
                raise RuntimeError(
                    f"LLMBatchGateway/{name} is stuck: " + "; ".join(_describe_conditions(failing, conditions))
                )
        else:
            stuck_key, stuck_count = stuck, (1 if stuck else 0)
        time.sleep(interval)

    last = "; ".join(_describe_conditions(required, conditions)) or "no conditions reported"
    raise TimeoutError(f"LLMBatchGateway/{name} not ready after {timeout:g}s ({last})")


def _describe_conditions(condition_types: tuple[str, ...] | list[str], conditions: dict[str, Condition]) -> list[str]:
    return [_format_condition(condition, conditions) for condition in condition_types]


def async_configured(deployer: Deployer, name: str) -> bool:
    """True when the CR sets ``spec.processor.asyncConfig`` (async dispatch, not sync passthrough).

    An unreadable CR reports False: 40a fails on the missing CR before the extra
    condition matters, and requiring it blindly would turn a parse problem into a
    misleading readiness failure.
    """
    raw = deployer.kubectl("get", BATCH_GATEWAY_KIND, name, "-n", deployer.namespace, "-o", "json", check=False)
    try:
        spec = (json.loads(raw) if raw.strip() else {}).get("spec") or {}
    except json.JSONDecodeError:
        return False
    return bool((spec.get("processor") or {}).get("asyncConfig"))


def require_batch_platform(deployer: Deployer, mode: str, transport: str = "") -> None:
    """Skip (``auto``) or fail (``require``) when the platform does not provide the batch gateway.

    Both the CRD and a running operator must be present: the CRD alone cannot
    render the API server the phases talk to. ``transport`` names the async queue
    transport the manifest declares (``sql`` for the shipped test case); an
    operator whose CRD lacks it cannot apply that manifest at all.
    """
    crd_present = deployer.check_crd_exists(BATCH_CRD)
    operator_running = batch_operator_running(deployer.kubectl)
    if crd_present and operator_running:
        if transport == "sql" and not batch_sql_transport_supported(deployer.kubectl):
            message = (
                "batch gateway operator does not support the SQL async transport "
                f"(no spec.processor.asyncConfig.sql in CRD {BATCH_CRD}) — needs an operator that ships it "
                "(RHOAI 3.6 GA or ODH nightly); set validation.batchGateway.dispatchUrl only for routing, "
                "it cannot add a transport"
            )
            if mode == "require":
                pytest.fail(message, pytrace=False)
            pytest.skip(message)
        return
    state = batch_platform_state(deployer.kubectl)
    message = (
        "batch gateway platform not available "
        f"(CRD {BATCH_CRD}={'present' if crd_present else 'missing'}, "
        f"operator running={operator_running}) — {state}"
    )
    if mode == "require":
        pytest.fail(message, pytrace=False)
    pytest.skip(message)


def require_manifest(tc: TestCase) -> None:
    """Skip when the manifest repo branch has no manifest for this testcase yet."""
    if not (_MANIFEST_DIR / tc.deployment.manifest_path).exists():
        pytest.skip(f"manifest not found for this branch: {tc.deployment.manifest_path}")


def require_batch_target(deployer: Deployer, tc: TestCase, test_mode: str, mode: str) -> None:
    """Every runtime gate a batch phase needs: manifest present, deployed, platform capable.

    Shared with the ``batch_client`` fixture, which runs before the phase bodies
    and must not start a port-forward for a testcase that was never deployed.
    """
    require_manifest(tc)
    if test_mode != "discover" and not deployer.is_deployed(tc.name):
        pytest.skip(f"skipped — deploy failed or was skipped for '{tc.name}'")
    require_batch_platform(deployer, mode, tc.validation.batch_gateway.transport)


def assert_http_error(status: int, call, what: str) -> None:
    """Assert ``call`` fails with ``status`` and an OpenAI-shaped ``error.message``."""
    try:
        call()
    except httpx.HTTPStatusError as e:
        detail = error_message(e.response)
        assert e.response.status_code == status, (
            f"{what}: expected HTTP {status}, got {e.response.status_code} ({detail or 'no error body'})"
        )
        assert detail, f"{what}: HTTP {status} response carries no error.message"
        return
    raise AssertionError(f"{what}: expected HTTP {status}, request succeeded")


def submit_batch(client: BatchGatewayClient, tc: TestCase, content: bytes, filename: str) -> tuple[dict, str]:
    """Upload ``content`` and submit a batch; return ``(batch, input_file_id)``.

    The caller owns the input file and must delete it only once every batch that
    references it is terminal — the processor reads it while the batch runs.
    """
    cfg = tc.validation.batch_gateway
    uploaded = client.create_file(filename, content)
    file_id = uploaded.get("id", "")
    if not file_id:
        raise AssertionError(f"POST /v1/files returned no id: {uploaded}")
    batch = client.create_batch(
        file_id,
        endpoint=BATCH_ENDPOINT,
        completion_window=cfg.completion_window,
        metadata={"testcase": tc.name},
    )
    if not batch.get("id"):
        raise AssertionError(f"POST /v1/batches returned no id: {batch}")
    _log(f"Submitted batch {batch['id']} from {filename} ({len(content)} bytes)")
    return batch, file_id


def run_batch(client: BatchGatewayClient, tc: TestCase, content: bytes, filename: str) -> dict:
    """Upload, submit, wait for a terminal status, then delete the input file."""
    cfg = tc.validation.batch_gateway
    batch, file_id = submit_batch(client, tc, content, filename)
    try:
        return client.wait_for_terminal(
            batch["id"],
            timeout=cfg.request_timeout,
            interval=cfg.poll_interval,
            on_poll=lambda polled: _log(
                f"batch {batch['id']}: status={polled.get('status', 'unknown')} "
                f"counts={polled.get('request_counts') or {}}"
            ),
        )
    except TimeoutError:
        _cancel_quietly(client, batch["id"])
        raise
    finally:
        _delete_quietly(client, file_id)


def wait_for_ingestion_failure(
    client: BatchGatewayClient,
    batch_id: str,
    timeout: float,
    interval: float = BATCH_POLL_INTERVAL_SECONDS,
) -> dict:
    """Poll until the batch reaches ``failed``; any other terminal status is a failure.

    Ported from the upstream e2e helper of the same name: duplicate custom_ids
    and streaming requests must fail ingestion instead of completing with
    per-request errors.
    """
    from conformance.batch import TERMINAL_BATCH_STATUSES

    deadline = time.monotonic() + timeout
    while True:
        batch = client.get_batch(batch_id)
        status = batch.get("status", "")
        if status == "failed":
            return batch
        if status in TERMINAL_BATCH_STATUSES:
            raise AssertionError(
                f"batch {batch_id} reached terminal status {status!r} instead of 'failed' "
                f"(request_counts={batch.get('request_counts') or {}})"
            )
        if time.monotonic() >= deadline:
            raise TimeoutError(f"batch {batch_id} never failed within {timeout:g}s (status={status or 'unknown'})")
        time.sleep(interval)


class BatchPhases:
    """Batch-gateway checks for testcases with ``validation.batchGateway.enabled``."""

    @_for_batch_gateways
    def test_40a_batch_ready(self, deployer: Deployer, tc: TestCase, test_mode: str, batch_gateway_mode: str):
        """LLMBatchGateway exists and reports Ready plus every component condition."""
        require_batch_target(deployer, tc, test_mode, batch_gateway_mode)
        name = batch_gateway_name(tc)
        if not deployer.check_resource_exists(BATCH_GATEWAY_KIND, name):
            pytest.fail(
                f"{BATCH_GATEWAY_KIND}/{name} not found in '{deployer.namespace}' — the testcase manifest "
                f"({tc.deployment.manifest_path}) must declare it, and its name must match the testcase name",
                pytrace=False,
            )
        is_async = async_configured(deployer, name)
        _log(f"Waiting for {BATCH_GATEWAY_KIND}/{name} conditions (async={is_async})")
        conditions = wait_for_batch_ready(
            deployer,
            name,
            tc.deployment.ready_timeout.total_seconds(),
            require_async=is_async,
            print_fn=_log,
        )
        required = (BATCH_READY_CONDITION,) + BATCH_READY_CONDITIONS
        required += (BATCH_ASYNC_CONDITION,) if is_async else ()
        for condition in required:
            _log(_format_condition(condition, conditions))
        assert _status(conditions, BATCH_READY_CONDITION) == "True", (
            f"{BATCH_GATEWAY_KIND}/{name} Ready={_status(conditions, BATCH_READY_CONDITION)}"
        )
        _log(f"{BATCH_GATEWAY_KIND}/{name} is Ready")

    @_for_batch_gateways
    def test_40b_batch_files(
        self,
        deployer: Deployer,
        tc: TestCase,
        test_mode: str,
        batch_gateway_mode: str,
        batch_client: BatchGatewayClient,
    ):
        """Files API: upload a JSONL file, read metadata and content back, delete it, leave no residue."""
        require_batch_target(deployer, tc, test_mode, batch_gateway_mode)
        cfg = tc.validation.batch_gateway
        content = batch_jsonl([("llm-d-e2e-probe", "ping")], model=cfg.model or tc.model.name).encode()
        _log(f"Uploading a {len(content)}-byte probe file")
        uploaded = batch_client.create_file("llm-d-e2e-batch-probe.jsonl", content)
        file_id = uploaded.get("id", "")
        assert file_id, f"POST /v1/files returned no file id: {uploaded}"
        assert uploaded.get("purpose") == "batch", f"uploaded file is not a batch file: {uploaded}"
        assert uploaded.get("bytes") == len(content), f"uploaded byte count mismatch: {uploaded}"
        try:
            fetched = batch_client.get_file(file_id)
            assert fetched.get("id") == file_id, fetched
            assert fetched.get("purpose") == "batch", f"GET /v1/files/{{id}} lost purpose: {fetched}"
            assert fetched.get("bytes") == len(content), f"GET /v1/files/{{id}} lost byte count: {fetched}"
            assert batch_client.file_content(file_id) == content, "file content round-trip mismatch"
            _log(f"File {file_id}: purpose=batch, {len(content)} bytes, content round-trips")
        finally:
            deleted = batch_client.delete_file(file_id)
        _log(f"Deleted file {file_id}")
        if "deleted" in deleted:
            assert deleted["deleted"] is True, f"DELETE /v1/files/{{id}} did not report deletion: {deleted}"
        with pytest.raises(httpx.HTTPStatusError) as deleted_error:
            batch_client.get_file(file_id)
        assert deleted_error.value.response.status_code == 404, (
            f"a deleted file must be gone, GET returned {deleted_error.value.response.status_code}"
        )

    @_for_batch_gateways
    def test_40c_batch_inference(
        self,
        deployer: Deployer,
        tc: TestCase,
        test_mode: str,
        batch_gateway_mode: str,
        batch_client: BatchGatewayClient,
    ):
        """A JSONL batch over the testcase prompts completes with one usable result per request."""
        require_batch_target(deployer, tc, test_mode, batch_gateway_mode)
        cfg = tc.validation.batch_gateway
        prompts = cfg.prompts or tc.validation.test_prompts
        assert prompts, "batchGateway.prompts and validation.testPrompts are both empty"
        expected = cfg.expect_outputs or len(prompts)
        model = cfg.model or tc.model.name
        requests = [(f"{tc.name}-{index}", prompt) for index, prompt in enumerate(prompts)]
        content = batch_jsonl(requests, model=model).encode()

        input_file = batch_client.create_file(f"{tc.name}-input.jsonl", content)
        input_file_id = input_file["id"]
        try:
            batch = batch_client.create_batch(
                input_file_id,
                endpoint=BATCH_ENDPOINT,
                completion_window=cfg.completion_window,
                metadata={"testcase": tc.name},
            )
            batch_id = batch.get("id", "")
            assert batch_id, f"POST /v1/batches returned no batch id: {batch}"
            _log(f"Created batch {batch_id} with {len(requests)} request(s) (model={model})")
            try:
                final = batch_client.wait_for_terminal(
                    batch_id,
                    timeout=cfg.request_timeout,
                    interval=cfg.poll_interval,
                    on_poll=lambda polled: _log(
                        f"batch {batch_id}: status={polled.get('status', 'unknown')} "
                        f"counts={polled.get('request_counts') or {}}"
                    ),
                )
            except TimeoutError:
                _cancel_quietly(batch_client, batch_id)
                raise
        finally:
            _delete_quietly(batch_client, input_file_id)

        assert final.get("status") == "completed", (
            f"batch {batch_id} ended in status={final.get('status')!r} "
            f"(request_counts={final.get('request_counts') or {}}, errors={final.get('errors')})"
        )
        counts = final.get("request_counts") or {}
        assert counts.get("total") == expected, f"expected {expected} dispatched request(s): {counts}"
        assert counts.get("completed") == expected, f"expected {expected} completed request(s): {counts}"

        output_file_id = final.get("output_file_id", "")
        assert output_file_id, f"a completed batch must carry output_file_id: {final}"
        if final.get("error_file_id"):
            _log(f"WARNING: batch also wrote error_file_id={final['error_file_id']}")

        results = batch_client.output_lines(final)
        assert len(results) == expected, f"output file has {len(results)} line(s), expected {expected}"
        expected_ids = {custom_id for custom_id, _ in requests}
        assert {result.get("custom_id") for result in results} == expected_ids, "output custom_ids do not match"
        for result in results:
            response = result.get("response") or {}
            assert response.get("status_code") == 200, f"request {result.get('custom_id')} failed: {result}"
            choices = (response.get("body") or {}).get("choices") or []
            assert choices, f"request {result.get('custom_id')} returned no choices: {result}"
            text = ((choices[0].get("message") or {}).get("content") or "").strip()
            assert text, f"request {result.get('custom_id')} returned empty assistant content: {result}"

        _log(f"Output file {output_file_id}: {len(results)} result line(s)")
        _log(f"Sample response: {((results[0]['response']['body']['choices'][0]['message']['content'])[:80])!r}")

    @_for_batch_gateways
    def test_40d_batch_files_contract(
        self,
        deployer: Deployer,
        tc: TestCase,
        test_mode: str,
        batch_gateway_mode: str,
        batch_client: BatchGatewayClient,
    ):
        """Files contract: duplicate filenames, purpose filter, cursor and offset pagination."""
        require_batch_target(deployer, tc, test_mode, batch_gateway_mode)
        cfg = tc.validation.batch_gateway
        # A dedicated tenant keeps the pagination counts ours alone (upstream does the same).
        client = batch_client.for_tenant(f"{tc.name}-files-{uuid4().hex[:8]}")
        try:
            content = batch_jsonl([("files-contract", "ping")], model=cfg.model or tc.model.name).encode()
            filename = f"{tc.name}-duplicate.jsonl"
            first = client.create_file(filename, content)
            second = client.create_file(filename, content)
            assert first["id"] != second["id"], f"two uploads of {filename} reused one id: {first['id']}"
            for uploaded in (first, second):
                fetched = client.get_file(uploaded["id"])
                assert fetched.get("filename") == filename, f"filename lost on upload: {fetched}"
                assert fetched.get("purpose") == "batch", f"purpose lost on upload: {fetched}"
            third = client.create_file(f"{tc.name}-third.jsonl", content)
            ids = [first["id"], second["id"], third["id"]]
            _log(f"Uploaded {len(ids)} files, two sharing the name {filename}")
            try:
                listed = client.list_files(purpose="batch", limit=100)
                assert {entry["id"] for entry in listed["data"]} == set(ids), listed
                page1 = client.list_files(purpose="batch", limit=2)
                assert len(page1["data"]) == 2, f"limit=2 returned {len(page1['data'])} files: {page1}"
                assert page1["has_more"] is True, f"page 1 of 3 must report has_more: {page1}"
                page2 = client.list_files(purpose="batch", limit=2, after=page1["data"][-1]["id"])
                assert len(page2["data"]) == 1, f"page 2 must hold the remaining file: {page2}"
                assert page2["has_more"] is False, f"page 2 must not report has_more: {page2}"
                # Ordering is server-defined (this build lists newest first), so assert
                # coverage and the absence of duplicates rather than creation order.
                paged = [entry["id"] for entry in page1["data"] + page2["data"]]
                assert len(paged) == len(set(paged)), f"pagination repeated a file: {paged}"
                assert set(paged) == set(ids), f"pagination lost a file: {paged} vs {ids}"
                # `after` also accepts a non-negative offset, which upstream exercises.
                offset_page = client.list_files(purpose="batch", limit=2, after="2")
                offset_ids = [entry["id"] for entry in offset_page["data"]]
                assert len(offset_ids) == 1, f"after=2 (offset) must return the last file: {offset_page}"
                assert offset_ids[0] in set(ids) - set(paged[:2]), (
                    f"offset page must return the file page 1 did not: {offset_ids}"
                )
                _log("pagination: limit/after by id and by offset both page correctly")
            finally:
                for file_id in ids:
                    _delete_quietly(client, file_id)
        finally:
            client.close()

    @_for_batch_gateways
    def test_40e_batch_validation(
        self,
        deployer: Deployer,
        tc: TestCase,
        test_mode: str,
        batch_gateway_mode: str,
        batch_client: BatchGatewayClient,
    ):
        """Batch creation rejects a missing input_file_id, a bad endpoint, and an unknown file."""
        require_batch_target(deployer, tc, test_mode, batch_gateway_mode)
        cfg = tc.validation.batch_gateway
        assert_http_error(
            400,
            lambda: batch_client.create_batch("", endpoint=BATCH_ENDPOINT, completion_window=cfg.completion_window),
            "missing input_file_id",
        )
        content = batch_jsonl([("validation", "ping")], model=cfg.model or tc.model.name).encode()
        uploaded = batch_client.create_file(f"{tc.name}-validation.jsonl", content)
        try:
            assert_http_error(
                400,
                lambda: batch_client.create_batch(uploaded["id"], endpoint="/v1/invalid"),
                "unsupported endpoint",
            )
        finally:
            _delete_quietly(batch_client, uploaded["id"])
        assert_http_error(
            400,
            lambda: batch_client.create_batch("file-does-not-exist-12345", endpoint=BATCH_ENDPOINT),
            "unknown input file",
        )
        _log("all three invalid batch creations rejected with 400 + error.message")

    @_for_batch_gateways
    def test_40f_batch_ingestion_rejections(
        self,
        deployer: Deployer,
        tc: TestCase,
        test_mode: str,
        batch_gateway_mode: str,
        batch_client: BatchGatewayClient,
    ):
        """Ingestion rejects duplicate custom_ids and streaming requests: the batch must fail."""
        require_batch_target(deployer, tc, test_mode, batch_gateway_mode)
        cfg = tc.validation.batch_gateway
        model = cfg.model or tc.model.name
        duplicate = batch_jsonl([("dup-1", "Hello"), ("dup-1", "World")], model=model).encode()
        streaming = batch_jsonl(
            [("stream-1", {"model": model, "stream": True, "messages": [{"role": "user", "content": "Hi"}]})]
        ).encode()
        for label, content, name in (
            ("duplicate custom_id", duplicate, f"{tc.name}-dup-custom-id.jsonl"),
            ("streaming request", streaming, f"{tc.name}-streaming.jsonl"),
        ):
            batch, file_id = submit_batch(batch_client, tc, content, name)
            try:
                final = wait_for_ingestion_failure(
                    batch_client,
                    batch["id"],
                    timeout=cfg.request_timeout.total_seconds(),
                    interval=cfg.poll_interval.total_seconds(),
                )
            finally:
                _delete_quietly(batch_client, file_id)
            _log(f"{label}: batch {final['id']} failed ingestion as expected")

    @_for_batch_gateways
    def test_40g_batch_partial_results(
        self,
        deployer: Deployer,
        tc: TestCase,
        test_mode: str,
        batch_gateway_mode: str,
        batch_client: BatchGatewayClient,
    ):
        """Partial results: mixed success/failure counts, output and error files, model_not_found."""
        require_batch_target(deployer, tc, test_mode, batch_gateway_mode)
        cfg = tc.validation.batch_gateway
        model = cfg.model or tc.model.name
        mixed = batch_jsonl(
            [
                ("good-1", "Hello"),
                ("bad-1", {"model": "nonexistent-model-xyz", "messages": [{"role": "user", "content": "Hello"}]}),
                ("good-2", "World"),
            ],
            model=model,
        ).encode()
        final = run_batch(batch_client, tc, mixed, f"{tc.name}-mixed.jsonl")
        assert final.get("status") == "completed", f"mixed batch ended {final.get('status')!r}: {final}"
        counts = final.get("request_counts") or {}
        assert (counts.get("total"), counts.get("completed"), counts.get("failed")) == (3, 2, 1), (
            f"mixed batch counts must be total=3 completed=2 failed=1: {counts}"
        )
        assert final.get("output_file_id"), f"two successes must write an output file: {final}"
        assert final.get("error_file_id"), f"one failure must write an error file: {final}"
        errors = batch_client.output_lines({"output_file_id": final["error_file_id"]})
        assert [entry.get("custom_id") for entry in errors] == ["bad-1"], f"error file carried {errors}"
        _log(f"mixed batch {final['id']}: 2 completed, 1 failed, error file lists bad-1")

        unregistered = batch_jsonl(
            [
                (f"bad-{index}", {"model": "nonexistent-model-xyz", "messages": [{"role": "user", "content": "Hi"}]})
                for index in (1, 2)
            ],
            model=model,
        ).encode()
        all_bad = run_batch(batch_client, tc, unregistered, f"{tc.name}-unregistered.jsonl")
        assert all_bad.get("status") == "completed", f"unregistered-model batch ended {all_bad.get('status')!r}"
        bad_counts = all_bad.get("request_counts") or {}
        assert (bad_counts.get("completed"), bad_counts.get("failed")) == (0, 2), f"counts: {bad_counts}"
        error_file_id = all_bad.get("error_file_id", "")
        assert error_file_id, f"an all-failed batch must carry an error file: {all_bad}"
        body = batch_client.file_content(error_file_id).decode("utf-8", errors="replace")
        assert "model_not_found" in body, f"error file must report model_not_found, got: {body[:400]}"
        _log("unregistered model: failed=2 and the error file reports model_not_found")

    @_for_batch_gateways
    def test_40h_batch_file_reuse(
        self,
        deployer: Deployer,
        tc: TestCase,
        test_mode: str,
        batch_gateway_mode: str,
        batch_client: BatchGatewayClient,
    ):
        """Input files are reusable: no trailing newline, and one file feeding two batches."""
        require_batch_target(deployer, tc, test_mode, batch_gateway_mode)
        cfg = tc.validation.batch_gateway
        model = cfg.model or tc.model.name
        content = batch_jsonl([("reuse-1", "Hello"), ("reuse-2", "World")], model=model).encode()
        no_newline = content.rstrip(b"\n")
        final = run_batch(batch_client, tc, no_newline, f"{tc.name}-no-trailing-newline.jsonl")
        assert final.get("status") == "completed", f"input without a trailing newline ended {final.get('status')!r}"
        counts = final.get("request_counts") or {}
        assert (counts.get("total"), counts.get("completed"), counts.get("failed")) == (2, 2, 0), f"counts: {counts}"
        assert final.get("output_file_id") and not final.get("error_file_id"), f"unexpected files: {final}"
        _log("input without a trailing newline ingested 2/2")

        batch, file_id = submit_batch(batch_client, tc, content, f"{tc.name}-shared-input.jsonl")
        second, _ = submit_batch(batch_client, tc, content, f"{tc.name}-shared-input-2.jsonl")
        try:
            finals = [
                batch_client.wait_for_terminal(submitted["id"], timeout=cfg.request_timeout, interval=cfg.poll_interval)
                for submitted in (batch, second)
            ]
        finally:
            _delete_quietly(batch_client, file_id)
        for submitted in finals:
            assert submitted.get("status") == "completed", (
                f"shared-input batch {submitted['id']} ended {submitted.get('status')!r}"
            )
            assert (submitted.get("request_counts") or {}).get("completed") == 2, submitted
        assert finals[0]["output_file_id"] != finals[1]["output_file_id"], "two batches must not share an output file"
        _log(f"one input file fed two batches: {finals[0]['id']} and {finals[1]['id']}")

    @_for_batch_gateways
    def test_40i_batch_cancel(
        self,
        deployer: Deployer,
        tc: TestCase,
        test_mode: str,
        batch_gateway_mode: str,
        batch_client: BatchGatewayClient,
    ):
        """Cancel is idempotent before processing and rejected once the batch is terminal."""
        require_batch_target(deployer, tc, test_mode, batch_gateway_mode)
        cfg = tc.validation.batch_gateway
        model = cfg.model or tc.model.name
        slow = batch_jsonl(
            [
                (
                    f"cancel-{index}",
                    {
                        "model": model,
                        "max_tokens": 200,
                        "messages": [{"role": "user", "content": f"slow {index}"}],
                    },
                )
                for index in range(1, max(cfg.cancel_requests, 1) + 1)
            ],
            model=model,
        ).encode()
        batch, file_id = submit_batch(batch_client, tc, slow, f"{tc.name}-cancel.jsonl")
        try:
            first = batch_client.cancel_batch(batch["id"])
            assert first.get("status") in {"cancelling", "cancelled"}, (
                f"cancel returned {first.get('status')!r}, expected cancelling or cancelled"
            )
            retry = batch_client.cancel_batch(batch["id"])
            assert retry.get("status") in {"cancelling", "cancelled"}, (
                f"idempotent cancel retry returned {retry.get('status')!r}"
            )
            final = batch_client.wait_for_terminal(batch["id"], timeout=cfg.request_timeout, interval=cfg.poll_interval)
            assert final.get("status") == "cancelled", f"batch settled at {final.get('status')!r}"
            assert (final.get("request_counts") or {}).get("completed", 0) == 0, (
                f"a batch cancelled at submit must complete nothing: {final.get('request_counts')}"
            )
            _log(f"batch {final['id']} cancelled (idempotent retry accepted)")
        finally:
            _delete_quietly(batch_client, file_id)

        done = run_batch(
            batch_client,
            tc,
            batch_jsonl([("done-1", "Hello")], model=model).encode(),
            f"{tc.name}-cancel-terminal.jsonl",
        )
        assert done.get("status") == "completed", f"terminal-cancel batch ended {done.get('status')!r}"
        assert_http_error(400, lambda: batch_client.cancel_batch(done["id"]), "cancel of a completed batch")
        _log("cancelling a completed batch is rejected with 400")

    @_for_batch_gateways
    def test_40j_batch_multi_tenant(
        self,
        deployer: Deployer,
        tc: TestCase,
        test_mode: str,
        batch_gateway_mode: str,
        batch_client: BatchGatewayClient,
    ):
        """Tenant isolation: another tenant cannot read or reuse this tenant's files and batches."""
        require_batch_target(deployer, tc, test_mode, batch_gateway_mode)
        cfg = tc.validation.batch_gateway
        model = cfg.model or tc.model.name
        suffix = uuid4().hex[:8]
        other = batch_client.for_tenant(f"{cfg.tenant or 'default'}-other-{suffix}")
        try:
            content = batch_jsonl([("isolation", "ping")], model=model).encode()
            file_id = batch_client.create_file(f"{tc.name}-tenant-a-{suffix}.jsonl", content)["id"]
            try:
                assert_http_error(404, lambda: other.get_file(file_id), "other tenant reading a file")
                listed = other.list_files(limit=100, purpose="batch")
                assert file_id not in {entry["id"] for entry in listed["data"]}, "other tenant lists our file"
                assert_http_error(
                    400,
                    lambda: other.create_batch(file_id, endpoint=BATCH_ENDPOINT),
                    "other tenant submitting a batch over our file",
                )
                batch = batch_client.create_batch(
                    file_id, endpoint=BATCH_ENDPOINT, completion_window=cfg.completion_window
                )
                try:
                    assert_http_error(404, lambda: other.get_batch(batch["id"]), "other tenant reading a batch")
                    batches = other.list_batches(limit=100)
                    assert batch["id"] not in {entry["id"] for entry in batches["data"]}, "other tenant lists our batch"
                    _log(f"tenant isolation holds for file {file_id} and batch {batch['id']}")
                finally:
                    _cancel_quietly(batch_client, batch["id"])
            finally:
                _delete_quietly(batch_client, file_id)
        finally:
            other.close()

    @_for_batch_gateways_concurrent
    def test_40k_batch_concurrency(
        self,
        deployer: Deployer,
        tc: TestCase,
        test_mode: str,
        batch_gateway_mode: str,
        batch_client: BatchGatewayClient,
    ):
        """Several batches run concurrently and each completes independently."""
        require_batch_target(deployer, tc, test_mode, batch_gateway_mode)
        cfg = tc.validation.batch_gateway
        model = cfg.model or tc.model.name
        count = cfg.concurrency
        submitted: list[dict] = []
        file_ids: list[str] = []
        try:
            for index in range(count):
                content = batch_jsonl(
                    [(f"concurrent-{index}-1", "Hello"), (f"concurrent-{index}-2", "World")], model=model
                ).encode()
                batch, file_id = submit_batch(batch_client, tc, content, f"{tc.name}-concurrent-{index}.jsonl")
                submitted.append(batch)
                file_ids.append(file_id)
            deadline = time.monotonic() + cfg.request_timeout.total_seconds()
            finals = []
            for batch in submitted:
                remaining = max(deadline - time.monotonic(), 1.0)
                finals.append(
                    batch_client.wait_for_terminal(batch["id"], timeout=remaining, interval=cfg.poll_interval)
                )
        finally:
            for file_id in file_ids:
                _delete_quietly(batch_client, file_id)
        for final in finals:
            assert final.get("status") == "completed", f"concurrent batch {final['id']} ended {final.get('status')!r}"
            counts = final.get("request_counts") or {}
            assert (counts.get("completed"), counts.get("failed")) == (2, 0), f"{final['id']}: {counts}"
        outputs = {final["output_file_id"] for final in finals}
        assert len(outputs) == count, f"{count} batches must produce {count} output files, saw {len(outputs)}"
        _log(f"{count} concurrent batches each completed 2/2 with distinct output files")


def _cancel_quietly(client: BatchGatewayClient, batch_id: str) -> None:
    """Best effort: do not leave a runaway batch behind when the phase times out."""
    try:
        client.cancel_batch(batch_id)
        _log(f"Cancelled batch {batch_id} after the timeout")
    except Exception as e:  # noqa: BLE001 - cancellation is cleanup, never the failure being reported
        _log(f"WARNING: could not cancel batch {batch_id}: {e}")


def _delete_quietly(client: BatchGatewayClient, file_id: str) -> None:
    """Best effort: cleanup must not mask the assertion that failed."""
    try:
        client.delete_file(file_id)
    except Exception as e:  # noqa: BLE001 - deletion is cleanup, never the failure being reported
        _log(f"WARNING: could not delete file {file_id}: {e}")
