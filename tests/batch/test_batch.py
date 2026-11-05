"""Unit tests for batch-gateway helpers (``src/conformance/batch.py``) — no cluster required.

Run with: ``uv run pytest tests/batch/test_batch.py -v``

``BatchGatewayClient`` is exercised against a mocked httpx transport so request
construction (paths, multipart/JSON bodies, tenant and auth headers), JSONL
output parsing, and terminal-status polling/timeout need no API server. The
platform diagnostics are driven from a fake ``kubectl`` so the CRD/operator/
DataScienceCluster probe can be checked without a cluster.
"""

from __future__ import annotations

import json
import sys
from datetime import timedelta
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

import conformance.batch as batch
from conformance.batch import (
    BATCH_CRD,
    BATCH_OPERATOR_DEPLOYMENT,
    BATCH_OPERATOR_LABEL,
    BatchGatewayClient,
    batch_jsonl,
    batch_operator_running,
    batch_platform_state,
    error_message,
)


def _client(handler, **kwargs) -> BatchGatewayClient:
    """Client wired to a mocked transport; ``handler(request) -> httpx.Response``."""
    return BatchGatewayClient("http://batch-gateway:8000", transport=httpx.MockTransport(handler), **kwargs)


class _FakeClock:
    """Deterministic replacement for the ``time`` module inside ``conformance.batch``."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


# ---------------------------------------------------------------------------
# batch_jsonl
# ---------------------------------------------------------------------------


def test_batch_jsonl_wraps_a_prompt_as_a_user_message():
    body = batch_jsonl([("req-1", "hello")], model="test-model")

    assert body.endswith("\n")
    assert json.loads(body) == {
        "custom_id": "req-1",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {"messages": [{"role": "user", "content": "hello"}], "model": "test-model"},
    }


def test_batch_jsonl_accepts_messages_and_body_dicts():
    messages = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]
    lines = batch_jsonl(
        [
            ("messages", messages),
            ("body", {"messages": messages, "temperature": 0}, "other-model"),
        ],
        model="test-model",
    ).splitlines()

    entries = [json.loads(line) for line in lines]
    assert entries[0]["body"] == {"messages": messages, "model": "test-model"}
    assert entries[1]["body"] == {"messages": messages, "temperature": 0, "model": "other-model"}


def test_batch_jsonl_keeps_a_model_already_present_in_the_body():
    body = json.loads(batch_jsonl([("req-1", {"messages": [], "model": "body-model"})], model="kwarg-model"))

    assert body["body"]["model"] == "body-model"


def test_batch_jsonl_honours_a_custom_endpoint():
    assert json.loads(batch_jsonl([("req-1", "hi")], endpoint="/v1/completions"))["url"] == "/v1/completions"


def test_batch_jsonl_is_empty_for_no_requests():
    assert batch_jsonl([], model="test-model") == ""


def test_batch_jsonl_rejects_an_unsupported_payload():
    with pytest.raises(TypeError, match="must carry"):
        batch_jsonl([("req-1", 42)])


# ---------------------------------------------------------------------------
# Files API request construction
# ---------------------------------------------------------------------------


def test_create_file_posts_a_multipart_batch_file():
    content = b'{"custom_id": "req-1"}\n'
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["content_type"] = request.headers["content-type"]
        seen["body"] = request.read()
        return httpx.Response(200, json={"id": "file-1", "purpose": "batch", "bytes": len(content)})

    with _client(handler) as client:
        uploaded = client.create_file("input.jsonl", content)

    assert uploaded == {"id": "file-1", "purpose": "batch", "bytes": len(content)}
    assert seen["method"] == "POST"
    assert seen["path"] == "/v1/files"
    assert seen["content_type"].startswith("multipart/form-data; boundary=")
    assert b'name="purpose"' in seen["body"]
    assert b'name="file"; filename="input.jsonl"' in seen["body"]
    assert content in seen["body"]


def test_file_operations_use_the_files_paths():
    calls: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.url.path == "/v1/files/file-1/content":
            return httpx.Response(200, content=b"line one\n")
        if request.method == "DELETE":
            return httpx.Response(200, json={"id": "file-1", "deleted": True})
        return httpx.Response(200, json={"id": "file-1", "purpose": "batch", "bytes": 9})

    with _client(handler) as client:
        assert client.get_file("file-1")["purpose"] == "batch"
        assert client.file_content("file-1") == b"line one\n"
        assert client.delete_file("file-1") == {"id": "file-1", "deleted": True}

    assert calls == [
        ("GET", "/v1/files/file-1"),
        ("GET", "/v1/files/file-1/content"),
        ("DELETE", "/v1/files/file-1"),
    ]


def test_delete_file_tolerates_an_empty_response_body():
    with _client(lambda request: httpx.Response(204)) as client:
        assert client.delete_file("file-1") == {}


def test_http_errors_are_raised():
    with _client(lambda request: httpx.Response(503, json={"error": "unavailable"})) as client:
        with pytest.raises(httpx.HTTPStatusError) as excinfo:
            client.get_file("file-1")

    assert excinfo.value.response.status_code == 503


# ---------------------------------------------------------------------------
# Batches API request construction
# ---------------------------------------------------------------------------


def test_create_batch_sends_the_openai_batch_body():
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.read()))
        return httpx.Response(200, json={"id": "batch-1", "status": "validating"})

    with _client(handler) as client:
        client.create_batch("file-1", endpoint="/v1/completions", completion_window="1h", metadata={"case": "smoke"})
        client.create_batch("file-2")

    assert bodies[0] == {
        "input_file_id": "file-1",
        "endpoint": "/v1/completions",
        "completion_window": "1h",
        "metadata": {"case": "smoke"},
    }
    assert bodies[1] == {
        "input_file_id": "file-2",
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h",
    }
    assert "metadata" not in bodies[1]


def test_get_and_cancel_batch_use_the_batches_paths():
    calls: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.method == "POST":
            return httpx.Response(200, json={"id": "batch-1", "status": "cancelling"})
        return httpx.Response(200, json={"id": "batch-1", "status": "in_progress"})

    with _client(handler) as client:
        assert client.get_batch("batch-1")["status"] == "in_progress"
        assert client.cancel_batch("batch-1")["status"] == "cancelling"

    assert calls == [
        ("GET", "/v1/batches/batch-1"),
        ("POST", "/v1/batches/batch-1/cancel"),
    ]


# ---------------------------------------------------------------------------
# Client configuration
# ---------------------------------------------------------------------------


def test_tenant_and_bearer_token_headers_are_sent():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["tenant"] = request.headers.get("x-maas-username")
        seen["authorization"] = request.headers.get("authorization")
        return httpx.Response(200, json={})

    with _client(handler, bearer_token="secret", tenant="team-a") as client:
        client.get_batch("batch-1")

    assert seen == {"tenant": "team-a", "authorization": "Bearer secret"}


def test_default_tenant_header_and_empty_tenant_omission():
    seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("x-maas-username"))
        return httpx.Response(200, json={})

    with _client(handler) as client:
        client.get_batch("batch-1")
    with _client(handler, tenant="") as client:
        client.get_batch("batch-1")

    assert seen == ["default", None]


def test_base_url_trailing_slash_is_normalised():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json={})

    transport = httpx.MockTransport(handler)
    with BatchGatewayClient("http://gateway:8000/", transport=transport) as client:
        client.get_batch("batch-1")

    assert seen["url"] == "http://gateway:8000/v1/batches/batch-1"


def test_for_tenant_clones_the_client_with_a_second_tenant_header():
    seen: list[tuple[str, str | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, request.headers.get("x-maas-username")))
        return httpx.Response(200, json={})

    with _client(handler, bearer_token="secret", tenant="team-a") as client:
        other = client.for_tenant("team-b")
        try:
            other.get_file("file-1")
        finally:
            other.close()
        client.get_file("file-1")

    assert seen == [("/v1/files/file-1", "team-b"), ("/v1/files/file-1", "team-a")]


# ---------------------------------------------------------------------------
# List request construction (upstream Files/List and Batches/List)
# ---------------------------------------------------------------------------


def test_list_files_sends_limit_after_and_purpose():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["params"] = dict(request.url.params)
        return httpx.Response(200, json={"object": "list", "data": [], "has_more": False})

    with _client(handler) as client:
        client.list_files(limit=2, after="file-2", purpose="batch")

    assert seen["path"] == "/v1/files"
    assert seen["params"] == {"limit": "2", "after": "file-2", "purpose": "batch"}


def test_list_calls_omit_unset_params():
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(dict(request.url.params))
        return httpx.Response(200, json={"data": [], "has_more": False})

    with _client(handler) as client:
        client.list_files()
        client.list_batches()
        client.list_files(limit=0, after="", purpose="")

    assert seen == [{}, {}, {}], "0/empty must fall back to the server default, not send limit=0"


def test_list_batches_sends_limit_and_after():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["params"] = dict(request.url.params)
        return httpx.Response(200, json={"data": [], "has_more": False})

    with _client(handler) as client:
        client.list_batches(limit=2, after="2")

    assert seen == {"path": "/v1/batches", "params": {"limit": "2", "after": "2"}}


# ---------------------------------------------------------------------------
# Error bodies
# ---------------------------------------------------------------------------


def test_error_message_reads_the_openai_error_object():
    response = httpx.Response(
        400,
        json={"error": {"message": "Input file 'file-x' does not exist", "type": "invalid_request_error"}},
        request=httpx.Request("POST", "http://batch-gateway:8000/v1/batches"),
    )

    assert error_message(response) == "Input file 'file-x' does not exist"


def test_error_message_tolerates_non_json_and_plain_strings():
    request = httpx.Request("POST", "http://batch-gateway:8000/v1/batches")

    assert error_message(httpx.Response(502, content=b"<html>bad gateway</html>", request=request)) == ""
    assert error_message(httpx.Response(400, json={"error": "plain"}, request=request)) == "plain"
    assert error_message(httpx.Response(400, json={}, request=request)) == ""


# ---------------------------------------------------------------------------
# Output JSONL parsing
# ---------------------------------------------------------------------------


def test_output_lines_parses_the_output_file_jsonl():
    results = [
        {"custom_id": "req-1", "response": {"status_code": 200}},
        {"custom_id": "req-2", "response": {"status_code": 200}},
    ]
    content = ("\n".join(json.dumps(result) for result in results) + "\n\n").encode()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/files/out-1/content"
        return httpx.Response(200, content=content)

    with _client(handler) as client:
        parsed = client.output_lines({"id": "batch-1", "status": "completed", "output_file_id": "out-1"})

    assert parsed == results


def test_output_lines_requires_an_output_file():
    with _client(lambda request: httpx.Response(200)) as client:
        with pytest.raises(ValueError, match="no output_file_id"):
            client.output_lines({"id": "batch-1", "status": "in_progress"})


# ---------------------------------------------------------------------------
# wait_for_terminal
# ---------------------------------------------------------------------------


def test_wait_for_terminal_polls_until_a_terminal_status(monkeypatch):
    clock = _FakeClock()
    monkeypatch.setattr(batch, "time", clock)
    statuses = iter(["validating", "in_progress", "completed"])
    polls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "batch-1", "status": next(statuses)})

    with _client(handler) as client:
        final = client.wait_for_terminal("batch-1", timeout=30, interval=5, on_poll=polls.append)

    assert final["status"] == "completed"
    assert [poll["status"] for poll in polls] == ["validating", "in_progress", "completed"]
    assert clock.sleeps == [5, 5]


@pytest.mark.parametrize("terminal_status", ["completed", "failed", "cancelled", "expired"])
def test_wait_for_terminal_returns_every_terminal_status(terminal_status):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "batch-1", "status": terminal_status})

    with _client(handler) as client:
        assert client.wait_for_terminal("batch-1", timeout=1, interval=0)["status"] == terminal_status


def test_wait_for_terminal_times_out_with_the_last_status_and_counts(monkeypatch):
    clock = _FakeClock()
    monkeypatch.setattr(batch, "time", clock)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"id": "batch-1", "status": "in_progress", "request_counts": {"total": 2, "completed": 0}},
        )

    with _client(handler) as client:
        with pytest.raises(
            TimeoutError,
            match=r"after 10s \(status=in_progress, request_counts=\{'total': 2, 'completed': 0\}\)",
        ):
            client.wait_for_terminal("batch-1", timeout=timedelta(seconds=10), interval=timedelta(seconds=5))

    assert clock.sleeps == [5, 5]


# ---------------------------------------------------------------------------
# Platform diagnostics
# ---------------------------------------------------------------------------


def test_platform_state_reports_crd_operator_and_dsc():
    calls: list[tuple] = []

    def kubectl(*args: str) -> str:
        calls.append(args)
        if args[1] == "crd":
            return BATCH_CRD
        if args[1] == "deployment":
            return f"deployment.apps/{BATCH_OPERATOR_DEPLOYMENT}"
        if args[1] == "datasciencecluster":
            return "Managed"
        raise AssertionError(f"unexpected kubectl call: {args}")

    state = batch_platform_state(kubectl)

    assert f"CRD {BATCH_CRD} present" in state
    assert f"operator Deployment redhat-ods-applications/{BATCH_OPERATOR_DEPLOYMENT} present" in state
    assert "managementState=Managed" in state
    assert ("get", "crd", BATCH_CRD) in calls


def test_platform_state_never_raises_when_kubectl_fails():
    def kubectl(*args: str) -> str:
        raise RuntimeError("Error from server (NotFound)\nmore detail on the next line")

    state = batch_platform_state(kubectl)

    assert f"CRD {BATCH_CRD} not found (Error from server (NotFound))" in state
    assert f"operator Deployment {BATCH_OPERATOR_DEPLOYMENT} not found in redhat-ods-applications/opendatahub" in state
    assert "managementState unavailable" in state


def test_platform_state_falls_back_to_the_opendatahub_namespace():
    def kubectl(*args: str) -> str:
        if args[1] == "crd":
            return BATCH_CRD
        if args[1] == "deployment":
            namespace = args[args.index("-n") + 1]
            if namespace == "opendatahub":
                return f"deployment.apps/{BATCH_OPERATOR_DEPLOYMENT}"
            raise RuntimeError("Error from server (NotFound)")
        return ""

    state = batch_platform_state(kubectl)

    assert f"operator Deployment opendatahub/{BATCH_OPERATOR_DEPLOYMENT} present" in state
    assert "managementState=unset" in state


def test_operator_running_detects_a_running_pod():
    calls: list[tuple] = []

    def kubectl(*args: str) -> str:
        calls.append(args)
        return "Pending Running"

    assert batch_operator_running(kubectl) is True
    assert calls[0][0:2] == ("get", "pods")
    assert "-l" in calls[0]
    assert BATCH_OPERATOR_LABEL in calls[0]


def test_operator_running_is_false_without_a_running_pod():
    assert batch_operator_running(lambda *args: "Pending") is False
    assert batch_operator_running(lambda *args: "") is False


def test_operator_running_falls_back_to_the_second_namespace():
    def kubectl(*args: str) -> str:
        namespace = args[args.index("-n") + 1]
        if namespace == "redhat-ods-applications":
            raise RuntimeError("pods is forbidden")
        return "Running"

    assert batch_operator_running(kubectl) is True


def test_operator_running_is_false_when_kubectl_fails():
    def kubectl(*args: str) -> str:
        raise RuntimeError("connection refused")

    assert batch_operator_running(kubectl) is False
