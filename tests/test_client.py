"""Unit tests for the LLMClient HTTP client.

Run with: ``uv run pytest tests/test_client.py -v``
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from conformance.client import LLMClient


def test_llm_client_init():
    c = LLMClient(base_url="http://localhost:8000", bearer_token="test-token")
    assert c._client.headers.get("authorization") == "Bearer test-token"
    c.close()


def test_chat_string_prompt_wraps_as_user_message(monkeypatch):
    """chat() with a plain string should wrap it as [{'role': 'user', 'content': ...}]."""
    captured = {}

    def fake_post(url, json=None, **kwargs):
        captured["json"] = json

        class FakeResp:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {"choices": [{"message": {"content": "ok"}}], "usage": {"total_tokens": 1}}

        return FakeResp()

    c = LLMClient(base_url="http://localhost:8000")
    monkeypatch.setattr(c._client, "post", fake_post)
    c.chat(model="test-model", prompt="hello")
    assert captured["json"]["messages"] == [{"role": "user", "content": "hello"}]
    c.close()


def test_chat_list_prompt_passes_through(monkeypatch):
    """chat() with a list of message dicts should pass them through unmodified."""
    captured = {}

    def fake_post(url, json=None, **kwargs):
        captured["json"] = json

        class FakeResp:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {"choices": [{"message": {"content": "ok"}}], "usage": {"total_tokens": 1}}

        return FakeResp()

    msgs = [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "hi"}]
    c = LLMClient(base_url="http://localhost:8000")
    monkeypatch.setattr(c._client, "post", fake_post)
    c.chat(model="test-model", prompt=msgs)
    assert captured["json"]["messages"] == msgs
    c.close()


def test_chat_with_tools_includes_tools_in_body(monkeypatch):
    """chat() with tools should include the tools list in the request body."""
    captured = {}

    def fake_post(url, json=None, **kwargs):
        captured["json"] = json

        class FakeResp:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "choices": [
                        {
                            "message": {"tool_calls": [{"function": {"name": "get_weather", "arguments": "{}"}}]},
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {"total_tokens": 10},
                }

        return FakeResp()

    tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get weather"}}]
    c = LLMClient(base_url="http://localhost:8000")
    monkeypatch.setattr(c._client, "post", fake_post)
    c.chat(model="test-model", prompt="What is the weather?", tools=tools)
    assert captured["json"]["tools"] == tools
    assert captured["json"]["messages"] == [{"role": "user", "content": "What is the weather?"}]
    c.close()


def test_chat_without_tools_omits_tools_key(monkeypatch):
    """chat() without tools should not include a 'tools' key in the request body."""
    captured = {}

    def fake_post(url, json=None, **kwargs):
        captured["json"] = json

        class FakeResp:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {"choices": [{"message": {"content": "ok"}}], "usage": {"total_tokens": 1}}

        return FakeResp()

    c = LLMClient(base_url="http://localhost:8000")
    monkeypatch.setattr(c._client, "post", fake_post)
    c.chat(model="test-model", prompt="hello")
    assert "tools" not in captured["json"]
    c.close()


def test_chat_response_preserves_attribution_headers(monkeypatch):
    import httpx

    response = httpx.Response(
        200,
        request=httpx.Request("POST", "http://grid/v1/chat/completions"),
        headers={"x-grid-site": "east", "x-grid-backend": "provider-a"},
        json={"choices": [{"message": {"content": "Hello"}}]},
    )
    client = LLMClient("http://grid")
    monkeypatch.setattr(client._client, "post", lambda *args, **kwargs: response)
    try:
        assert client.chat_response("model", "hello").headers["x-grid-backend"] == "provider-a"
        assert client.chat("model", "hello") == response.json()
    finally:
        client.close()
