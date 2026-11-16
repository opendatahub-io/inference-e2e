"""Response checks for an existing Grid consumer endpoint."""

from __future__ import annotations

import httpx

from conformance.config import GridCheck


def validate_grid_response(response: httpx.Response, model: str, check: GridCheck) -> tuple[str, str]:
    """Require a successful chat completion and an allowed site/backend pair."""
    assert response.status_code == 200, f"Grid inference returned HTTP {response.status_code}"
    try:
        body = response.json()
    except ValueError as error:
        raise AssertionError("Grid inference did not return JSON") from error
    assert isinstance(body, dict), "Grid inference must return a JSON object"
    assert body.get("model") == model, f"Expected model {model!r}, got {body.get('model')!r}"
    assert body.get("object") == "chat.completion", "Expected a chat.completion response"
    choices = body.get("choices")
    assert isinstance(choices, list) and choices, "Grid inference returned no completion choices"
    for choice in choices:
        assert isinstance(choice, dict), "Completion choice must be an object"
        message = choice.get("message")
        assert isinstance(message, dict) and message.get("role") == "assistant", "Expected an assistant message"
        assert isinstance(message.get("content"), str) and message["content"].strip(), "Expected assistant text"
    site = response.headers.get("x-grid-site", "")
    backend = response.headers.get("x-grid-backend", "")
    assert site, "Missing x-grid-site attribution header"
    assert backend, "Missing x-grid-backend attribution header"
    assert backend in check.expected_routes.get(site, []), f"Unexpected Grid route: site={site!r}, backend={backend!r}"
    return site, backend
