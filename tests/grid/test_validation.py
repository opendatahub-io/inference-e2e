"""Offline tests for Grid attribution, response validation and testcase loading."""

from __future__ import annotations

import copy

import httpx
import pytest

from conformance.config import GridCheck, load_testcase
from conformance.grid import validate_grid_response


def completion():
    return {
        "object": "chat.completion",
        "model": "model-a",
        "choices": [{"message": {"role": "assistant", "content": "Hello"}}],
    }


def response(body=None, headers=None, status=200):
    return httpx.Response(
        status,
        json=completion() if body is None else body,
        headers={"X-Grid-Site": "east", "X-Grid-Backend": "provider-a"} if headers is None else headers,
    )


def check():
    return GridCheck(expected_routes={"east": ["provider-a"], "west": ["provider-b"]})


def test_accepts_case_insensitive_attribution():
    assert validate_grid_response(response(), "model-a", check()) == ("east", "provider-a")


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"x-grid-site": "east"},
        {"x-grid-backend": "provider-a"},
        {"x-grid-site": "east", "x-grid-backend": "provider-b"},
        {"x-grid-site": "untrusted", "x-grid-backend": "provider-a"},
    ],
)
def test_rejects_missing_or_mismatched_route_pairs(headers):
    with pytest.raises(AssertionError):
        validate_grid_response(response(headers=headers), "model-a", check())


@pytest.mark.parametrize("status", [401, 429, 500, 503])
def test_rejects_http_errors(status):
    with pytest.raises(AssertionError, match=f"HTTP {status}"):
        validate_grid_response(response(status=status), "model-a", check())


@pytest.mark.parametrize(
    "update",
    [
        {"model": "another-model"},
        {"object": "status"},
        {"choices": []},
        {"choices": "not-a-list"},
        {"choices": [None]},
        {"choices": [{"message": {"role": "user", "content": "Hello"}}]},
        {"choices": [{"message": {"role": "assistant", "content": ""}}]},
        {"choices": [{"message": {"role": "assistant", "content": None}}]},
    ],
)
def test_rejects_invalid_completions(update):
    body = copy.deepcopy(completion())
    body.update(update)
    with pytest.raises(AssertionError):
        validate_grid_response(response(body=body), "model-a", check())


def test_rejects_non_json_response():
    with pytest.raises(AssertionError, match="did not return JSON"):
        validate_grid_response(httpx.Response(200, text="gateway status page"), "model-a", check())


def test_rejects_json_array():
    with pytest.raises(AssertionError, match="JSON object"):
        validate_grid_response(response(body=[]), "model-a", check())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"request_count": 0},
        {"request_count": -1},
        {"request_count": True},
        {"max_tokens": 0},
        {"expected_routes": {}},
        {"expected_routes": {"east": []}},
        {"expected_routes": {"east": "provider-a"}},
        {"expected_routes": {"": ["provider-a"]}},
        {"expected_routes": {"east": [None]}},
    ],
)
def test_rejects_invalid_grid_configuration(kwargs):
    options = {"expected_routes": {"east": ["provider-a"]}}
    options.update(kwargs)
    with pytest.raises(ValueError):
        GridCheck(**options)


def test_loads_grid_case_without_a_deployment_manifest():
    testcase = load_testcase("configs/grid/testcases/grid-smoke.yaml")
    assert testcase.validation.grid.request_count == 10
    assert testcase.validation.grid.expected_routes == {"grid-local": ["llmd-a", "llmd-b"]}
    assert not testcase.deployment.manifest_path
    assert not testcase.cleanup
