"""Endpoint-only Grid smoke checks; no Kubernetes deployment or cleanup fixtures."""

from __future__ import annotations

from collections import Counter

import pytest

from conformance.client import LLMClient
from conformance.config import TestCase
from conformance.grid import validate_grid_response


@pytest.mark.applies_when.with_args(lambda tc: tc.validation.grid is not None)
class TestGrid:
    @pytest.fixture(scope="class")
    def grid_client(self, request, tc: TestCase):
        endpoint = request.config.getoption("--endpoint")
        if request.config.getoption("--mode") != "discover" or not endpoint:
            raise pytest.UsageError("Grid checks require --mode discover and an explicit --endpoint")
        if not tc.model.name:
            raise pytest.UsageError("Grid checks require a model name in the testcase or --model")
        client = LLMClient(
            base_url=endpoint,
            bearer_token=request.config.getoption("--bearer-token"),
            timeout=tc.validation.timeout.total_seconds(),
        )
        yield client
        client.close()

    def test_40_grid_inference_attribution(self, grid_client: LLMClient, tc: TestCase):
        """Every response must be a valid completion from an allowed site/backend pair."""
        check = tc.validation.grid
        prompts = tc.validation.test_prompts or ["Hello"]
        routes = Counter()
        for index in range(check.request_count):
            response = grid_client.chat_response(
                tc.model.name, prompts[index % len(prompts)], max_tokens=check.max_tokens
            )
            routes[validate_grid_response(response, tc.model.name, check)] += 1
        print(f"Grid attributed responses: {dict(routes)}")
