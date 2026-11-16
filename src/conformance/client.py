"""HTTP client for vLLM endpoints (httpx).

``LLMClient`` talks to ``/health``, ``/v1/models``, ``/v1/completions``,
``/v1/chat/completions``, ``/v1/messages`` (Anthropic), and ``/v1/responses``
(OpenAI Responses API). Optional bearer token; TLS verify disabled for
self-signed pod certs.

In conformance tests, two instances are used:
  - Gateway base URL (``client`` fixture) — inference only; EPP routes these
  - Direct pod base URL (``pod_client`` fixture) — health and model list

``chat()`` accepts a plain string (wrapped as a user message) or a list of
OpenAI message dicts (e.g. from ``chat_prompt_to_messages`` / LoRA adapter
model names).
"""

from __future__ import annotations

import httpx


class LLMClient:
    """Health, model list, completions, and chat against an OpenAI-compatible base URL."""

    def __init__(self, base_url: str, bearer_token: str = "", timeout: float = 120):
        headers = {}
        if bearer_token:
            headers["Authorization"] = f"Bearer {bearer_token}"
        self._client = httpx.Client(
            base_url=base_url,
            headers=headers,
            verify=False,
            timeout=timeout,
        )

    def close(self):
        self._client.close()

    def health_check(self) -> bool:
        r = self._client.get("/health")
        r.raise_for_status()
        return True

    def list_models(self) -> dict:
        r = self._client.get("/v1/models")
        r.raise_for_status()
        return r.json()

    def completions(self, model: str, prompt: str, max_tokens: int = 64, temperature: float = 0.1) -> dict:
        r = self._client.post(
            "/v1/completions",
            json={
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        r.raise_for_status()
        return r.json()

    def chat(
        self,
        model: str,
        prompt: str | list[dict],
        max_tokens: int = 64,
        temperature: float = 0.1,
        tools: list[dict] | None = None,
    ) -> dict:
        return self.chat_response(model, prompt, max_tokens, temperature, tools).json()

    def chat_response(
        self,
        model: str,
        prompt: str | list[dict],
        max_tokens: int = 64,
        temperature: float = 0.1,
        tools: list[dict] | None = None,
    ) -> httpx.Response:
        """Chat response including headers, for routing attribution checks."""
        messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
        body: dict = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            body["tools"] = tools
        r = self._client.post("/v1/chat/completions", json=body)
        r.raise_for_status()
        return r

    def messages(self, model: str, prompt: str | list[dict], max_tokens: int = 64, temperature: float = 0.1) -> dict:
        """Anthropic /v1/messages endpoint."""
        msgs = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
        r = self._client.post(
            "/v1/messages",
            headers={"anthropic-version": "2023-06-01"},
            json={
                "model": model,
                "messages": msgs,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        r.raise_for_status()
        return r.json()

    def responses(self, model: str, prompt: str, max_output_tokens: int = 64, temperature: float = 0.1) -> dict:
        """OpenAI /v1/responses endpoint."""
        r = self._client.post(
            "/v1/responses",
            json={
                "model": model,
                "input": prompt,
                "max_output_tokens": max_output_tokens,
                "temperature": temperature,
            },
        )
        r.raise_for_status()
        return r.json()
