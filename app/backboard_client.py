"""One HTTP helper for every Backboard call. Open-weight models only."""

import time

import httpx

from .config import settings
from .open_models import assert_open_provider, resolve_open_model


def post_message(
    content: str,
    *,
    model: str | None = None,
    system_prompt: str = "",
    thread_id: str | None = None,
    assistant_id: str | None = None,
    memory: str = "off",
    json_output: bool = False,
    timeout: float = 120.0,
) -> tuple[dict, float]:
    """POST /threads/messages. Returns (json body, latency milliseconds).

    A missing thread_id starts a throwaway thread. Pass her thread_id only
    when the call should join the memory she keeps across drills.
    """
    if not settings.backboard_api_key:
        raise RuntimeError("BACKBOARD_API_KEY is not set")

    provider = assert_open_provider(settings.backboard_llm_provider)
    body: dict = {
        "content": content,
        "llm_provider": provider,
        "model_name": resolve_open_model(model or settings.backboard_model),
        "stream": False,
        "memory": memory,
        "json_output": json_output,
    }
    if system_prompt:
        body["system_prompt"] = system_prompt
    if thread_id:
        body["thread_id"] = thread_id
    if assistant_id:
        body["assistant_id"] = assistant_id

    started = time.monotonic()
    response = httpx.post(
        f"{settings.backboard_base_url.rstrip('/')}/threads/messages",
        headers={"X-API-Key": settings.backboard_api_key},
        json=body,
        timeout=timeout,
    )
    latency_ms = (time.monotonic() - started) * 1000
    response.raise_for_status()
    return response.json(), latency_ms


def token_usage(payload: dict) -> tuple[int, int]:
    usage = payload.get("usage") or {}
    incoming = (
        payload.get("input_tokens")
        or usage.get("prompt_tokens")
        or usage.get("input_tokens")
        or 0
    )
    outgoing = (
        payload.get("output_tokens")
        or usage.get("completion_tokens")
        or usage.get("output_tokens")
        or 0
    )
    return int(incoming or 0), int(outgoing or 0)
