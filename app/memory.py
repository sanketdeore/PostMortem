"""One Backboard assistant and one thread for her drill history.

Drill events are appended to the thread. The dashboard reads that thread
back so the pattern view can show what Backboard remembers.
"""

import json

import httpx

from .backboard_client import post_message
from .config import settings
from .preferences import get_prefs, set_pref

_ASSISTANT = "backboard_assistant_id"
_THREAD = "backboard_thread_id"
_SUMMARY = "backboard_summary"

_SYSTEM = (
    "You are PostMortem's memory for one job hunter. "
    "She practices interview answers. After each drill, update a two-sentence "
    "summary of the weakness pattern you have actually seen. "
    "Do not invent companies, scores, or tags she did not log. "
    'Reply with JSON only: {"summary":"..."}'
)


def _headers() -> dict[str, str]:
    return {"X-API-Key": settings.backboard_api_key}


def _base() -> str:
    return settings.backboard_base_url.rstrip("/")


def get_or_create_assistant() -> str | None:
    """One assistant for this install. Id is stored in Preference."""
    existing = get_prefs().get(_ASSISTANT) or ""
    if existing:
        return existing
    if not settings.backboard_api_key:
        return None
    try:
        response = httpx.post(
            f"{_base()}/assistants",
            headers=_headers(),
            json={"name": "PostMortem", "system_prompt": _SYSTEM},
            timeout=30.0,
        )
        response.raise_for_status()
        assistant_id = response.json()["assistant_id"]
        set_pref(_ASSISTANT, assistant_id)
        return assistant_id
    except (httpx.HTTPError, KeyError, TypeError):
        return None


def get_or_create_thread(assistant_id: str) -> str | None:
    existing = get_prefs().get(_THREAD) or ""
    if existing:
        return existing
    try:
        response = httpx.post(
            f"{_base()}/assistants/{assistant_id}/threads",
            headers=_headers(),
            json={},
            timeout=30.0,
        )
        response.raise_for_status()
        thread_id = response.json()["thread_id"]
        set_pref(_THREAD, thread_id)
        return thread_id
    except (httpx.HTTPError, KeyError, TypeError):
        return None


def log_drill(question: str, score: int, tags: list[str]) -> None:
    """Append one drill to her thread. Failures are ignored so practice still saves."""
    assistant_id = get_or_create_assistant()
    if not assistant_id:
        return
    thread_id = get_or_create_thread(assistant_id)
    if not thread_id:
        return
    tag_text = ", ".join(tags) if tags else "none"
    content = (
        "Drill logged.\n"
        f"Question: {question}\n"
        f"Score: {score}/10\n"
        f"Top tags: {tag_text}\n"
        "Update the running summary."
    )
    try:
        payload, _latency = post_message(
            content,
            system_prompt=_SYSTEM,
            thread_id=thread_id,
            assistant_id=assistant_id,
            memory="Auto",
            json_output=True,
            timeout=45.0,
        )
        summary = _summary_from_payload(payload)
        if summary:
            set_pref(_SUMMARY, summary)
    except (httpx.HTTPError, RuntimeError, KeyError, TypeError, ValueError):
        return


def read_thread_summary() -> str:
    """Read the thread on dashboard load. Cached text is the fallback."""
    cached = get_prefs().get(_SUMMARY) or ""
    thread_id = get_prefs().get(_THREAD) or ""
    if not thread_id or not settings.backboard_api_key:
        return cached
    try:
        response = httpx.get(
            f"{_base()}/threads/{thread_id}",
            headers=_headers(),
            timeout=15.0,
        )
        response.raise_for_status()
        summary = _summary_from_payload(response.json()) or cached
        if summary and summary != cached:
            set_pref(_SUMMARY, summary)
        return summary
    except (httpx.HTTPError, KeyError, TypeError, ValueError):
        return cached


def _summary_from_payload(payload: dict) -> str:
    if isinstance(payload.get("summary"), str) and payload["summary"].strip():
        return payload["summary"].strip()
    content = payload.get("content")
    if isinstance(content, str) and content.strip():
        return _unwrap_summary(content)
    messages = payload.get("messages") or []
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        role = (message.get("role") or message.get("sender") or "").lower()
        text = message.get("content") or ""
        if role in {"assistant", "model", ""} and isinstance(text, str) and text.strip():
            return _unwrap_summary(text)
    return ""


def _unwrap_summary(text: str) -> str:
    from .analyzers.prompts import extract_json

    try:
        blob = json.loads(extract_json(text))
        if isinstance(blob, dict) and isinstance(blob.get("summary"), str):
            return blob["summary"].strip()
    except (ValueError, TypeError, json.JSONDecodeError):
        pass
    return " ".join(text.split())[:500]
