"""The strict-JSON prompt shared by Ollama, Backboard, Tinker, and the training set."""

import json

from ..models import QA

ALLOWED = (
    "rambling, no_metrics, no_structure, weak_closing, too_short, off_topic, unclear"
)


def system_prompt(preferences: dict[str, str]) -> str:
    role = preferences.get("target_role") or "software engineer"
    domain = preferences.get("domain") or "unspecified"
    weak = preferences.get("weak_areas") or "unspecified"
    return (
        "You are PostMortem, an interview post-mortem coach for one person.\n"
        f"Her target role: {role}. Domain: {domain}. "
        f"She says she is weak at: {weak}.\n"
        "Read the interview question and her spoken answer. "
        "Tag only weaknesses that are actually present.\n"
        f"Allowed tags: {ALLOWED}.\n"
        "Severity is low, med, or high.\n"
        "evidence_quote must be a short verbatim span copied from her answer.\n"
        "Return strict JSON only, no markdown:\n"
        '{"tags":[{"tag":"...","severity":"...","evidence_quote":"..."}]}\n'
        'If the answer is strong, return {"tags":[]}.'
    )


def user_prompt(qa: QA) -> str:
    duration = f"\nDuration seconds: {qa.duration_sec}" if qa.duration_sec else ""
    return f"Question: {qa.question_text}\nAnswer: {qa.her_answer}{duration}"


def question_prompt(role: str, focus_tags: list[str], n: int = 5) -> str:
    focus = ", ".join(focus_tags) if focus_tags else "none"
    return (
        f"Write {n} practice interview questions for a {role} candidate.\n"
        "Vary the formats: behavioral, technical, situational, "
        '"tell me about a time", and one curveball.\n'
        f"Bias at least two questions toward these weaknesses: {focus}.\n"
        'Return strict JSON only: {"questions":["..."]}'
    )


def coach_prompt(qa: QA, tag_names: list[str]) -> str:
    tags = ", ".join(tag_names) if tag_names else "none"
    return (
        "Coach her on this single answer. Use her words. Do not invent employers, "
        "metrics, or projects she did not say.\n"
        f"Weakness tags already found: {tags}.\n"
        f"Question: {qa.question_text}\n"
        f"Her answer: {qa.her_answer}\n"
        "Return strict JSON only:\n"
        '{"feedback":"2-4 concrete sentences","strong_version":'
        '"her answer rewritten with structure and a place for a real number",'
        '"score":0}'
    )


def extract_json(raw: str) -> str:
    """Pull the first JSON object out of fences or a chatty preamble."""
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object in model response")
    return text[start : end + 1]


def dumps(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)
