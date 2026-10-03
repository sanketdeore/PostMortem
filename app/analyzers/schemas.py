"""Shared shapes for every analyzer. LLM output is validated against these."""

from typing import Literal

from pydantic import BaseModel, Field

TAG = Literal[
    "rambling",
    "no_metrics",
    "no_structure",
    "weak_closing",
    "too_short",
    "off_topic",
    "unclear",
]
SEVERITY = Literal["low", "med", "high"]


class TagDraft(BaseModel):
    tag: TAG
    severity: SEVERITY
    evidence_quote: str = ""


class TagList(BaseModel):
    tags: list[TagDraft] = Field(default_factory=list)


class QuestionList(BaseModel):
    questions: list[str]


class CoachNote(BaseModel):
    feedback: str
    strong_version: str
    score: int = Field(ge=0, le=10)


def snap_quote(answer: str, quote: str) -> str:
    """Keep evidence as a real span of her answer, not a paraphrase."""
    quote = (quote or "").strip().strip('"')
    if quote and quote in answer:
        return quote[:240]
    if quote:
        idx = answer.lower().find(quote.lower())
        if idx >= 0:
            return answer[idx : idx + len(quote)][:240]
    compact = " ".join(answer.split())
    return compact[:180]
