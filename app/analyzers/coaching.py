"""Concrete feedback and a rewritten answer, using her words."""

from ..models import QA
from .schemas import CoachNote, TagDraft

_ADVICE = {
    "rambling": "Cut the filler and keep one story with a beginning and an end.",
    "no_metrics": "Add one real number from that work: time, users, errors, or money.",
    "no_structure": "Order it as situation, what you did, and what changed.",
    "weak_closing": "End on the result, not on how the story felt.",
    "too_short": "Add the constraint, the action, and the outcome. One sentence is a dodge.",
    "off_topic": "Answer the question that was asked before adding extra context.",
    "unclear": "Split the long sentence. One idea, then the next.",
}

_PENALTY = {"low": 1, "med": 2, "high": 3}


def score_from_tags(tags: list[TagDraft]) -> int:
    penalty = sum(_PENALTY[tag.severity] for tag in tags)
    return max(0, min(10, 10 - penalty))


def heuristic_coach(qa: QA, tags: list[TagDraft]) -> CoachNote:
    if not tags:
        feedback = "This answer already has a shape. Keep the number and the ending."
    else:
        lines = [f"{tag.tag}: {_ADVICE[tag.tag]}" for tag in tags]
        feedback = " ".join(lines)

    body = " ".join((qa.her_answer or "").split())
    rewritten = (
        f"Situation: {body[:220]}. "
        "Action: I did the part I was responsible for, in order. "
        "Result: [put the real number here — latency, users, dollars, or time]."
    )
    return CoachNote(
        feedback=feedback,
        strong_version=rewritten,
        score=score_from_tags(tags),
    )


def format_feedback(note: CoachNote) -> str:
    """Pack both parts into the single feedback column on Drill."""
    return f"{note.feedback}\n\nStrong version:\n{note.strong_version}"


def split_feedback(stored: str) -> tuple[str, str]:
    marker = "\n\nStrong version:\n"
    if marker in stored:
        feedback, strong = stored.split(marker, 1)
        return feedback.strip(), strong.strip()
    return stored.strip(), ""
