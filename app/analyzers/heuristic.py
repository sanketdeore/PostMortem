"""Length, number, filler, and structure checks. No model required."""

import re

from ..models import QA
from .coaching import heuristic_coach
from .questions import heuristic_questions
from .schemas import TAG, CoachNote, TagDraft

_FILLER = re.compile(
    r"\b(um+|uh+|like|basically|actually|literally|sort of|kind of|you know|i mean)\b",
    re.IGNORECASE,
)
_WORD = re.compile(r"[a-z0-9']+")
_STRUCTURE = (
    "first",
    "then",
    "because",
    "so",
    "situation",
    "task",
    "action",
    "result",
    "specifically",
    "for example",
)
_CLOSING = (
    "result",
    "learned",
    "impact",
    "outcome",
    "improved",
    "reduced",
    "increased",
    "shipped",
    "delivered",
    "percent",
)
_STOP = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with",
    "about", "that", "this", "it", "is", "was", "were", "be", "you", "your",
    "me", "my", "we", "our", "i", "how", "what", "when", "why", "tell", "time",
}

_SEVERITY_RANK = {"high": 0, "med": 1, "low": 2}


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _quote_around(answer: str, needle: str) -> str:
    idx = answer.lower().find(needle.lower())
    if idx < 0:
        return " ".join(answer.split())[:160]
    start = max(0, idx - 20)
    end = min(len(answer), idx + len(needle) + 40)
    return answer[start:end].strip()


class HeuristicAnalyzer:
    """Rule analyzer. Also the fallback when a model call fails."""

    name = "heuristic"

    def __init__(self) -> None:
        self.used_fallback = False

    def analyze(self, qa: QA, preferences: dict[str, str]) -> list[TagDraft]:
        self.used_fallback = False
        answer = qa.her_answer or ""
        words = _words(answer)
        count = len(words)
        tags: list[TagDraft] = []
        behavioral = any(
            phrase in (qa.question_text or "").lower()
            for phrase in ("tell me", "time you", "describe a")
        )
        # A direct technical answer can be short. A story that is this short is a dodge.
        short_limit = 35 if behavioral else 18
        if count < short_limit:
            tags.append(
                TagDraft(
                    tag="too_short",
                    severity="high" if count < 18 else "med",
                    evidence_quote=" ".join(answer.split())[:160],
                )
            )

        has_number = bool(re.search(r"\d", answer)) or any(
            token in answer.lower() for token in ("percent", "million", "thousand")
        )
        if not has_number and count >= 12:
            tags.append(
                TagDraft(
                    tag="no_metrics",
                    severity="high",
                    evidence_quote=" ".join(answer.split())[:160],
                )
            )

        filler_hits = list(_FILLER.finditer(answer))
        density = len(filler_hits) / max(count, 1)
        lower = answer.lower()
        structure_hits = 0
        for marker in _STRUCTURE:
            if " " in marker:
                if marker in lower:
                    structure_hits += 1
            elif re.search(rf"\b{marker}\b", lower):
                structure_hits += 1
        if density >= 0.06 or (count > 180 and structure_hits == 0):
            evidence = (
                _quote_around(answer, filler_hits[0].group(0))
                if filler_hits
                else " ".join(answer.split())[:160]
            )
            tags.append(
                TagDraft(
                    tag="rambling",
                    severity="high" if density >= 0.1 or count > 220 else "med",
                    evidence_quote=evidence,
                )
            )

        if behavioral and count >= 25 and structure_hits == 0:
            tags.append(
                TagDraft(
                    tag="no_structure",
                    severity="med",
                    evidence_quote=" ".join(answer.split())[:160],
                )
            )

        tail = " ".join(words[-30:])
        if behavioral and count >= 25 and not any(word in tail for word in _CLOSING):
            tags.append(
                TagDraft(
                    tag="weak_closing",
                    severity="low" if count < 80 else "med",
                    evidence_quote=" ".join(answer.split())[-160:],
                )
            )

        q_words = {w for w in _words(qa.question_text) if w not in _STOP and len(w) > 3}
        # Only when none of the question's content words appear. A precise
        # technical answer often doesn't repeat the prompt, so a low ratio
        # false-flags too often.
        if count >= 20 and q_words and not (q_words & set(words)):
            tags.append(
                TagDraft(
                    tag="off_topic",
                    severity="med",
                    evidence_quote=" ".join(answer.split())[:160],
                )
            )

        sentences = [s for s in re.split(r"[.!?]+", answer) if s.strip()]
        long_sentence = any(len(_words(s)) > 45 for s in sentences)
        diversity = (len(set(words)) / count) if count else 1.0
        if long_sentence or (count > 40 and diversity < 0.35):
            tags.append(
                TagDraft(
                    tag="unclear",
                    severity="med",
                    evidence_quote=(max(sentences, key=lambda s: len(s)) if sentences else answer)[:180],
                )
            )

        # One row per tag, strongest severity first, cap so the page stays readable.
        best: dict[TAG, TagDraft] = {}
        for tag in tags:
            current = best.get(tag.tag)
            if current is None or _SEVERITY_RANK[tag.severity] < _SEVERITY_RANK[current.severity]:
                best[tag.tag] = tag
        ordered = sorted(best.values(), key=lambda t: _SEVERITY_RANK[t.severity])
        return ordered[:5]

    def generate_questions(self, role: str, focus_tags: list[str], n: int = 5) -> list[str]:
        self.used_fallback = False
        return heuristic_questions(role, focus_tags, n)

    def coach(self, qa: QA, tags: list[TagDraft], preferences: dict[str, str]) -> CoachNote:
        self.used_fallback = False
        return heuristic_coach(qa, tags)
