"""Turn a messy post-interview dump into question/answer pairs.

People don't write clean transcripts after a rejection. This accepts
"Q:" / "Question:" / "A:" / "Answer:" with colons, dots, or dashes,
markdown bullets, and answers that run across several lines.
"""

import re
from dataclasses import dataclass

_MARKER = re.compile(
    r"^\s*(?:[-*]\s*)?(?P<kind>q|question|a|answer)\s*[:.)\-]\s*(?P<rest>.*)$",
    re.IGNORECASE,
)
_DURATION = re.compile(
    r"\((?P<secs>\d+)\s*(?:s|sec|secs|seconds)\)\s*$",
    re.IGNORECASE,
)


@dataclass
class ParsedQA:
    question_text: str
    her_answer: str
    duration_sec: int | None = None


def _clean(lines: list[str]) -> str:
    return "\n".join(line.rstrip() for line in lines).strip()


def parse_debrief(text: str) -> list[ParsedQA]:
    """Return every Q/A block. An empty list means the dump had no markers."""
    pairs: list[ParsedQA] = []
    question: list[str] = []
    answer: list[str] = []
    mode: str | None = None

    def flush() -> None:
        q = _clean(question)
        a = _clean(answer)
        if not q or not a:
            return
        duration = None
        match = _DURATION.search(a)
        if match:
            duration = int(match.group("secs"))
            a = _DURATION.sub("", a).strip()
        pairs.append(ParsedQA(q, a, duration))

    for raw in text.replace("\r\n", "\n").split("\n"):
        match = _MARKER.match(raw)
        if match:
            kind = match.group("kind").lower()
            rest = match.group("rest")
            if kind in {"q", "question"}:
                if question or answer:
                    flush()
                    question, answer = [], []
                mode = "q"
                question = [rest]
            elif question:
                mode = "a"
                answer = [rest]
            continue
        if mode == "q":
            question.append(raw)
        elif mode == "a":
            answer.append(raw)

    if question or answer:
        flush()
    return pairs
