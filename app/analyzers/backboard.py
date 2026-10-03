"""Backboard analyzer. Same contract as the others, open-weight models only."""

from ..backboard_client import post_message
from ..config import settings
from ..models import QA
from .coaching import heuristic_coach
from .heuristic import HeuristicAnalyzer
from .prompts import coach_prompt, extract_json, question_prompt, system_prompt, user_prompt
from .questions import heuristic_questions
from .schemas import CoachNote, QuestionList, TagDraft, TagList, snap_quote


class BackboardAnalyzer:
    def __init__(self, model: str | None = None):
        self.model = model or settings.backboard_model
        self.name = f"backboard:{self.model}"
        self.used_fallback = False

    def _complete(self, system: str, user: str) -> str:
        # memory off and no thread_id: analysis must not write into her drill history.
        payload, _latency = post_message(
            user,
            model=self.model,
            system_prompt=system,
            memory="off",
            json_output=True,
        )
        content = payload.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("empty Backboard response")
        return content

    def analyze(self, qa: QA, preferences: dict[str, str]) -> list[TagDraft]:
        try:
            raw = self._complete(system_prompt(preferences), user_prompt(qa))
            parsed = TagList.model_validate_json(extract_json(raw))
            self.used_fallback = False
            return [
                TagDraft(
                    tag=item.tag,
                    severity=item.severity,
                    evidence_quote=snap_quote(qa.her_answer, item.evidence_quote),
                )
                for item in parsed.tags
            ][:5]
        except Exception:
            self.used_fallback = True
            return HeuristicAnalyzer().analyze(qa, preferences)

    def generate_questions(self, role: str, focus_tags: list[str], n: int = 5) -> list[str]:
        try:
            raw = self._complete(
                "You write practice interview questions. Return JSON only.",
                question_prompt(role, focus_tags, n),
            )
            parsed = QuestionList.model_validate_json(extract_json(raw))
            questions = [q.strip() for q in parsed.questions if q.strip()]
            if len(questions) < n:
                raise ValueError("model returned too few questions")
            self.used_fallback = False
            return questions[:n]
        except Exception:
            self.used_fallback = True
            return heuristic_questions(role, focus_tags, n)

    def coach(self, qa: QA, tags: list[TagDraft], preferences: dict[str, str]) -> CoachNote:
        fallback = heuristic_coach(qa, tags)
        try:
            raw = self._complete(
                "You rewrite interview answers. Return JSON only. Do not invent facts.",
                coach_prompt(qa, [tag.tag for tag in tags]),
            )
            note = CoachNote.model_validate_json(extract_json(raw))
            note.score = fallback.score
            self.used_fallback = False
            return note
        except Exception:
            self.used_fallback = True
            return fallback
