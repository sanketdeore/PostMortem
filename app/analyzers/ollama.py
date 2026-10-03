"""Local qwen2.5:7b through Ollama. Never raises out of analyze()."""

import httpx

from ..config import settings
from ..models import QA
from .coaching import heuristic_coach
from .heuristic import HeuristicAnalyzer
from .prompts import coach_prompt, extract_json, question_prompt, system_prompt, user_prompt
from .questions import heuristic_questions
from .schemas import CoachNote, QuestionList, TagDraft, TagList, snap_quote


class OllamaAnalyzer:
    name = "ollama"

    def __init__(self, base_url: str | None = None, model: str | None = None):
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.ollama_model
        self.name = f"ollama:{self.model}"
        self.used_fallback = False

    def _chat(self, system: str, user: str) -> str:
        response = httpx.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=180.0,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    def analyze(self, qa: QA, preferences: dict[str, str]) -> list[TagDraft]:
        try:
            raw = self._chat(system_prompt(preferences), user_prompt(qa))
            parsed = TagList.model_validate_json(extract_json(raw))
            tags = [
                TagDraft(
                    tag=item.tag,
                    severity=item.severity,
                    evidence_quote=snap_quote(qa.her_answer, item.evidence_quote),
                )
                for item in parsed.tags
            ]
            self.used_fallback = False
            return tags[:5]
        except Exception:
            self.used_fallback = True
            return HeuristicAnalyzer().analyze(qa, preferences)

    def generate_questions(self, role: str, focus_tags: list[str], n: int = 5) -> list[str]:
        try:
            raw = self._chat(
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
            raw = self._chat(
                "You rewrite interview answers. Return JSON only. Do not invent facts.",
                coach_prompt(qa, [tag.tag for tag in tags]),
            )
            note = CoachNote.model_validate_json(extract_json(raw))
            # The saved score comes from the tags so a chatty model can't inflate it.
            note.score = fallback.score
            self.used_fallback = False
            return note
        except Exception:
            self.used_fallback = True
            return fallback
