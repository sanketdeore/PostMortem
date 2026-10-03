"""Fine-tuned Qwen served by Tinker. Same analyze() contract, heuristic fallback.

Sampling uses the Tinker SDK (TINKER_API_KEY + TINKER_MODEL_NAME). The model
path is whatever scripts/train_tinker.py wrote after the LoRA run.
"""

import os

from ..config import settings
from ..models import QA
from .coaching import heuristic_coach
from .heuristic import HeuristicAnalyzer
from .prompts import coach_prompt, extract_json, question_prompt, system_prompt, user_prompt
from .questions import heuristic_questions
from .schemas import CoachNote, QuestionList, TagDraft, TagList, snap_quote

_client = None
_tokenizer = None
_renderer = None


def _get_client(model_name: str, api_key: str):
    global _client, _tokenizer, _renderer
    if _client is None:
        import tinker
        from tinker_cookbook.model_info import get_recommended_renderer_name
        from tinker_cookbook.renderers import get_renderer

        os.environ.setdefault("TINKER_API_KEY", api_key)
        service = tinker.ServiceClient()
        _client = service.create_sampling_client(model_path=model_name)
        _tokenizer = _client.get_tokenizer()
        try:
            renderer_name = get_recommended_renderer_name(_client.get_base_model())
        except Exception:
            renderer_name = "qwen3"
        _renderer = get_renderer(renderer_name, _tokenizer)
    return _client, _tokenizer, _renderer


class TinkerFineTunedAnalyzer:
    def __init__(self, model_name: str | None = None, api_key: str | None = None):
        self.model_name = model_name or settings.tinker_model_name
        self.api_key = api_key or settings.tinker_api_key
        self.name = f"tinker:{self.model_name or 'unset'}"
        self.used_fallback = False

    def _sample(self, messages: list[dict[str, str]], max_tokens: int = 600) -> str:
        if not self.api_key or not self.model_name:
            raise RuntimeError("TINKER_API_KEY or TINKER_MODEL_NAME is not set")
        import tinker

        client, _tokenizer, renderer = _get_client(self.model_name, self.api_key)
        # build_generation_prompt already returns a ModelInput. Encoding it
        # again would throw and every call would silently fall back.
        model_input = renderer.build_generation_prompt(messages)
        result = client.sample(
            model_input,
            1,
            tinker.SamplingParams(
                max_tokens=max_tokens,
                temperature=0.0,
                stop=renderer.get_stop_sequences(),
            ),
        ).result()
        message, _termination = renderer.parse_response(result.sequences[0].tokens)
        content = message["content"] if isinstance(message, dict) else message.content
        if isinstance(content, list):
            content = "".join(
                part if isinstance(part, str) else part.get("text", "")
                for part in content
            )
        return content or ""

    def analyze(self, qa: QA, preferences: dict[str, str]) -> list[TagDraft]:
        try:
            raw = self._sample(
                [
                    {"role": "system", "content": system_prompt(preferences)},
                    {"role": "user", "content": user_prompt(qa)},
                ]
            )
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
            raw = self._sample(
                [
                    {
                        "role": "system",
                        "content": "You write practice interview questions. Return JSON only.",
                    },
                    {"role": "user", "content": question_prompt(role, focus_tags, n)},
                ]
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
            raw = self._sample(
                [
                    {
                        "role": "system",
                        "content": "You rewrite interview answers. Return JSON only. Do not invent facts.",
                    },
                    {"role": "user", "content": coach_prompt(qa, [tag.tag for tag in tags])},
                ]
            )
            note = CoachNote.model_validate_json(extract_json(raw))
            note.score = fallback.score
            self.used_fallback = False
            return note
        except Exception:
            self.used_fallback = True
            return fallback
