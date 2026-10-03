"""Pick the analyzer named by ANALYZER. Unknown values fall back to heuristic."""

from ..config import settings
from .backboard import BackboardAnalyzer
from .heuristic import HeuristicAnalyzer
from .ollama import OllamaAnalyzer
from .tinker import TinkerFineTunedAnalyzer


def get_analyzer(name: str | None = None):
    """heuristic | ollama | backboard | tinker.

    The tinker slot is TinkerFineTunedAnalyzer (wired in the fine-tune phase).
    """
    chosen = (name or settings.analyzer or "heuristic").strip().lower()
    if chosen == "ollama":
        return OllamaAnalyzer()
    if chosen == "backboard":
        return BackboardAnalyzer()
    if chosen == "tinker":
        return TinkerFineTunedAnalyzer()
    return HeuristicAnalyzer()
