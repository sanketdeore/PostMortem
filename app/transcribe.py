"""Local transcription. Audio is written to a temp file and deleted."""

from ..config import settings

_model = None


def transcribe_path(path: str) -> str:
    """Run faster-whisper on a local file. The caller deletes the file."""
    global _model
    from faster_whisper import WhisperModel

    if _model is None:
        _model = WhisperModel(settings.whisper_model, device="cpu", compute_type="int8")
    segments, _info = _model.transcribe(path)
    return " ".join(segment.text.strip() for segment in segments).strip()
