"""Speech to text with faster-whisper (runs on your computer, no cloud, Arabic and English)."""
import io
import os
from functools import lru_cache


class TranscriptionError(Exception):
    """Raised with a message that is safe to show to the user."""


@lru_cache(maxsize=1)
def _load_model():
    from faster_whisper import WhisperModel
    # "small" is a good balance. Use "base" for speed, "medium" for accuracy.
    return WhisperModel(os.getenv("WHISPER_MODEL", "small"), device="cpu", compute_type="int8")


def transcribe(audio: bytes, language: str | None = None) -> dict:
    """audio bytes (wav, mp3, webm, ogg...) -> {"text": ..., "language": ...}.
    language=None lets Whisper detect it (for example "ar" or "en")."""
    if not audio:
        raise TranscriptionError("No audio received.")

    model = _load_model()
    try:
        segments, info = model.transcribe(
            io.BytesIO(audio), language=language or None, beam_size=5, vad_filter=True
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
    except Exception as error:
        raise TranscriptionError(f"Could not read the audio: {error}") from error

    if not text:
        raise TranscriptionError("No speech detected. Please try again and speak clearly.")
    return {"text": text, "language": info.language}
