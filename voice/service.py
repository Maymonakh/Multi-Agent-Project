"""Voice query: audio -> text -> the SAME pipeline as a typed question -> answer."""
from voice.speech_to_text import transcribe, TranscriptionError  # noqa: F401


def _default_ask(question: str, history: list):
    from orchestrator import ask          # imported late: keeps this module light
    return ask(question, history)


def voice_to_text(audio: bytes, language: str | None = None) -> dict:
    return transcribe(audio, language)


def voice_ask(audio: bytes, history: list | None = None, language: str | None = None) -> dict:
    heard = transcribe(audio, language)
    reply = _default_ask(heard["text"], history or [])
    return {
        "transcript": heard["text"],
        "language": heard["language"],
        "answer": reply.answer,
        "cached": reply.cached,
        "analysis": reply.analysis,
    }
