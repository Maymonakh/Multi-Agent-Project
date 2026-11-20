"""Speech to text with faster-whisper (runs on your computer, no cloud, Arabic and English)."""
import io
import os
import wave
from functools import lru_cache

import numpy as np

SAMPLE_RATE = 16000   # Whisper always works at 16 kHz


class TranscriptionError(Exception):
    """Raised with a message that is safe to show to the user."""


@lru_cache(maxsize=1)
def _load_model():
    from faster_whisper import WhisperModel
    # "small" is a good balance. Use "base" for speed, "medium" for accuracy.
    return WhisperModel(os.getenv("WHISPER_MODEL", "small"), device="cpu", compute_type="int8")


def _decode_wav(audio: bytes):
    """Decode a PCM WAV (what Streamlit's microphone recorder produces) with the standard
    library only -> mono float32 at 16 kHz. Returns None if it is not a plain PCM WAV.
    This avoids PyAV, so a broken/outdated `av` install can no longer break voice input."""
    try:
        with wave.open(io.BytesIO(audio), "rb") as wav:
            channels, width, rate = wav.getnchannels(), wav.getsampwidth(), wav.getframerate()
            frames = wav.readframes(wav.getnframes())
    except (wave.Error, EOFError):
        return None

    if width == 1:                                   # 8-bit unsigned
        data = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:                                 # 16-bit signed (the usual case)
        data = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:                                 # 32-bit signed
        data = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        return None

    if channels > 1:                                 # stereo -> mono
        data = data[: len(data) // channels * channels].reshape(-1, channels).mean(axis=1)
    if rate != SAMPLE_RATE and len(data) > 1:        # resample to 16 kHz
        target = int(len(data) * SAMPLE_RATE / rate)
        data = np.interp(np.linspace(0, len(data) - 1, target),
                         np.arange(len(data)), data).astype(np.float32)
    return np.ascontiguousarray(data, dtype=np.float32)


def _prepare_audio(audio: bytes):
    """WAV -> numpy array (no PyAV). Anything else (mp3, webm, ogg...) -> let faster-whisper
    decode it, which needs a working `av` package."""
    decoded = _decode_wav(audio)
    return decoded if decoded is not None else io.BytesIO(audio)


def transcribe(audio: bytes, language: str | None = None) -> dict:
    """audio bytes (wav, mp3, webm, ogg...) -> {"text": ..., "language": ...}.
    language=None lets Whisper detect it (for example "ar" or "en")."""
    if not audio:
        raise TranscriptionError("No audio received.")

    model = _load_model()
    try:
        segments, info = model.transcribe(
            _prepare_audio(audio), language=language or None, beam_size=5, vad_filter=True
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
    except Exception as error:
        raise TranscriptionError(f"Could not read the audio: {error}") from error

    if not text:
        raise TranscriptionError("No speech detected. Please try again and speak clearly.")
    return {"text": text, "language": info.language}