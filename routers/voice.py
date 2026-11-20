"""FastAPI routes for voice. Partner: app.include_router(voice.router)"""
import json

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from voice import service
from voice.speech_to_text import TranscriptionError

router = APIRouter(prefix="/voice", tags=["voice"])


@router.post("/transcribe")
def transcribe_audio(file: UploadFile = File(...), language: str | None = Form(None)):
    """Audio file -> text only."""
    try:
        return service.voice_to_text(file.file.read(), language)
    except TranscriptionError as error:
        raise HTTPException(status_code=422, detail=str(error))


@router.post("/ask")
def ask_by_voice(file: UploadFile = File(...), language: str | None = Form(None),
                 history: str | None = Form(None)):
    """Audio question -> transcript + answer. `history` is an optional JSON list like
    [{"role": "user", "content": "..."}]."""
    try:
        chat_history = json.loads(history) if history else []
    except json.JSONDecodeError:
        raise HTTPException(status_code=422, detail="history must be a JSON list.")
    try:
        return service.voice_ask(file.file.read(), chat_history, language)
    except TranscriptionError as error:
        raise HTTPException(status_code=422, detail=str(error))
