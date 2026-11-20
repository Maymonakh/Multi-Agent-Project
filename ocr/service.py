from __future__ import annotations

import re
from pathlib import Path

MAX_QUESTION_OCR_CHARS = 1500


def ocr_document_name(filename: str) -> str:
    stem = re.sub(r"[^\w\-. ]+", "_", Path(filename or "image").stem, flags=re.UNICODE).strip(" ._") or "image"
    return f"{stem[:80]}_ocr.txt"


def ingest_image(store, filename: str, data: bytes, engine) -> dict:
    """OCR the image and index the text as a document. Returns {document, characters, text}."""
    text = engine.extract_text(data)

    path = store.incoming_path(ocr_document_name(filename))
    path.write_text(text, encoding="utf-8")
    try:
        message = store.ingest(path)
    except Exception:
        path.unlink(missing_ok=True)
        raise

    return {"document": path.name, "characters": len(text), "text": text, "message": message}


def compose_question(question: str, ocr_text: str) -> str:
    """Combine the user's question with text read from an image."""
    question = (question or "").strip()
    snippet = ocr_text.strip()[:MAX_QUESTION_OCR_CHARS]
    if not question:
        return snippet                                   # the image IS the question
    return f"{question}\n\nText from the attached image:\n{snippet}"
