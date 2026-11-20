from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from api.errors import ApiError
from ocr.engine import InvalidImageError, OCRError, OCRUnavailableError
from ocr.service import compose_question, ingest_image

logger = logging.getLogger("api")

DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".txt"}

STEP_LABELS = {
    "rewrite": "Understanding the question",
    "cache_lookup": "Checking the semantic cache",
    "retrieve": "Retriever Agent: searching the documents",
    "collect": "Collecting evidence",
    "analyst": "Analyst Agent: checking the evidence",
    "followup": "Analyst asked the Retriever for more evidence",
    "finalize": "Analyst tools: tables, comparison, calculations",
    "answer": "Answer Agent: writing the answer",
    "cache_store": "Saving to cache",
}


def read_limited(fileobj, max_bytes: int) -> bytes:
    """Read an upload but never more than max_bytes (+1 to detect oversize)."""
    data = fileobj.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ApiError(413, f"File is too large (max {max_bytes // (1024 * 1024)} MB).")
    if not data:
        raise ApiError(400, "The uploaded file is empty.")
    return data


def safe_filename(filename: str | None) -> str:
    name = Path(filename or "").name.strip()
    if not name or name.startswith("."):
        raise ApiError(400, "A valid file name is required.")
    return name


def _ocr(engine, data: bytes) -> str:
    try:
        return engine.extract_text(data)
    except InvalidImageError as error:
        raise ApiError(415, str(error)) from error
    except OCRUnavailableError as error:
        raise ApiError(503, str(error)) from error
    except OCRError as error:
        raise ApiError(422, str(error)) from error


def list_documents(store) -> list[str]:
    return store.document_names()


def upload_document(store, filename: str | None, data: bytes) -> dict:
    name = safe_filename(filename)
    if Path(name).suffix.lower() not in DOCUMENT_EXTENSIONS:
        raise ApiError(415, "Unsupported file type. Supported: PDF, DOCX, TXT (use /documents/image for images).")

    incoming = store.incoming_path(name)
    incoming.write_bytes(data)
    try:
        message = store.ingest(incoming)
    except ValueError as error:
        incoming.unlink(missing_ok=True)
        raise ApiError(422, str(error)) from error
    except Exception:
        incoming.unlink(missing_ok=True)
        raise
    return {"document": name, "message": message}


def upload_image_document(store, engine, filename: str | None, data: bytes) -> dict:
    safe_filename(filename)
    try:
        result = ingest_image(store, filename, data, engine)
    except InvalidImageError as error:
        raise ApiError(415, str(error)) from error
    except OCRUnavailableError as error:
        raise ApiError(503, str(error)) from error
    except OCRError as error:
        raise ApiError(422, str(error)) from error
    except ValueError as error:
        raise ApiError(422, str(error)) from error
    return {
        "document": result["document"],
        "message": result["message"],
        "characters": result["characters"],
        "ocr_text": result["text"],
    }


def delete_document(store, name: str) -> None:
    try:
        store.delete_document(name)
    except KeyError:
        raise ApiError(404, f"Document not found: {name}") from None
    except ValueError as error:
        raise ApiError(400, str(error)) from error


def ocr_only(engine, data: bytes) -> dict:
    text = _ocr(engine, data)
    return {"text": text, "characters": len(text), "engine": engine.name}


def parse_history(raw: str | None) -> list[dict]:
    """History arrives as a JSON string in multipart requests."""
    if not raw or not raw.strip():
        return []
    try:
        items = json.loads(raw)
        return [{"role": m["role"], "content": str(m["content"])} for m in items][-20:]
    except (ValueError, KeyError, TypeError):
        raise ApiError(422, 'history must be a JSON list like [{"role": "user", "content": "..."}].') from None


def ask(pipeline, question: str, history: list[dict] | None = None, ocr_text: str | None = None) -> dict:
    started = time.perf_counter()
    reply = pipeline.ask(question, history or [])
    return {
        "answer": reply.answer,
        "cached": reply.cached,
        "analysis": reply.analysis,
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
        "ocr_text": ocr_text,
    }


def ask_with_image(pipeline, engine, question: str, data: bytes, history: list[dict] | None = None) -> dict:
    text = _ocr(engine, data)
    return ask(pipeline, compose_question(question, text), history, ocr_text=text)


def step_detail(node: str, update: dict) -> str:
    """One short human-readable detail per graph step (shown next to the step in the UI)."""
    if node == "rewrite":
        return update.get("query", "")
    if node == "cache_lookup":
        return "hit" if update.get("cached") else "miss"
    if node == "retrieve":
        return f"{len(update.get('retrieved', []))} chunks"
    if node == "collect":
        return f"{len(update.get('evidences', []))} evidence chunks in total"
    if node == "analyst":
        if update.get("follow_up"):
            return f"not enough evidence, asking for: {update['follow_up']}"
        decision = update.get("decision")
        return "enough evidence" if decision is not None and decision.enough_evidence else "stopping"
    if node == "followup":
        return update.get("search_query", "")
    return ""


def _line(event: dict) -> str:
    return json.dumps(event, ensure_ascii=False) + "\n"


def ask_stream(pipeline, question: str, history: list[dict] | None = None):
    """NDJSON events: {"type": "step"|"result"|"error", ...}. One JSON object per line."""
    started = time.perf_counter()
    try:
        for event in pipeline.stream(question, history or []):
            if event[0] == "step":
                node, update = event[1], event[2]
                yield _line({
                    "type": "step",
                    "node": node,
                    "label": STEP_LABELS.get(node, node),
                    "detail": step_detail(node, update),
                })
            else:
                reply = event[1]
                yield _line({
                    "type": "result",
                    "answer": reply.answer,
                    "cached": reply.cached,
                    "analysis": reply.analysis,
                    "elapsed_ms": int((time.perf_counter() - started) * 1000),
                    "ocr_text": None,
                })
    except Exception as error:                 # the HTTP status is already sent: report inside the stream
        logger.exception("Streaming request failed")
        message = ("The language model is not configured."
                   if "OPENROUTER_API_KEY" in str(error) else "Internal server error.")
        yield _line({"type": "error", "detail": message})
