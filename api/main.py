from __future__ import annotations

import logging
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Response, Security, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.security import APIKeyHeader

from api import services
from api.errors import ApiError
from api.schemas import (
    AskRequest,
    AskResponse,
    DocumentsResponse,
    HealthResponse,
    IngestResponse,
    OCRResponse,
)

logger = logging.getLogger("api")
STATIC_DIR = Path(__file__).resolve().parent / "static"


def _max_bytes() -> int:
    return int(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024


def create_app(pipeline_getter=None, engine_getter=None) -> FastAPI:
    """Factory. Getters are injectable for tests; defaults load the real pipeline / OCR lazily."""
    if pipeline_getter is None:
        from orchestrator import get_pipeline as pipeline_getter
    if engine_getter is None:
        from ocr.engine import get_ocr_engine as engine_getter

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:                                   # warm up: load documents + models before the first request
            pipeline_getter().store.initialize()
        except Exception:
            logger.exception("Startup warm-up failed (it will be retried on the first request).")
        yield

    app = FastAPI(
        title="Multi-Agent Document Assistant",
        description="Chat with your documents (Retriever -> Analyst -> Answer agents on LangGraph), "
                    "with image upload and OCR.",
        version="1.0.0",
        lifespan=lifespan,
    )

    origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
    if origins:
        app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])

    # ---- optional API-key protection: set API_KEY in .env to enable ----
    api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

    def require_api_key(key: str | None = Security(api_key_header)):
        expected = os.getenv("API_KEY")
        if expected and not (key and secrets.compare_digest(key, expected)):
            raise ApiError(401, "Invalid or missing API key.")

    protected = [Depends(require_api_key)]

    # ---- errors ----
    @app.exception_handler(ApiError)
    async def api_error_handler(request, error: ApiError):
        return JSONResponse(status_code=error.status_code, content={"detail": error.detail})

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request, error: Exception):
        if "OPENROUTER_API_KEY" in str(error):
            return JSONResponse(status_code=503, content={"detail": "The language model is not configured."})
        logger.exception("Unhandled error")
        return JSONResponse(status_code=500, content={"detail": "Internal server error."})

    # ---- web interface (the page is public; the API calls it makes use the API key) ----
    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse("/ui")

    @app.get("/ui", include_in_schema=False)
    def ui():
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

    # ---- system ----
    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health():
        return {
            "status": "ok",
            "documents": len(services.list_documents(pipeline_getter().store)),
            "ocr_engine": engine_getter().name,
        }

    @app.delete("/cache", status_code=204, dependencies=protected, tags=["system"])
    def clear_cache():
        pipeline_getter().cache.clear()
        return Response(status_code=204)

    # ---- documents ----
    @app.get("/documents", response_model=DocumentsResponse, dependencies=protected, tags=["documents"])
    def list_documents():
        return {"documents": services.list_documents(pipeline_getter().store)}

    @app.post("/documents", response_model=IngestResponse, status_code=201, dependencies=protected,
              tags=["documents"])
    def upload_document(file: UploadFile = File(..., description="PDF, DOCX or TXT")):
        data = services.read_limited(file.file, _max_bytes())
        return services.upload_document(pipeline_getter().store, file.filename, data)

    @app.delete("/documents/{name}", status_code=204, dependencies=protected, tags=["documents"])
    def delete_document(name: str):
        services.delete_document(pipeline_getter().store, name)
        return Response(status_code=204)

    @app.post("/documents/image", response_model=IngestResponse, status_code=201, dependencies=protected,
              tags=["images / OCR"])
    def upload_image_document(image: UploadFile = File(..., description="PNG, JPEG, WEBP, BMP, TIFF or GIF")):
        """OCR the image and index its text as a searchable, citable document."""
        data = services.read_limited(image.file, _max_bytes())
        return services.upload_image_document(pipeline_getter().store, engine_getter(), image.filename, data)

    # ---- OCR ----
    @app.post("/ocr", response_model=OCRResponse, dependencies=protected, tags=["images / OCR"])
    def ocr(image: UploadFile = File(...)):
        """Only extract text from an image (nothing is stored)."""
        data = services.read_limited(image.file, _max_bytes())
        return services.ocr_only(engine_getter(), data)

    # ---- chat ----
    @app.post("/ask", response_model=AskResponse, dependencies=protected, tags=["chat"])
    def ask(request: AskRequest):
        history = [m.model_dump() for m in request.history]
        return services.ask(pipeline_getter(), request.question, history)

    @app.post("/ask/stream", dependencies=protected, tags=["chat"])
    def ask_stream(request: AskRequest):
        """Same as /ask, but streams one JSON line per agent step (NDJSON), then the final result."""
        history = [m.model_dump() for m in request.history]
        return StreamingResponse(
            services.ask_stream(pipeline_getter(), request.question, history),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/ask/image", response_model=AskResponse, dependencies=protected, tags=["chat"])
    def ask_with_image(
        image: UploadFile = File(...),
        question: str = Form("", max_length=2000, description="Optional. Empty = the text in the image is the question."),
        history: str = Form("", description='Optional JSON list: [{"role": "user", "content": "..."}]'),
    ):
        """Ask a question about / with an image: its OCR text is added to the question."""
        data = services.read_limited(image.file, _max_bytes())
        return services.ask_with_image(
            pipeline_getter(), engine_getter(), question, data, services.parse_history(history)
        )

    # ---- Task 5: voice query + report generator (same API-key protection as the rest) ----
    from routers import report as report_router, voice as voice_router
    app.include_router(voice_router.router, dependencies=protected)
    app.include_router(report_router.router, dependencies=protected)

    return app


app = create_app()
