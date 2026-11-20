from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, NamedTuple

from retriever_agent.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DB_PATH,
    DOCUMENTS_DIR,
    EMBEDDING_MODEL,
)
from retriever_agent.tools.keyword_search import BM25Index
from shared.schemas import Chunk

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}
INCOMING_DIR = ".incoming"      # uploads wait here until they were indexed successfully


class Snapshot(NamedTuple):
    vector_store: Any
    keyword_index: BM25Index
    documents: tuple


class DocumentStore:
    def __init__(
        self,
        documents_dir: Path | str = DOCUMENTS_DIR,
        db_path: Path | str = DB_PATH,
        *,
        prepare_file=None,
        prepare_files=None,
        get_vector_store=None,
        create_vector_store=None,
    ):
        self.documents_dir = Path(documents_dir)
        self.db_path = Path(db_path)
        self._prepare_file = prepare_file
        self._prepare_files = prepare_files
        self._get_vector_store = get_vector_store
        self._create_vector_store = create_vector_store

        self._lock = threading.RLock()
        self._ready = False
        self._chunks: list = []
        self._vector_store = None
        self._index = BM25Index([])
        self._version = ""
        self._hash_memo: dict = {}

    # ---- backends (imported lazily so importing this module stays cheap) ----
    def _backends(self):
        if self._prepare_file is None:
            from retriever_agent.document_loader import prepare_documents, prepare_documents_from_files
            from retriever_agent.vector_store import create_vector_store, get_vector_store

            self._prepare_file = prepare_documents
            self._prepare_files = prepare_documents_from_files
            self._get_vector_store = lambda: get_vector_store(self.db_path)
            self._create_vector_store = create_vector_store

    # ---- files / versioning ----
    def _document_files(self) -> list[Path]:
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        return sorted(
            p for p in self.documents_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )

    def _file_hash(self, path: Path) -> str:
        stat = path.stat()
        key = (path.name, stat.st_size, stat.st_mtime_ns)
        if key not in self._hash_memo:
            self._hash_memo[key] = hashlib.sha256(path.read_bytes()).hexdigest()
        return self._hash_memo[key]

    def _signature(self, files: list[Path] | None = None) -> str:
        """Content-based: copying/unzipping the project does not trigger a re-index."""
        files = self._document_files() if files is None else files
        parts = [f"model={EMBEDDING_MODEL}", f"chunk={CHUNK_SIZE}/{CHUNK_OVERLAP}"]
        parts += [f"{p.name}:{self._file_hash(p)}" for p in files]
        return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()

    @property
    def _manifest(self) -> Path:
        return self.db_path / ".ingestion_manifest.json"

    def _read_manifest(self) -> str | None:
        try:
            return json.loads(self._manifest.read_text(encoding="utf-8")).get("signature")
        except (FileNotFoundError, json.JSONDecodeError, OSError, AttributeError):
            return None

    def _write_manifest(self, signature: str) -> None:
        self._manifest.parent.mkdir(parents=True, exist_ok=True)
        self._manifest.write_text(json.dumps({"signature": signature}, indent=2), encoding="utf-8")

    # ---- lifecycle ----
    def initialize(self) -> None:
        if self._ready:
            return
        with self._lock:
            if self._ready:
                return
            self._backends()

            files = self._document_files()
            signature = self._signature(files)
            vector_store = self._get_vector_store()

            stored = vector_store.get(include=["documents", "metadatas"])
            stored_count = len(stored.get("ids") or [])

            up_to_date = (
                self._read_manifest() == signature
                and ((bool(files) and stored_count > 0) or (not files and stored_count == 0))
            )

            if up_to_date:
                chunks = [
                    Chunk(page_content=text, metadata=dict(meta or {}))
                    for text, meta in zip(stored.get("documents") or [], stored.get("metadatas") or [])
                    if text
                ]
            else:
                chunks = list(self._prepare_files(files)) if files else []
                if stored_count:
                    vector_store.delete(ids=stored["ids"])
                if chunks:
                    vector_store = self._create_vector_store(chunks, persist_directory=self.db_path)
                self._write_manifest(signature)

            self._install(chunks, vector_store, signature)
            self._ready = True

    def _install(self, chunks, vector_store, signature: str) -> None:
        self._chunks = chunks
        self._vector_store = vector_store
        self._index = BM25Index(chunks)
        self._version = signature

    # ---- public API ----
    @property
    def version(self) -> str:
        """Changes whenever the documents change (used to scope the answer cache)."""
        self.initialize()
        return self._version

    def document_names(self) -> list[str]:
        return list(self.snapshot().documents)

    def snapshot(self) -> Snapshot:
        self.initialize()
        with self._lock:
            names = sorted({c.metadata.get("document", "Unknown") for c in self._chunks})
            return Snapshot(self._vector_store, self._index, tuple(names))

    def ingest(self, file_path: str | Path) -> str:
        """Index (or replace) one PDF/DOCX/TXT. Files in `.incoming/` are moved into place on success."""
        path = Path(file_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Uploaded document not found: {path}")
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {path.suffix}. Supported formats: PDF, DOCX, TXT.")

        self.initialize()
        new_chunks = list(self._prepare_file(path))      # heavy work outside the lock
        if not new_chunks:
            raise ValueError(f"No readable text was found in {path.name}.")

        with self._lock:
            vector_store = self._create_vector_store(
                new_chunks, persist_directory=self.db_path, replace_documents=True
            )

            destination = self.documents_dir.resolve() / path.name
            if path != destination:
                os.replace(path, destination)

            kept = [c for c in self._chunks if c.metadata.get("document") != path.name]
            signature = self._signature()
            self._install(kept + new_chunks, vector_store, signature)
            self._write_manifest(signature)

        return f"Successfully ingested {path.name} ({len(new_chunks)} chunks)."

    def delete_document(self, name: str) -> None:
        """Remove one document from the folder, the vector store and the keyword index."""
        if Path(name).name != name:
            raise ValueError("Invalid document name.")

        self.initialize()
        with self._lock:
            if name not in {c.metadata.get("document") for c in self._chunks}:
                raise KeyError(name)

            self._vector_store.delete(where={"document": name})
            (self.documents_dir / name).unlink(missing_ok=True)

            signature = self._signature()
            self._install(
                [c for c in self._chunks if c.metadata.get("document") != name],
                self._vector_store,
                signature,
            )
            self._write_manifest(signature)

    def incoming_path(self, filename: str) -> Path:
        """Where an upload should be written before `ingest()`."""
        folder = self.documents_dir / INCOMING_DIR
        folder.mkdir(parents=True, exist_ok=True)
        return folder / Path(filename).name
