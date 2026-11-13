from __future__ import annotations

import hashlib
import json
import re
import threading
from pathlib import Path

from shared.schemas import EvidenceBundle


BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}

USING_FAKE = False

_lock = threading.RLock()
_initialized = False
_chunks = []
_vector_store = None


def _document_files() -> list[Path]:
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

    return sorted(
        path
        for path in DOCUMENTS_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _files_signature(files: list[Path] | None = None) -> str:
    """Return a stable version for the current source-document collection."""

    files = _document_files() if files is None else files

    entries = []

    for path in files:
        stat = path.stat()
        entries.append(
            f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}"
        )

    raw = "\n".join(entries)

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()


def get_corpus_version() -> str:
    """Version used to prevent cached answers surviving document changes."""

    return _files_signature()


def _get_message_role_content(message) -> tuple[str, str]:
    """Normalize Streamlit dictionaries and LangChain message objects."""

    if isinstance(message, dict):
        role = str(message.get("role", "user"))
        content = str(message.get("content", ""))
    else:
        role = str(
            getattr(
                message,
                "type",
                getattr(message, "role", "user"),
            )
        )
        content = str(getattr(message, "content", ""))

    role = role.casefold().strip()

    if role in {"human", "user"}:
        role = "user"
    elif role in {"ai", "assistant"}:
        role = "assistant"

    return role, content.strip()


def build_conversation_context(
    chat_history: list | None,
    current_question: str = "",
    *,
    max_messages: int = 10,
    max_chars: int = 5000,
) -> str:
    """
    Format conversation history for contextual retrieval.

    Excludes:
    - The current question if already appended to chat history.
    - Previous occurrences of the exact same question.
    - The assistant response immediately following those occurrences.

    This keeps the cache context stable when the user repeats a question,
    while preserving unrelated conversation turns.
    """

    if not chat_history:
        return ""

    messages = [
        _get_message_role_content(message)
        for message in chat_history
    ]

    normalized_question = current_question.strip()

    if messages:
        last_role, last_content = messages[-1]

        if (
            last_role == "user"
            and last_content == normalized_question
        ):
            messages.pop()

    # Remove previous identical question/answer pairs.
    # Other conversation turns remain available as context.
    filtered_messages = []

    index = 0

    while index < len(messages):
        role, content = messages[index]

        if (
            normalized_question
            and role == "user"
            and content == normalized_question
        ):
            index += 1

            # Skip the direct assistant response to this repeated question.
            if index < len(messages):
                next_role, _ = messages[index]

                if next_role == "assistant":
                    index += 1

            continue

        filtered_messages.append((role, content))
        index += 1

    # Keep only the most recent relevant messages.
    filtered_messages = filtered_messages[-max_messages:]

    lines = []

    for role, content in filtered_messages:
        if not content:
            continue

        display_role = (
            "User" if role == "user" else
            "Assistant" if role == "assistant" else
            role.capitalize()
        )

        lines.append(f"{display_role}: {content}")

    return "\n".join(lines)[-max_chars:]


def _manifest_path():
    from retriever_agent.vector_store import DB_PATH

    return Path(DB_PATH) / ".ingestion_manifest.json"


def _read_manifest_signature() -> str | None:
    path = _manifest_path()

    try:
        return json.loads(
            path.read_text(encoding="utf-8")
        ).get("signature")

    except (
        FileNotFoundError,
        json.JSONDecodeError,
        OSError,
        AttributeError,
    ):
        return None


def _write_manifest(signature: str) -> None:
    path = _manifest_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps({"signature": signature}, indent=2),
        encoding="utf-8",
    )


def _clear_vector_store(vector_store) -> None:
    """Remove all stored vectors before rebuilding from the source folder."""

    existing = vector_store.get(include=["metadatas"])
    ids = existing.get("ids", [])

    if ids:
        vector_store.delete(ids=ids)


def _ensure_initialized() -> None:
    """Load source files and synchronize Chroma once per Python process."""

    global _initialized, _chunks, _vector_store

    if _initialized:
        return

    with _lock:
        if _initialized:
            return

        from retriever_agent.document_loader import (
            prepare_documents_from_files,
        )

        from retriever_agent.vector_store import (
            DB_PATH,
            create_vector_store,
            get_vector_store,
        )

        files = _document_files()
        signature = _files_signature(files)

        chunks = (
            prepare_documents_from_files(files)
            if files
            else []
        )

        vector_store = get_vector_store()

        try:
            stored_count = vector_store._collection.count()
        except Exception:
            stored_count = 0

        needs_rebuild = (
            _read_manifest_signature() != signature
            or (bool(files) and stored_count == 0)
            or (not files and stored_count > 0)
        )

        if needs_rebuild:
            _clear_vector_store(vector_store)

            if chunks:
                vector_store = create_vector_store(
                    chunks,
                    persist_directory=DB_PATH,
                )

            _write_manifest(signature)

        _chunks = chunks
        _vector_store = vector_store
        _initialized = True


def initialize() -> None:
    """Explicitly initialize document/vector state before a batch upload."""

    _ensure_initialized()


def _extract_agent_text(agent_result) -> str:

    if not isinstance(agent_result, dict):
        raise RuntimeError(
            "Retriever Agent returned an unexpected result type: "
            f"{type(agent_result).__name__}."
        )

    messages = agent_result.get("messages", [])
    for message in reversed(messages):
        message_type = getattr(message, "type", "")
        if message_type != "ai":
            continue

        content = getattr(message, "content", "")
        if isinstance(content, str) and content.strip():
            return content.strip()

        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(str(block.get("text", "")))
            combined = "\n".join(part for part in text_parts if part.strip())
            if combined.strip():
                return combined.strip()

    raise RuntimeError(
        "Retriever Agent did not return a final AI message containing JSON."
    )


def _parse_agent_bundle(agent_result) -> EvidenceBundle:
    """Validate the agent's final JSON against the shared EvidenceBundle schema."""

    raw_text = _extract_agent_text(agent_result)
    cleaned_text = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        raw_text.strip(),
        flags=re.IGNORECASE,
    ).strip()

    try:
        payload = json.loads(cleaned_text)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "Retriever Agent returned invalid JSON. "
            f"Output: {raw_text[:1000]}"
        ) from error

    try:
        return EvidenceBundle.model_validate(payload)
    except Exception as error:
        raise RuntimeError(
            "Retriever Agent JSON does not match the EvidenceBundle schema. "
            f"Output: {raw_text[:1000]}"
        ) from error


def retrieve(
    query: str,
    chat_history: list | None = None,
    llm=None,
) -> EvidenceBundle:
    """Run the LangChain Retriever Agent and return a typed EvidenceBundle."""

    if not query or not query.strip():
        raise ValueError("The retrieval query must not be empty.")

    _ensure_initialized()

    if not _chunks:
        return EvidenceBundle(evidences=[])

    from retriever_agent.agent import create_retriever_agent
    from shared.llm_config import get_llm

    active_llm = llm if llm is not None else get_llm()
    context = build_conversation_context(
        chat_history,
        current_question=query,
    )

    user_message = (
        "Conversation context:\n"
        + (context if context else "(No previous conversation context.)")
        + "\n\nLatest question:\n"
        + query.strip()
    )

    agent = create_retriever_agent(
        chunks=_chunks,
        vector_store=_vector_store,
        llm=active_llm,
    )

    try:
        result = agent.invoke({
            "messages": [{"role": "user", "content": user_message}]
        })
    except Exception as error:
        raise RuntimeError(
            f"LangChain Retriever Agent execution failed: {error}"
        ) from error

    return _parse_agent_bundle(result)


def ingest(file_path: str) -> str:
    """Ingest or replace one uploaded PDF/DOCX/TXT file."""

    global _chunks, _vector_store

    path = Path(file_path).resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"Uploaded document not found: {path}"
        )

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {path.suffix}. "
            "Supported formats: PDF, DOCX, TXT."
        )

    _ensure_initialized()

    from retriever_agent.document_loader import prepare_documents

    from retriever_agent.vector_store import (
        DB_PATH,
        create_vector_store,
    )

    new_chunks = prepare_documents(path)

    if not new_chunks:
        raise ValueError(
            f"No readable text was found in {path.name}."
        )

    with _lock:
        _vector_store = create_vector_store(
            new_chunks,
            persist_directory=DB_PATH,
            replace_documents=True,
        )

        _chunks = [
            chunk
            for chunk in _chunks
            if chunk.metadata.get("document") != path.name
        ]

        _chunks.extend(new_chunks)

        _write_manifest(_files_signature())

    return (
        f"Successfully ingested {path.name} "
        f"({len(new_chunks)} chunks)."
    )


def reset_bridge_state() -> None:
    """Test helper: force document and vector-store state to reload."""

    global _initialized, _chunks, _vector_store

    with _lock:
        _initialized = False
        _chunks = []
        _vector_store = None
