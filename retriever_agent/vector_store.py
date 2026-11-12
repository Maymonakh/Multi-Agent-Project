import hashlib
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DB_PATH = Path(__file__).resolve().parent.parent / "chroma_db"
COLLECTION_NAME = "external_knowledge"
_embeddings = None


def get_embeddings():
    """Load the embedding model once, on first use."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def get_vector_store(persist_directory: str | Path = DB_PATH):
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(persist_directory),
    )


def _chunk_id(chunk, index: int) -> str:
    source = chunk.metadata.get("document", "unknown")
    page = chunk.metadata.get("page")
    start = chunk.metadata.get("start_index", index)
    digest = hashlib.sha1(chunk.page_content.encode("utf-8")).hexdigest()[:12]
    return f"{source}:{page}:{start}:{digest}"


def create_vector_store(chunks, persist_directory: str | Path = DB_PATH, replace_documents: bool = True):
    """Persist chunks; replace old chunks for each ingested source file."""
    vector_store = get_vector_store(persist_directory)
    if not chunks:
        return vector_store

    grouped = {}
    for chunk in chunks:
        name = chunk.metadata.get("document", "Unknown")
        grouped.setdefault(name, []).append(chunk)

    for document_name, document_chunks in grouped.items():
        if replace_documents:
            vector_store.delete(where={"document": document_name})
        ids = [_chunk_id(chunk, i) for i, chunk in enumerate(document_chunks)]
        vector_store.add_documents(documents=document_chunks, ids=ids)
    return vector_store
