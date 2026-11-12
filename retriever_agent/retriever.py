import json

from shared.schemas import Evidence, EvidenceBundle
from retriever_agent.document_loader import load_document, split_documents
from retriever_agent.vector_store import create_vector_store
from retriever_agent.tools.query_rewriter import rewrite_query
from retriever_agent.tools.reranker import rerank_evidences
from retriever_agent.tools.context_selector import select_context


def prepare_documents(
    file_path,
    chunk_size: int = 800,
    chunk_overlap: int = 120
):
    """Load, clean, and chunk one supported document."""

    return split_documents(
        load_document(file_path),
        chunk_size,
        chunk_overlap
    )


def prepare_documents_from_files(
    file_paths,
    chunk_size: int = 800,
    chunk_overlap: int = 120
):
    """Load and chunk multiple PDF/DOCX/TXT files."""

    chunks = []

    for file_path in file_paths:
        chunks.extend(
            prepare_documents(
                file_path,
                chunk_size,
                chunk_overlap
            )
        )

    return chunks


def prepare_vector_store(file_path):
    """Backward-compatible helper: ingest one file and return ChromaDB."""

    chunks = prepare_documents(file_path)
    return create_vector_store(chunks)


def ingest_files(file_paths, persist_directory=None):
    """Ingest one or more files and return (chunks, vector_store)."""

    chunks = prepare_documents_from_files(file_paths)

    if persist_directory is None:
        vector_store = create_vector_store(chunks)
    else:
        vector_store = create_vector_store(
            chunks,
            persist_directory=persist_directory
        )

    return chunks, vector_store


def _parse_tool_json(value):
    """Parse JSON tool output when necessary."""

    if isinstance(value, str):
        return json.loads(value)

    return value


def retrieve_evidence(
    question: str,
    chunks,
    vector_store,
    *,
    document: str | None = None,
    page: int | None = None,
    conversation_context: str = "",
    semantic_k: int = 10,
    keyword_k: int = 10,
    final_k: int = 6,
    max_context_chars: int = 9000,
    use_query_rewriting: bool = True,
) -> EvidenceBundle:
    """
    Run the complete Retriever workflow.

    Flow:
    rewrite -> semantic + BM25 -> optional metadata lookup
    -> merge/deduplicate -> CrossEncoder rerank
    -> diverse context selection -> EvidenceBundle.
    """

    if not question or not question.strip():
        raise ValueError("question must not be empty")

    # Rewrite the question using conversation context when available.
    if use_query_rewriting:
        query = rewrite_query(
            query=question,
            conversation_context=conversation_context
        )
    else:
        query = question.strip()

    from retriever_agent.tools.semantic_search import (
        create_semantic_search_tool
    )
    from retriever_agent.tools.keyword_search import (
        create_keyword_search_tool
    )
    from retriever_agent.tools.metadata_filter import (
        create_metadata_filter_tool
    )

    # Semantic retrieval
    semantic_tool = create_semantic_search_tool(vector_store)

    semantic_results = _parse_tool_json(
        semantic_tool.invoke({
            "query": query,
            "k": semantic_k
        })
    )

    # Keyword retrieval
    keyword_tool = create_keyword_search_tool(chunks)

    keyword_results = _parse_tool_json(
        keyword_tool.invoke({
            "query": query,
            "k": keyword_k
        })
    )

    candidates = list(semantic_results) + list(keyword_results)

    # Optional metadata filtering
    if document is not None or page is not None:
        metadata_tool = create_metadata_filter_tool(vector_store)

        metadata_results = _parse_tool_json(
            metadata_tool.invoke({
                "document": document,
                "page": page
            })
        )

        candidates.extend(metadata_results)

        # Do not include other documents when a document is specified.
        if document is not None:
            candidates = [
                item for item in candidates
                if item.get("document") == document
            ]

        # Do not include other pages when a page is specified.
        if page is not None:
            candidates = [
                item for item in candidates
                if item.get("page") == page
            ]

    # Deduplicate candidates before reranking.
    deduplicated = []
    seen = set()

    for item in candidates:
        content = str(item.get("content", "")).strip()

        key = (
            item.get("document"),
            item.get("page"),
            " ".join(content.casefold().split())
        )

        if content and key not in seen:
            seen.add(key)
            deduplicated.append(item)

    # Rerank using the rewritten query.
    ranked = rerank_evidences(
        query,
        deduplicated,
        top_k=max(final_k * 4, 20)
    )

    # Select the final evidence context.
    selected = select_context(
        ranked,
        top_k=final_k,
        max_chars=max_context_chars
    )

    return EvidenceBundle(
        evidences=[
            Evidence.model_validate(item)
            for item in selected
        ]
    )


def retrieve_evidence_json(
    question: str,
    chunks,
    vector_store,
    **kwargs
) -> str:
    """
    JSON-string adapter for direct integration with Person B.

    Supports optional conversation_context through kwargs.
    """

    return retrieve_evidence(
        question,
        chunks,
        vector_store,
        **kwargs
    ).to_json()
