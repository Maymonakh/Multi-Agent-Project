
from langchain.agents import create_agent

from shared.llm_config import get_llm

from retriever_agent.document_loader import (
    prepare_documents as _prepare_documents,
    prepare_documents_from_files as _prepare_documents_from_files,
)

from retriever_agent.vector_store import create_vector_store

from retriever_agent.tools.query_rewriter import (
    create_query_rewriter_tool,
)
from retriever_agent.tools.semantic_search import (
    create_semantic_search_tool,
)
from retriever_agent.tools.keyword_search import (
    create_keyword_search_tool,
)
from retriever_agent.tools.metadata_filter import (
    create_metadata_filter_tool,
)
from retriever_agent.tools.reranker import reranker
from retriever_agent.tools.context_selector import context_selector


RETRIEVER_SYSTEM_PROMPT = """
You are a Retrieval Agent responsible for finding relevant evidence from the
available documents. Your job is to retrieve, rank, and select evidence. You
must not generate the final answer to the user's question.

The user message contains the latest question and may contain conversation
context. Use that context only to understand references in the latest question.

TOOLS:
- query_rewriter
- semantic_search
- keyword_search
- metadata_filter
- reranker
- context_selector

MANDATORY WORKFLOW — EXECUTE IN THIS ORDER:

1. QUERY REWRITING
   - Call query_rewriter first.
   - Pass the latest question as `query`.
   - Pass the conversation context from the user message as
     `conversation_context` (empty string if none).
   - Wait for its result and use the rewritten query below.

2. RETRIEVAL
   - Call semantic_search with the rewritten query and wait for its result.
   - Then call keyword_search with the rewritten query and wait for its result.
   - Do not run these two calls in parallel.

3. METADATA FILTERING
   - If the user explicitly identifies a document or page, call metadata_filter
     with that restriction. Otherwise do not call it.
   - Include any relevant evidence it returns without discarding search results.

4. RERANKING
   - Combine the semantic, keyword, and (if used) metadata evidence.
   - Remove exact duplicates when possible.
   - Call reranker with the rewritten query and the combined evidence encoded as
     a valid JSON string. Do not call it until retrieval has finished.
   - Wait for the actual reranker result.

5. CONTEXT SELECTION
   - Call context_selector only after reranker has returned.
   - Pass the exact reranker output as its `evidences` argument.
   - Never pass an empty string. If reranker returned no evidence, skip this
     tool and return an empty evidence bundle.
   - Wait for the context_selector result.

6. FINAL OUTPUT
   Return ONLY a valid JSON object with this structure:
   {
     "evidences": [
       {
         "document": "report.pdf",
         "page": 3,
         "score": 0.92,
         "content": "Retrieved text..."
       }
     ]
   }

RULES:
- Never invent evidence, document names, page numbers, scores, or content.
- Keep page as null when the source has no page number.
- Preserve original document names and scores returned by context_selector.
- If no relevant evidence is found, return exactly {"evidences": []}.
- Do not wrap JSON in Markdown fences or add explanations.
- Never call a dependent tool before its prerequisite result is available.
- Never call reranker in the same parallel batch as retrieval tools.
- Never call context_selector in the same parallel batch as reranker.
"""

def prepare_documents(
    file_path,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
):
    """Load, clean, and chunk one supported document."""

    return _prepare_documents(
        file_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def prepare_documents_from_files(
    file_paths,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
):
    """Load and chunk multiple PDF/DOCX/TXT files."""

    return _prepare_documents_from_files(
        file_paths,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def prepare_vector_store(file_path):
    """Prepare one document and create a Chroma vector store."""

    chunks = prepare_documents(file_path)

    return create_vector_store(chunks)


def ingest_files(file_paths, persist_directory=None):
    """Ingest multiple files and return (chunks, vector_store)."""

    chunks = prepare_documents_from_files(file_paths)

    if persist_directory is None:
        vector_store = create_vector_store(chunks)
    else:
        vector_store = create_vector_store(
            chunks,
            persist_directory=persist_directory,
        )

    return chunks, vector_store


def create_retriever_tools(chunks, vector_store, llm=None):
    """Build tools bound to the current document chunks and vector store."""

    return [
        create_query_rewriter_tool(llm=llm),
        create_semantic_search_tool(vector_store),
        create_keyword_search_tool(chunks),
        create_metadata_filter_tool(vector_store),
        reranker,
        context_selector,
    ]


def create_retriever_agent(chunks, vector_store, llm=None):
    """Create the LangChain Retriever Agent for the current corpus."""

    active_llm = llm if llm is not None else get_llm()

    tools = create_retriever_tools(
        chunks=chunks,
        vector_store=vector_store,
        llm=active_llm,
    )

    return create_agent(
        model=active_llm,
        tools=tools,
        system_prompt=RETRIEVER_SYSTEM_PROMPT,
    )
