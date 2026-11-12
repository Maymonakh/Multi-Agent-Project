
from langchain.agents import create_agent

from shared.llm_config import get_llm

from retriever_agent.tools.query_rewriter import query_rewriter

from retriever_agent.tools.semantic_search import (
    create_semantic_search_tool
)

from retriever_agent.tools.keyword_search import (
    create_keyword_search_tool
)

from retriever_agent.tools.metadata_filter import (
    create_metadata_filter_tool
)

from retriever_agent.tools.reranker import reranker

from retriever_agent.tools.context_selector import (
    context_selector
)


RETRIEVER_SYSTEM_PROMPT = """
You are a Retrieval Agent responsible for finding
relevant evidence from the available documents.

Your job is to retrieve, rank, and select evidence.
You must not generate the final answer to the user's question.

You have access to these tools:

1. query_rewriter
2. semantic_search
3. keyword_search
4. metadata_filter
5. reranker
6. context_selector


IMPORTANT: TOOL EXECUTION ORDER

Follow this workflow carefully.

STEP 1 - QUERY REWRITING

- Call query_rewriter first.
- Pass the latest user question as query.
- Pass relevant previous conversation messages
  as conversation_context.
- Wait for the tool result.
- Use the rewritten query in subsequent retrieval steps.

STEP 2 - RETRIEVAL

- Call semantic_search using the rewritten query.
- Wait for its result.
- Call keyword_search using the rewritten query.
- Wait for its result.
- Collect the evidence returned by both tools.

STEP 3 - METADATA FILTERING

- If the user explicitly specifies a document or page,
  use metadata_filter when appropriate.
- Do not apply unnecessary metadata restrictions.
- Preserve the evidence returned by the search tools.

STEP 4 - RERANKING

- Combine the evidence returned by the retrieval tools.
- Remove duplicate evidence when possible.
- Call reranker with:
  - query: the rewritten query
  - evidences: the combined evidence as a valid JSON string
- Wait for the reranker result.

STEP 5 - CONTEXT SELECTION

- Call context_selector only after receiving
  the actual output of reranker.
- Pass the exact reranker output as evidences.
- Never pass an empty string.
- Wait for the context_selector result.

STEP 6 - FINAL OUTPUT

Return only a valid JSON object matching this structure:

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

Rules:

- Do not invent evidence.
- Do not invent document names or page numbers.
- Keep page as null when the source has no page number.
- Preserve the original document names.
- Preserve the scores returned by the context selector.
- If no relevant evidence is found, return:
  {"evidences": []}
- Do not include Markdown fences around the JSON.
- Do not include explanations outside the JSON.

CRITICAL EXECUTION RULES

- Never call dependent tools before receiving the
  results of their prerequisite tools.
- Never call reranker in the same parallel tool-call
  batch as semantic_search or keyword_search.
- Never call context_selector in the same parallel
  tool-call batch as reranker.
- Never fabricate intermediate tool outputs.
- Never pass an empty evidences argument.
"""


def create_retriever_tools(chunks, vector_store):
    """Create all tools used by the Retriever Agent."""

    return [
        query_rewriter,
        create_semantic_search_tool(vector_store),
        create_keyword_search_tool(chunks),
        create_metadata_filter_tool(vector_store),
        reranker,
        context_selector
    ]


def create_retriever_agent(
    chunks,
    vector_store,
    llm=None
):
    """
    Create the Retriever Agent.

    An optional LLM can be injected for testing.
    If no LLM is provided, the configured OpenRouter model
    will be used.
    """

    tools = create_retriever_tools(
        chunks=chunks,
        vector_store=vector_store
    )

    return create_agent(
        model=llm if llm is not None else get_llm(),
        tools=tools,
        system_prompt=RETRIEVER_SYSTEM_PROMPT
    )
