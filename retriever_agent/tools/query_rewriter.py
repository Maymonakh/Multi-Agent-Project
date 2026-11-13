from langchain_core.tools import tool

from shared.llm_config import get_llm


def rewrite_query(
    query: str,
    conversation_context: str = "",
    llm=None,
) -> str:
    """Rewrite a conversational question into a standalone search query."""

    if not query or not query.strip():
        return ""

    try:
        context_section = ""
        if conversation_context and conversation_context.strip():
            context_section = (
                "\n\nConversation context:\n"
                + conversation_context.strip()
            )

        prompt = (
            "Rewrite the user's latest question as a concise, "
            "standalone search query for document retrieval.\n\n"
            "Instructions:\n"
            "- Use the conversation context to resolve pronouns "
            "and references such as it, its, they, and them.\n"
            "- Preserve the original intent.\n"
            "- Preserve names, numbers, and quoted terms.\n"
            "- Do not answer the question.\n"
            "- Do not introduce unsupported information.\n"
            "- If the question is already clear, keep it mostly unchanged.\n"
            "- Return only the rewritten search query.\n"
            + context_section
            + "\n\nLatest question:\n"
            + query.strip()
        )

        active_llm = llm if llm is not None else get_llm()
        response = active_llm.invoke(prompt)
        content = getattr(response, "content", response)

        if isinstance(content, str) and content.strip():
            return content.strip()

    except Exception as error:
        print(f"Query rewriting failed: {error}")

    return query.strip()


def create_query_rewriter_tool(llm=None):
    """Create a query-rewriter tool using the supplied shared LLM instance."""

    @tool
    def query_rewriter(query: str, conversation_context: str = "") -> str:
        """Rewrite the user's question using conversation context when available."""
        return rewrite_query(
            query=query,
            conversation_context=conversation_context,
            llm=llm,
        )

    return query_rewriter


query_rewriter = create_query_rewriter_tool()
