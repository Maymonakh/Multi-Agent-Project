
from langchain_core.tools import tool
from shared.llm_config import get_llm


def rewrite_query(
    query: str,
    conversation_context: str = ""
) -> str:
    """
    Rewrite a conversational question into a standalone search query.
    Uses conversation context to resolve pronouns and references.
    """

    # Handle empty input before calling the LLM
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

        response = get_llm().invoke(prompt)
        content = response.content

        if isinstance(content, str) and content.strip():
            return content.strip()

    except Exception as e:
        print(f"Query rewriting failed: {e}")

    # Safe fallback
    return query.strip()


@tool
def query_rewriter(
    query: str,
    conversation_context: str = ""
) -> str:
    """
    Rewrite the user's question using conversation context when available.
    """

    return rewrite_query(
        query=query,
        conversation_context=conversation_context
    )
