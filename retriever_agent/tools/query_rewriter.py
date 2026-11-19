from shared.llm_config import get_llm


def rewrite_query(query: str, conversation_context: str = "", llm=None) -> str:
    """Return a standalone query. Without conversation context no LLM call is made."""
    query = (query or "").strip()
    if not query or not (conversation_context or "").strip():
        return query

    prompt = (
        "Rewrite the user's latest question as a concise, standalone search query "
        "for document retrieval.\n\n"
        "Instructions:\n"
        "- Use the conversation context to resolve pronouns and references "
        "(it, its, they, them, the previous paper...).\n"
        "- Preserve the original intent, language, names, numbers and quoted terms.\n"
        "- Do not answer the question and do not add unsupported information.\n"
        "- If the question is already clear, keep it unchanged.\n"
        "- Return only the rewritten query on a single line.\n\n"
        f"Conversation context:\n{conversation_context.strip()}\n\n"
        f"Latest question:\n{query}"
    )

    try:
        active_llm = llm if llm is not None else get_llm()
        response = active_llm.invoke(prompt)
        content = getattr(response, "content", response)
        if isinstance(content, str):
            lines = [line.strip().strip('"').strip() for line in content.splitlines() if line.strip()]
            if lines and len(lines[0]) <= max(300, 4 * len(query)):
                return lines[0]
    except Exception as error:
        print(f"Query rewriting failed, using the original question: {error}")

    return query
