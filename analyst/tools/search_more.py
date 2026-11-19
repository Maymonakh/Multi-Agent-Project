"""Search / Retrieve More Evidence.

In the LangGraph pipeline this tool is the feedback edge Analyst -> Retriever.
It only decides WHAT to ask next; the graph performs the retrieval.
"""


def next_query(missing_query: str, tried_queries: list[str]) -> str | None:
    """Return a new, specific query, or None if it is empty or was already tried
    (asking the Retriever the same thing again cannot add evidence)."""
    query = " ".join((missing_query or "").split())
    if not query:
        return None
    if query.casefold() in {" ".join(t.split()).casefold() for t in tried_queries}:
        return None
    return query
