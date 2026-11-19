from retriever_agent.tools.metadata_filter import has_filters, to_chroma_where


def semantic_search(vector_store, query: str, k: int = 8, filters: dict | None = None) -> list[dict]:
    if not query or not query.strip():
        return []

    where = to_chroma_where(filters)
    results = vector_store.similarity_search_with_score(query, k=max(1, k), filter=where)

    if not results and has_filters(filters):
        results = vector_store.similarity_search_with_score(query, k=max(1, k))

    evidences = []
    for doc, distance in results:
        similarity = 1.0 / (1.0 + max(0.0, float(distance)))
        evidences.append({
            "document": doc.metadata.get("document", "Unknown"),
            "page": doc.metadata.get("page"),
            "score": similarity,
            "content": doc.page_content,
        })
    return evidences
