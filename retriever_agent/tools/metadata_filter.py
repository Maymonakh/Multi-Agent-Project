import json
from langchain_core.tools import tool


def create_metadata_filter_tool(vector_store):
    """Create a metadata lookup tool. Score is 0 until the reranker scores it."""
    @tool
    def metadata_filter(document: str | None = None, page: int | None = None) -> str:
        """Find chunks restricted to an exact document filename and/or 1-based page."""
        filters = []
        if document:
            filters.append({"document": document})
        if page is not None:
            filters.append({"page": int(page)})
        if not filters:
            return json.dumps([], ensure_ascii=False)
        where = filters[0] if len(filters) == 1 else {"$and": filters}
        results = vector_store.get(where=where, include=["documents", "metadatas"])
        evidences = []
        for content, metadata in zip(results.get("documents") or [], results.get("metadatas") or []):
            if not content:
                continue
            evidences.append({
                "document": metadata.get("document", "Unknown"),
                "page": metadata.get("page"),
                "score": 0.0,
                "content": content,
            })
        return json.dumps(evidences, ensure_ascii=False)
    return metadata_filter
