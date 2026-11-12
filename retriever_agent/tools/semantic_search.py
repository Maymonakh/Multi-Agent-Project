
import json
from langchain_core.tools import tool


def create_semantic_search_tool(vector_store):
    """Create semantic search using Chroma distance scores."""

    @tool
    def semantic_search(query: str, k: int = 8) -> str:
        """Retrieve chunks by semantic meaning."""

        # Get raw distance scores instead of assuming relevance scores
        results = vector_store.similarity_search_with_score(
            query,
            k=max(1, k)
        )

        evidences = []

        for doc, distance in results:

            distance = max(0.0, float(distance))

            # Convert distance into a bounded similarity indicator.
            similarity_score = 1.0 / (1.0 + distance)

            evidences.append({
                "document": doc.metadata.get(
                    "document", "Unknown"
                ),
                "page": doc.metadata.get("page"),
                "score": similarity_score,
                "content": doc.page_content,
            })

        return json.dumps(
            evidences,
            ensure_ascii=False
        )

    return semantic_search
