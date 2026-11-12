import json
import re
from rank_bm25 import BM25Plus
from langchain_core.tools import tool


def tokenize(text: str):
    return re.findall(r"\w+", (text or "").casefold(), flags=re.UNICODE)


def create_keyword_search_tool(chunks):
    """Create BM25Plus search over the current ingested chunk collection."""

    tokenized_chunks = [tokenize(doc.page_content) for doc in chunks]
    bm25 = BM25Plus(tokenized_chunks) if tokenized_chunks else None

    @tool
    def keyword_search(query: str, k: int = 8) -> str:
        """Retrieve chunks by exact terms, names, identifiers, and phrases."""
        
        if bm25 is None or not tokenized_chunks:
            return json.dumps([], ensure_ascii=False)
        query_words = tokenize(query)
        if not query_words:
            return json.dumps([], ensure_ascii=False)
        scores = bm25.get_scores(query_words)
        indexes = scores.argsort()[::-1][:max(1, k)]
        evidences = []
        for index in indexes:
            score = float(scores[index])
            doc = chunks[int(index)]
            # BM25Plus has a positive baseline; require an actual lexical overlap.
            if score <= 0 or not set(query_words).intersection(tokenized_chunks[int(index)]):
                continue
            evidences.append({
                "document": doc.metadata.get("document", "Unknown"),
                "page": doc.metadata.get("page"),
                "score": score,
                "content": doc.page_content,
            })
        return json.dumps(evidences, ensure_ascii=False)
    return keyword_search
