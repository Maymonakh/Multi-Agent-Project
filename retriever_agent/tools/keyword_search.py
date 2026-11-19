import math
import re
from collections import Counter, defaultdict

from retriever_agent.tools.metadata_filter import has_filters, matches


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", (text or "").casefold(), flags=re.UNICODE)


class BM25Index:
    """Small Okapi-BM25 implementation with an inverted index and metadata filtering."""

    def __init__(self, chunks, k1: float = 1.5, b: float = 0.75):
        self.chunks = list(chunks)
        self.k1, self.b = k1, b
        self._lengths: list[int] = []
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)

        for index, chunk in enumerate(self.chunks):
            tokens = tokenize(chunk.page_content)
            self._lengths.append(len(tokens))
            for term, count in Counter(tokens).items():
                self._postings[term].append((index, count))

        size = len(self.chunks)
        self._avg_length = (sum(self._lengths) / size) if size else 0.0
        self._idf = {
            term: math.log(1 + (size - len(posting) + 0.5) / (len(posting) + 0.5))
            for term, posting in self._postings.items()
        }

    def search(self, query: str, k: int = 8, filters: dict | None = None) -> list[dict]:
        scores: dict[int, float] = defaultdict(float)

        for term in set(tokenize(query)):
            idf = self._idf.get(term)
            if idf is None:
                continue
            for index, tf in self._postings[term]:
                norm = 1 - self.b + self.b * self._lengths[index] / (self._avg_length or 1.0)
                scores[index] += idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)

        def build(allowed):
            out = []
            for index, score in ranked:
                chunk = self.chunks[index]
                if not allowed(chunk.metadata):
                    continue
                out.append({
                    "document": chunk.metadata.get("document", "Unknown"),
                    "page": chunk.metadata.get("page"),
                    "score": float(score),
                    "content": chunk.page_content,
                })
                if len(out) >= max(1, k):
                    break
            return out

        results = build(lambda metadata: matches(metadata, filters))
        if not results and has_filters(filters):   
            results = build(lambda metadata: True)
        return results
