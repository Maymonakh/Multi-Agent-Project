"""One function: text -> vector. Used only by the semantic cache.
Better: use the SAME embedding model as Person A (src/embeddings.py)."""
_model = None


def embed(text: str) -> list[float]:
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model.encode(text).tolist()
