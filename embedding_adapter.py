"""Use the same embedding model as the Retriever's Chroma vector store."""


def embed(text: str) -> list[float]:
    from retriever_agent.vector_store import get_embeddings

    return get_embeddings().embed_query(text)
