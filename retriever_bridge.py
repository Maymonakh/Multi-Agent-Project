"""The ONLY place that touches Person A's code. Replace the two fake functions."""
from shared.schemas import Evidence, EvidenceBundle

USING_FAKE = True  # set to False when the real retriever is connected


def retrieve(query: str, chat_history: list | None = None) -> EvidenceBundle:
    # TODO (Person A's code), for example:
    #   from src.retriever.retriever_agent import ...
    return EvidenceBundle(evidences=[
        Evidence(document="A.pdf", page=1, score=0.94, content="CNN acc = 92%"),
        Evidence(document="B.pdf", page=2, score=0.91, content="CNN acc = 95%"),
    ])


def ingest(file_path: str) -> str:
    # TODO (Person A's code): load, clean, split, embed, store
    return f"(fake) {file_path} not ingested"
