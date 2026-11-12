"""Search / Retrieve More Evidence: asks the Retriever again (feedback loop)."""
from typing import Callable
from shared.schemas import Evidence, EvidenceBundle


def search_more(missing_query: str, retrieve: Callable[[str], EvidenceBundle]) -> list[Evidence]:
    return retrieve(missing_query).evidences
