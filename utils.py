from shared.schemas import Evidence


def format_evidence(evidences: list[Evidence]) -> str:
    """Numbered evidence. The Analyst and the Answer Agent use the SAME numbers."""
    return "\n".join(f"[{i}] {e.document}, page {e.page}: {e.content}"
                     for i, e in enumerate(evidences, 1))
