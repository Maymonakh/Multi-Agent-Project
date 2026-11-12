"""Citation Formatter: turns [1] or [1, 2] into (A.pdf, page 1)."""
import re
from shared.schemas import Evidence

CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def cite(e: Evidence) -> str:
    return f"{e.document}, page {e.page}" if e.page is not None else e.document


def cited_numbers(text: str) -> list[int]:
    numbers = []
    for group in CITATION.findall(text):
        numbers += [int(n) for n in group.split(",")]
    return numbers


def format_citations(text: str, evidences: list[Evidence]) -> str:
    def replace(match):
        parts = [int(n) for n in match.group(1).split(",")]
        if not all(1 <= n <= len(evidences) for n in parts):
            return match.group(0)
        return "(" + "; ".join(cite(evidences[n - 1]) for n in parts) + ")"
    return CITATION.sub(replace, text)
