"""Source Formatter: unique list of the sources the answer really used."""
from shared.schemas import Evidence
from answer.citation_formatter import cite


def format_sources(numbers: list[int], evidences: list[Evidence]) -> list[str]:
    used = [evidences[n - 1] for n in numbers if 1 <= n <= len(evidences)]
    if not used:               # the model cited nothing -> show everything it was given
        used = evidences
    return list(dict.fromkeys(cite(e) for e in used))
