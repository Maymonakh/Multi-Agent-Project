
import json

from langchain_core.tools import tool


def parse_evidences(evidences):
    """Parse and validate evidence input."""

    if isinstance(evidences, str):
        evidences = evidences.strip()

        if not evidences:
            raise ValueError(
                "Context Selector received empty evidences. "
                "Check whether the Reranker returned its results."
            )

        try:
            evidences = json.loads(evidences)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Context Selector received invalid JSON: {exc}"
            ) from exc

    if not isinstance(evidences, list):
        raise TypeError(
            "Evidences must be a list of dictionaries."
        )

    for index, item in enumerate(evidences):
        if not isinstance(item, dict):
            raise TypeError(
                f"Evidence at index {index} must be a dictionary."
            )

    return evidences


def select_context(
    evidences: list[dict],
    top_k: int = 6,
    max_chars: int = 9000,
    max_per_page: int = 2
):
    """
    Select relevant, non-redundant evidence
    while retaining source coverage.
    """

    ranked = sorted(
        evidences,
        key=lambda item: float(
            item.get("score", 0.0)
        ),
        reverse=True
    )

    selected = []
    seen_content = set()
    page_counts = {}

    used_chars = 0

    for item in ranked:
        content = str(
            item.get("content", "")
        ).strip()

        if not content:
            continue

        normalized = " ".join(
            content.lower().split()
        )

        if normalized in seen_content:
            continue

        page_key = (
            item.get("document"),
            item.get("page")
        )

        if page_counts.get(page_key, 0) >= max_per_page:
            continue

        cost = len(content)

        if selected and used_chars + cost > max_chars:
            continue

        selected.append({
            "document": item.get("document", "Unknown"),
            "page": item.get("page"),
            "score": float(
                item.get("score", 0.0)
            ),
            "content": content
        })

        seen_content.add(normalized)

        page_counts[page_key] = (
            page_counts.get(page_key, 0) + 1
        )

        used_chars += cost

        if len(selected) >= top_k:
            break

    return selected


@tool
def context_selector(
    evidences: str,
    top_k: int = 6,
    max_chars: int = 9000
) -> str:
    """
    Choose a compact set of relevant,
    diverse evidence chunks for the Analyst.
    """

    parsed = parse_evidences(evidences)

    selected = select_context(
        evidences=parsed,
        top_k=top_k,
        max_chars=max_chars
    )

    return json.dumps(
        selected,
        ensure_ascii=False
    )
