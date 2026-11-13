
import json

from retriever_agent.tools.json_utils import parse_evidence_json
import re
from functools import lru_cache

from langchain_core.tools import tool


@lru_cache(maxsize=1)
def _load_model():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device="cpu"
    )


def parse_evidences(evidences):
    """Parse and validate evidence input."""

    if isinstance(evidences, str):
        evidences = evidences.strip()

        if not evidences:
            raise ValueError(
                "Reranker received empty evidences. "
                "Semantic Search and Keyword Search results may be missing."
            )

        try:
            evidences = parse_evidence_json(evidences)
        except ValueError as exc:
            raise ValueError(
                f"Reranker received invalid JSON: {exc}"
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


def rerank_evidences(
    query: str,
    evidences: list[dict],
    top_k: int = 20
):
    """
    Deduplicate and rerank candidate evidence.
    Score is a CrossEncoder logit when available.
    """

    unique = []
    seen = set()

    for item in evidences:
        content = re.sub(
            r"\s+",
            " ",
            str(item.get("content", ""))
        ).strip()

        key = (
            item.get("document"),
            item.get("page"),
            content.casefold()
        )

        if content and key not in seen:
            seen.add(key)

            unique.append({
                **item,
                "content": content
            })

    if not unique:
        return []

    try:
        scores = _load_model().predict(
            [
                [query, item["content"]]
                for item in unique
            ]
        )

        for item, score in zip(unique, scores):
            item["score"] = float(score)

    except Exception:
        # Offline fallback:
        # Preserve the first-stage similarity/BM25 score.
        for item in unique:
            item["score"] = float(
                item.get("score", 0.0)
            )

    unique.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return unique[:max(1, top_k)]


@tool
def reranker(
    query: str,
    evidences: str,
    top_k: int = 20
) -> str:
    """
    Rerank candidate evidence against the query
    using a CrossEncoder model.
    """

    parsed = parse_evidences(evidences)

    ranked = rerank_evidences(
        query=query,
        evidences=parsed,
        top_k=top_k
    )

    return json.dumps(
        ranked,
        ensure_ascii=False
    )
