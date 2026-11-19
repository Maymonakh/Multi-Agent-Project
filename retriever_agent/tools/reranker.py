import math
import re
from collections import defaultdict

from retriever_agent.config import RERANKER_MODEL

_models: dict = {}


def _load_model(name: str):
    """Load once per process. A failed load is remembered (no retry on every question)."""
    if name not in _models:
        try:
            from sentence_transformers import CrossEncoder

            _models[name] = CrossEncoder(name, device="cpu")
        except Exception as error:
            print(f"Reranker model unavailable, using fused ranking only: {error}")
            _models[name] = None
    return _models[name]


def _sigmoid(x: float) -> float:
    x = max(-30.0, min(30.0, float(x)))
    return 1.0 / (1.0 + math.exp(-x))


def _key(item: dict) -> tuple:
    content = re.sub(r"\s+", " ", str(item.get("content", ""))).strip().casefold()
    return item.get("document"), item.get("page"), content


def fuse(candidates: list[dict], rrf_k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion across retrieval sources + exact de-duplication."""
    by_source: dict[str, list[dict]] = defaultdict(list)
    for item in candidates:
        by_source[item.get("source", "?")].append(item)

    merged: dict[tuple, dict] = {}
    for items in by_source.values():
        for rank, item in enumerate(items, start=1):
            if not str(item.get("content", "")).strip():
                continue
            entry = merged.setdefault(_key(item), {
                "document": item.get("document", "Unknown"),
                "page": item.get("page"),
                "content": re.sub(r"[ \t]+", " ", str(item["content"])).strip(),
                "rrf": 0.0,
            })
            entry["rrf"] += 1.0 / (rrf_k + rank)

    return sorted(merged.values(), key=lambda e: e["rrf"], reverse=True)


def rerank(query: str, candidates: list[dict], top_k: int = 20, model_name: str | None = None) -> list[dict]:
    fused = fuse(candidates)[: max(1, top_k)]
    if not fused:
        return []

    model = _load_model(model_name or RERANKER_MODEL)
    scores = None
    if model is not None:
        try:
            scores = [_sigmoid(s) for s in model.predict([[query, item["content"]] for item in fused])]
        except Exception as error:
            print("Reranker prediction failed, using fused ranking:", error)

    if scores is None:
        best = fused[0]["rrf"] or 1.0
        scores = [item["rrf"] / best for item in fused]

    ranked = [
        {"document": item["document"], "page": item["page"], "score": float(score), "content": item["content"]}
        for item, score in zip(fused, scores)
    ]
    ranked.sort(key=lambda e: e["score"], reverse=True)
    return ranked
