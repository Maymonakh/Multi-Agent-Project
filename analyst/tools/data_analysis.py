"""Data Analysis: average, best, worst, difference and % improvement per metric."""
from analyst.schemas import Metric
from analyst.tools.calculator import average, difference, percent_change


def _point(m: Metric) -> dict:
    return {"document": m.document, "page": m.page, "value": m.value}


def analyze_metrics(metrics: list[Metric]) -> dict:
    by_name = {}
    for m in metrics:
        by_name.setdefault(m.name, []).append(m)

    results = {}
    for name, items in by_name.items():
        best = max(items, key=lambda m: m.value)
        worst = min(items, key=lambda m: m.value)
        results[name] = {
            "average": round(average([m.value for m in items]), 2),
            "best": _point(best),
            "worst": _point(worst),
            "difference": round(difference(best.value, worst.value), 2),
            "percent_improvement": round(percent_change(worst.value, best.value), 2),
            "count": len(items),
        }
    return results
