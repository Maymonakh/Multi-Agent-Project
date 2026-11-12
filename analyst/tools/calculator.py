"""Calculator: Python does the math, not the LLM."""


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def difference(a: float, b: float) -> float:
    return a - b


def percent_change(old: float, new: float) -> float:
    return (new - old) / old * 100 if old else 0.0
