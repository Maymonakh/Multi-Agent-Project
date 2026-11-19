from typing import Optional
import re

from pydantic import BaseModel, Field, field_validator
from shared.schemas import EvidenceBundle


class Metric(BaseModel):
    """One number found in evidence, with its original source."""
    name: str
    document: str
    value: float
    page: Optional[int] = None   # needed for citations


def _to_number(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip().replace(",", "").rstrip("%").strip()
        if re.fullmatch(r"-?\d+(?:\.\d+)?", text):
            return float(text)
    return None


class AnalystDecision(BaseModel):
    """Structured decision returned by the Analyst LLM."""

    @field_validator("metrics", mode="before")
    @classmethod
    def _keep_numeric_metrics(cls, items):
        kept = []
        for item in items or []:
            if not isinstance(item, dict):
                kept.append(item)
                continue
            number = _to_number(item.get("value"))
            if number is not None:
                kept.append({**item, "value": number})
        return kept

    enough_evidence: bool
    missing_query: str = ""
    metrics: list[Metric] = Field(default_factory=list)
    similarities: list[str] = Field(default_factory=list)
    differences: list[str] = Field(default_factory=list)
    reasoning: str = ""
    failed: bool = False         # set by our code when the LLM gave no usable answer


class AnalysisResult(BaseModel):
    """Analysis payload passed from Analyst to Answer Agent."""
    enough_evidence: bool
    reasoning: str
    error: str = ""
    tables: list[dict] = Field(default_factory=list)
    comparison: dict = Field(default_factory=dict)
    calculations: dict = Field(default_factory=dict)
    evidence: EvidenceBundle
