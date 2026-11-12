from typing import Optional
from pydantic import BaseModel
from shared.schemas import EvidenceBundle


class Metric(BaseModel):
    """One number found in the evidence, e.g. CNN accuracy = 92 in A.pdf page 1."""
    name: str
    document: str
    value: float
    page: Optional[int] = None   # needed for citations


class AnalystDecision(BaseModel):
    """What the LLM returns."""
    enough_evidence: bool
    missing_query: str = ""
    metrics: list[Metric] = []
    similarities: list[str] = []
    differences: list[str] = []
    reasoning: str = ""
    failed: bool = False         # set by our code when the LLM gave no usable answer


class AnalysisResult(BaseModel):
    """What the Analyst gives to the Answer Agent."""
    enough_evidence: bool
    reasoning: str
    error: str = ""
    tables: list[dict] = []
    comparison: dict = {}
    calculations: dict = {}
    evidence: EvidenceBundle
