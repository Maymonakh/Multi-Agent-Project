from typing import Optional
from pydantic import BaseModel, Field
from shared.schemas import EvidenceBundle


class Metric(BaseModel):
    """One number found in evidence, with its original source."""
    name: str
    document: str
    value: float
    page: Optional[int] = None   # needed for citations


class AnalystDecision(BaseModel):
    """Structured decision returned by the Analyst LLM."""
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
