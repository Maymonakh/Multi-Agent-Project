from typing import Optional
from pydantic import BaseModel


class Evidence(BaseModel):
    document: str
    page: Optional[int] = None
    score: float
    content: str


class EvidenceBundle(BaseModel):
    evidences: list[Evidence]