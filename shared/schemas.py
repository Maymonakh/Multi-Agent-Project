from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class Evidence(BaseModel):
    """A source-grounded chunk passed from Retriever to Analyst."""

    model_config = ConfigDict(extra="ignore")

    document: str = Field(description="Original source filename")
    page: Optional[int] = Field(default=None, description="1-based PDF page; null when unavailable")
    score: float = Field(description="Final reranker relevance score")
    content: str = Field(min_length=1, description="Verbatim extracted source text")


class EvidenceBundle(BaseModel):

    evidences: list[Evidence] = Field(default_factory=list)

    def to_json(self) -> str:
        return self.model_dump_json(indent=2, exclude_none=False)
