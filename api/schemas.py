from typing import Literal, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000, examples=["What are the disadvantages of RAG?"])
    history: list[ChatMessage] = Field(default_factory=list, max_length=20,
                                       description="Previous turns (without the current question).")


class AskResponse(BaseModel):
    answer: str
    cached: bool = False
    analysis: dict = Field(default_factory=dict)
    elapsed_ms: int
    ocr_text: Optional[str] = Field(default=None, description="Text read from the image (image endpoints only).")


class IngestResponse(BaseModel):
    document: str
    message: str
    characters: Optional[int] = None
    ocr_text: Optional[str] = None


class OCRResponse(BaseModel):
    text: str
    characters: int
    engine: str


class DocumentsResponse(BaseModel):
    documents: list[str]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    documents: int
    ocr_engine: str
