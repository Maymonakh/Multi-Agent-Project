"""FastAPI routes for reports. Partner: app.include_router(report.router)"""
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from report.design import COLOR_KEYS, PRESETS, ROLES
from report.fonts import available_fonts
from report import service

router = APIRouter(prefix="/report", tags=["report"])


class ExtraSection(BaseModel):
    heading: str = ""
    text: str


class ReportRequest(BaseModel):
    """Send back what /ask returned (answer + analysis), plus the question."""
    question: str
    answer: str
    analysis: dict = Field(default_factory=dict)
    extra_sections: list[ExtraSection] = Field(default_factory=list)   # e.g. voice transcript, OCR text
    style: str = ""                                                    # free text, the agent follows it
    preset: Literal["professional", "classic", "modern"] = "professional"
    title: str = ""
    use_agent: bool | None = None


@router.post("", response_class=FileResponse)
def create_report(request: ReportRequest):
    """Returns the PDF file."""
    if not request.answer.strip():
        raise HTTPException(status_code=422, detail="answer is empty.")
    result = service.generate_report(
        request.question, request.answer, request.analysis,
        [e.model_dump() for e in request.extra_sections], request.style, request.preset,
        request.title, use_agent=request.use_agent)
    return FileResponse(result["path"], media_type="application/pdf", filename="report.pdf",
                        headers={"X-Agent-Used": str(result["agent_used"]).lower()})


@router.get("/options")
def report_options():
    return {"presets": list(PRESETS), "fonts": list(available_fonts()), "text_roles": ROLES, "color_names": COLOR_KEYS}
