"""Public function for the rest of the project: generate_report(...)."""
import datetime
import os
from pathlib import Path

from report.agent import run_report_agent
from report.builder import ReportBuilder


def generate_report(question: str, answer: str, analysis: dict | None = None,
                    extra_sections: list[dict] | None = None, style: str = "",
                    preset: str = "professional", title: str = "", output_path: str | None = None,
                    use_agent: bool | None = None, llm=None) -> dict:
    """Make a PDF report.
    - style: free text like "dark blue, Times font, big headings, landscape". The agent follows it.
    - use_agent: None = use the agent only when a style text is given; True/False to force it.
    - extra_sections: [{"heading": "Voice transcript", "text": "..."}, {"heading": "OCR text", ...}]
    Always returns a PDF, even if the agent fails (it falls back to the preset design)."""
    builder = ReportBuilder(question, answer, analysis, extra_sections, preset, title)

    log, agent_used = [], False
    if use_agent is None:
        use_agent = bool((style or "").strip())
    if use_agent:
        try:
            if llm is None:
                from shared.llm_config import get_llm
                llm = get_llm()
            log = run_report_agent(builder, style, llm)
            agent_used = bool(log)
        except Exception as error:
            log.append(f"Agent failed, used the '{preset}' design instead: {error}")

    if output_path is None:
        folder = Path(os.getenv("REPORTS_DIR", "reports"))
        output_path = str(folder / f"report_{datetime.datetime.now():%Y%m%d_%H%M%S}.pdf")
    path = builder.render(output_path)
    return {"path": path, "agent_used": agent_used, "log": log, "design": builder.design.model_dump()}
