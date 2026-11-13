import json
from analyst.schemas import AnalysisResult
from shared.llm_config import get_llm
from utils import format_evidence
from answer.citation_formatter import cited_numbers, format_citations
from answer.source_formatter import format_sources
from answer.response_formatter import format_response

PROMPT = """You are the Answer Agent. Write the final answer for the user.

Question: {question}

Evidence (cite with its number, like [1] or [1, 2]):
{evidence}

Analysis (already calculated. Use these numbers exactly):
{analysis}

Rules:
- Use ONLY the evidence and the analysis. No outside knowledge.
- Cite every fact with its evidence number.
- Use a short markdown table when you compare documents or models.
- {limit_note}
- Write the answer in the same language as the Question above.
  If the question is in English, answer in English only. Never use any other language."""


def run_answer(
    question: str,
    analysis: AnalysisResult,
    llm=None,
) -> tuple[str, bool]:
    """Return (answer_text, ok); ok=False means no valid answer was generated."""
    if analysis.error:
        return (
            "Sorry, I could not analyze the documents right now. Please try again.",
            False,
        )

    active_llm = llm if llm is not None else get_llm()
    evidences = analysis.evidence.evidences

    limit_note = (
        "Answer fully."
        if analysis.enough_evidence
        else (
            "The evidence is NOT enough. Say clearly what is missing and "
            "answer only the part the evidence supports."
        )
    )

    details = {
        "calculations": analysis.calculations,
        "comparison": analysis.comparison,
        "tables": analysis.tables,
        "reasoning": analysis.reasoning,
    }

    prompt = PROMPT.format(
        question=question,
        evidence=format_evidence(evidences),
        analysis=json.dumps(details, ensure_ascii=False, indent=1),
        limit_note=limit_note,
    )

    try:
        reply = active_llm.invoke(prompt)
        raw = reply.content if hasattr(reply, "content") else str(reply)
    except Exception as error:
        print("Answer LLM failed:", error)
        return "Sorry, I could not write the answer right now. Please try again.", False

    formatted_answer = format_citations(raw, evidences)
    sources = format_sources(cited_numbers(raw), evidences)
    return format_response(formatted_answer, sources), True
