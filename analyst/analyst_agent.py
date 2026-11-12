import re
from typing import Callable
from shared.schemas import Evidence, EvidenceBundle
from utils import format_evidence
from analyst.schemas import AnalystDecision, AnalysisResult, Metric
from analyst.tools.table_extractor import extract_tables
from analyst.tools.document_comparison import compare_documents
from analyst.tools.data_analysis import analyze_metrics
from analyst.tools.search_more import search_more

MAX_LOOPS = 2  # how many times the Analyst may ask the Retriever for more evidence

PROMPT = """You are the Analyst Agent. Use ONLY the evidence below.

Question: {question}

Evidence:
{evidence}

Tasks:
1. Decide if the evidence is enough to answer the question.
   If NOT enough: enough_evidence=false and write a short, specific search query in missing_query.
2. If the question needs numbers, list them in metrics (name, document, page, value).
   Copy document and page from the evidence line. Do NOT calculate anything yourself:
   Python will do the calculation later. If the needed numbers are in the evidence,
   enough_evidence must be true.
3. If there are 2 or more documents, write similarities and differences between them
   (short sentences, based only on the evidence).
4. Write a short reasoning."""

JSON_PROMPT = PROMPT + """

Answer with ONLY a JSON object, no other text, like this:
{{"enough_evidence": true, "missing_query": "", "reasoning": "...",
  "metrics": [{{"name": "CNN accuracy", "document": "A.pdf", "page": 1, "value": 92}}],
  "similarities": ["Both papers test CNN on the same task"],
  "differences": ["Paper B reports higher accuracy than paper A"]}}"""


def decide(llm, question: str, evidences: list[Evidence]) -> AnalystDecision:
    evidence_text = format_evidence(evidences)

    # Try 1: structured output (some free models return None or fail)
    try:
        result = llm.with_structured_output(AnalystDecision).invoke(
            PROMPT.format(question=question, evidence=evidence_text))
        if result is not None:
            return result
    except Exception as error:
        print("Structured output failed:", error)

    # Try 2: ask for plain JSON and parse it ourselves
    try:
        reply = llm.invoke(JSON_PROMPT.format(question=question, evidence=evidence_text))
        text = reply.content if hasattr(reply, "content") else str(reply)
        match = re.search(r"\{.*\}", text, re.DOTALL)
        return AnalystDecision.model_validate_json(match.group(0))
    except Exception as error:
        print("JSON fallback failed:", error)

    # Try 3: we do NOT know if the evidence is enough -> say "not enough"
    return AnalystDecision(enough_evidence=False, failed=True,
                           reasoning="The AI model did not return a usable analysis.")


def merge(old: list[Evidence], new: list[Evidence]) -> list[Evidence]:
    seen = {(e.document, e.page, e.content) for e in old}
    return old + [e for e in new if (e.document, e.page, e.content) not in seen]


def fill_pages(metrics: list[Metric], evidences: list[Evidence]) -> list[Metric]:
    """If the LLM forgot the page, find it in the evidence (needed for citations)."""
    for m in metrics:
        if m.page is None:
            number = f"{m.value:g}"
            for e in evidences:
                if e.document == m.document and number in e.content:
                    m.page = e.page
                    break
    return metrics


def run_analyst(question: str,
                bundle: EvidenceBundle,
                retrieve: Callable[[str], EvidenceBundle],
                llm=None,
                max_loops: int = MAX_LOOPS) -> AnalysisResult:
    if llm is None:
        from shared.llm_config import llm  # real LLM

    evidences = list(bundle.evidences)

    # Feedback loop: ask the Retriever for more until enough (or max loops)
    for loop in range(max_loops + 1):
        decision = decide(llm, question, evidences)
        if decision.failed or decision.enough_evidence or not decision.missing_query or loop == max_loops:
            break
        evidences = merge(evidences, search_more(decision.missing_query, retrieve))

    metrics = fill_pages(decision.metrics, evidences)

    # Tools (plain Python)
    return AnalysisResult(
        enough_evidence=decision.enough_evidence,
        reasoning=decision.reasoning,
        error=decision.reasoning if decision.failed else "",
        tables=extract_tables(evidences),
        comparison=compare_documents(evidences, metrics, decision.similarities, decision.differences),
        calculations=analyze_metrics(metrics),
        evidence=EvidenceBundle(evidences=evidences),
    )
