import json
from shared.schemas import Evidence, EvidenceBundle
from utils import format_evidence
from analyst.schemas import AnalystDecision, AnalysisResult, Metric
from analyst.tools.table_extractor import extract_tables
from analyst.tools.document_comparison import compare_documents
from analyst.tools.data_analysis import analyze_metrics

MAX_LOOPS = 2  # how many times the Analyst may ask the Retriever for more evidence (used by the graph)

PROMPT = """You are the Analyst Agent. Use ONLY the evidence below.

Question: {question}

Evidence:
{evidence}

Tasks:
1. Decide if the evidence is enough to answer the question.
   If NOT enough: enough_evidence=false and write a short, specific search query in missing_query.
2. If the question needs numbers, list them in metrics (name, document, page, value).
   value MUST be a plain number (92, 0.91). Never put text or formulas such as "O(n^2)" in
   metrics: non-numeric facts (complexity, methods, ...) go into similarities / differences.
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
        start = text.find("{")
        if start < 0:
            raise ValueError("The model did not return a JSON object.")
        payload, _ = json.JSONDecoder().raw_decode(text[start:])
        return AnalystDecision.model_validate(payload)
    except Exception as error:
        print("JSON fallback failed:", error)

    # Try 3: we do NOT know if the evidence is enough -> say "not enough"
    return AnalystDecision(enough_evidence=False, failed=True,
                           reasoning="The AI model did not return a usable analysis.")


def merge(old: list[Evidence], new: list[Evidence]) -> list[Evidence]:
    seen = {(e.document, e.page, e.content) for e in old}
    return old + [
        evidence for evidence in new
        if (evidence.document, evidence.page, evidence.content) not in seen
    ]


def fill_pages(metrics: list[Metric], evidences: list[Evidence]) -> list[Metric]:
    """Fill a missing citation page only when the value is found in that source."""
    for metric in metrics:
        if metric.page is None:
            number = f"{metric.value:g}"
            for evidence in evidences:
                if (
                    evidence.document == metric.document
                    and number in evidence.content
                ):
                    metric.page = evidence.page
                    break
    return metrics


def finalize_analysis(
    decision: AnalystDecision,
    evidences: list[Evidence],
) -> AnalysisResult:
    """Run the deterministic Analyst tools (tables, comparison, calculations) once the
    evidence loop has finished."""
    metrics = fill_pages(decision.metrics, evidences)

    return AnalysisResult(
        enough_evidence=decision.enough_evidence,
        reasoning=decision.reasoning,
        error=decision.reasoning if decision.failed else "",
        tables=extract_tables(evidences),
        comparison=compare_documents(evidences, metrics, decision.similarities, decision.differences),
        calculations=analyze_metrics(metrics),
        evidence=EvidenceBundle(evidences=evidences),
    )
