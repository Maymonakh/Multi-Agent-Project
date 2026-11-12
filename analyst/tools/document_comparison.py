"""Document Comparison: numbers side by side + similarities and differences."""
from shared.schemas import Evidence
from analyst.schemas import Metric


def compare_documents(evidences: list[Evidence], metrics: list[Metric],
                      similarities: list[str], differences: list[str]) -> dict:
    documents = list(dict.fromkeys(e.document for e in evidences))

    # numbers side by side: {document: {metric: {value, page}}}
    values = {}
    names = {}
    for m in metrics:
        values.setdefault(m.document, {})[m.name] = {"value": m.value, "page": m.page}
        names.setdefault(m.name, set()).add(m.document)

    return {
        "documents": documents,
        "values_by_document": values,
        "compared_metrics": [n for n, docs in names.items() if len(docs) > 1],
        "similarities": similarities,   # written by the LLM from the evidence
        "differences": differences,
    }
