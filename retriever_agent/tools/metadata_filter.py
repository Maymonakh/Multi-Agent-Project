import re

_PAGE = re.compile(r"(?:\bpage|\bp\.|\bصفحة|\bصفحه)\s*(\d{1,4})\b", re.IGNORECASE)


def _norm(text: str) -> str:
    return re.sub(r"[\s_\-]+", " ", text.casefold()).strip()


def detect_filters(query: str, known_documents) -> dict:
    """Return {"documents": [names] | None, "page": int | None}."""
    normalized_query = _norm(query or "")
    found = []

    for name in known_documents:
        full = _norm(name)
        stem = _norm(name.rsplit(".", 1)[0]) if "." in name else full
        for variant in {full, stem}:
            if len(variant) >= 3 and re.search(rf"(?<!\w){re.escape(variant)}(?!\w)", normalized_query):
                found.append(name)
                break

    page_match = _PAGE.search(query or "")
    return {
        "documents": found or None,
        "page": int(page_match.group(1)) if page_match else None,
    }


def has_filters(filters: dict | None) -> bool:
    return bool(filters) and bool(filters.get("documents") or filters.get("page") is not None)


def to_chroma_where(filters: dict | None) -> dict | None:
    if not has_filters(filters):
        return None
    conditions = []
    documents = filters.get("documents")
    if documents:
        conditions.append(
            {"document": documents[0]} if len(documents) == 1 else {"document": {"$in": list(documents)}}
        )
    if filters.get("page") is not None:
        conditions.append({"page": int(filters["page"])})
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


def matches(metadata: dict, filters: dict | None) -> bool:
    if not has_filters(filters):
        return True
    documents = filters.get("documents")
    if documents and metadata.get("document") not in documents:
        return False
    if filters.get("page") is not None and metadata.get("page") != filters["page"]:
        return False
    return True
