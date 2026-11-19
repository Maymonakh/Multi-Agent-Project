"""Context Selector: a small, diverse, citation-ready set of chunks for the Analyst."""


def select_context(
    evidences: list[dict],
    top_k: int = 6,
    max_chars: int = 9000,
    max_per_page: int = 2,
) -> list[dict]:
    ranked = sorted(evidences, key=lambda item: float(item.get("score", 0.0)), reverse=True)

    selected, seen_content, page_counts, used_chars = [], set(), {}, 0

    for item in ranked:
        content = str(item.get("content", "")).strip()
        if not content:
            continue

        normalized = " ".join(content.lower().split())
        if normalized in seen_content:
            continue

        page_key = (item.get("document"), item.get("page"))
        if page_counts.get(page_key, 0) >= max_per_page:
            continue

        if selected and used_chars + len(content) > max_chars:
            continue

        selected.append({
            "document": item.get("document", "Unknown"),
            "page": item.get("page"),
            "score": float(item.get("score", 0.0)),
            "content": content,
        })
        seen_content.add(normalized)
        page_counts[page_key] = page_counts.get(page_key, 0) + 1
        used_chars += len(content)

        if len(selected) >= top_k:
            break

    return selected
