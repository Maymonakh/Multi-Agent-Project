"""Response Formatter: answer text + Sources section."""


def format_response(answer_text: str, sources: list[str]) -> str:
    if not sources:
        return answer_text
    lines = "\n".join(f"- {s}" for s in sources)
    return f"{answer_text}\n\n**Sources**\n{lines}"
