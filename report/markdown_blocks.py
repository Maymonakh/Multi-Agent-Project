"""Turn the Answer Agent's markdown text into simple blocks the PDF builder understands."""
import re
from xml.sax.saxutils import escape


def inline_markup(text: str) -> str:
    """**bold**, *italic*, `code` -> reportlab tags (text is escaped first)."""
    text = escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<![\*\w])\*(?!\s)(.+?)(?<!\s)\*(?![\*\w])", r"<i>\1</i>", text)
    text = re.sub(r"`(.+?)`", r"<font face='Courier'>\1</font>", text)
    return text


def plain_text(text: str) -> str:
    return re.sub(r"[`*]", "", text)


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def md_to_blocks(text: str) -> list[dict]:
    blocks, paragraph = [], []

    def flush():
        if paragraph:
            blocks.append({"kind": "paragraph", "text": " ".join(paragraph)})
            paragraph.clear()

    lines = (text or "").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line:
            flush()
            continue

        heading = re.match(r"^(#{1,6})\s+(.*)", line)
        bold_only = re.match(r"^\*\*(.+?)\*\*:?$", line)
        if heading:
            flush()
            blocks.append({"kind": "heading", "text": heading.group(2), "level": 2 if len(heading.group(1)) <= 2 else 3})
        elif bold_only:
            flush()
            blocks.append({"kind": "heading", "text": bold_only.group(1), "level": 3})
        elif line.count("|") >= 2:
            flush()
            rows = [line]
            while i < len(lines) and lines[i].count("|") >= 2:
                rows.append(lines[i].strip())
                i += 1
            parsed = [_cells(r) for r in rows if not all(set(c) <= set("-: ") for c in _cells(r))]
            if len(parsed) >= 1:
                blocks.append({"kind": "table", "headers": parsed[0], "rows": parsed[1:]})
        elif re.match(r"^[-*\u2022]\s+", line) or re.match(r"^\d+[.)]\s+", line):
            flush()
            numbered = bool(re.match(r"^\d+[.)]\s+", line))
            items = [re.sub(r"^([-*\u2022]|\d+[.)])\s+", "", line)]
            while i < len(lines) and re.match(r"^\s*([-*\u2022]|\d+[.)])\s+", lines[i]):
                items.append(re.sub(r"^\s*([-*\u2022]|\d+[.)])\s+", "", lines[i]).strip())
                i += 1
            blocks.append({"kind": "bullets", "items": items, "numbered": numbered})
        else:
            paragraph.append(line)
    flush()
    return blocks
