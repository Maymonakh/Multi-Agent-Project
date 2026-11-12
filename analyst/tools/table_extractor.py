"""Table Extractor: finds tables in text. Best effort, handles 3 styles:
   1) | a | b |      2) columns separated by tabs / 2+ spaces      3) "Model 92.1 90.3"
"""
import re
from shared.schemas import Evidence

NUMBER = r"-?\d+(?:\.\d+)?%?"
LABEL_THEN_NUMBERS = re.compile(rf"^(.*?\S)\s+((?:{NUMBER}\s+)*{NUMBER})$")


def split_row(line: str):
    """Return a list of cells, "SEPARATOR" for |---|---|, or None if not a row."""
    line = line.strip()
    if not line:
        return None
    if line.count("|") >= 2:
        cells = [c.strip() for c in line.strip("|").split("|")]
        return "SEPARATOR" if all(set(c) <= set("-: ") for c in cells) else cells
    cells = [c for c in re.split(r"\t|\s{2,}", line) if c.strip()]
    if len(cells) >= 2:
        return cells
    m = LABEL_THEN_NUMBERS.match(line)
    if m and len(m.group(2).split()) >= 2:
        return [m.group(1)] + m.group(2).split()
    return None


def extract_tables(evidences: list[Evidence]) -> list[dict]:
    tables = []
    for e in evidences:
        block, previous_line = [], ""

        def close_block():
            if len(block) >= 2 and any(re.search(r"\d", c) for row in block for c in row):
                tables.append({"document": e.document, "page": e.page, "rows": list(block)})

        for line in e.content.splitlines():
            row = split_row(line)
            if row == "SEPARATOR":
                continue
            if row and (not block or len(row) == len(block[-1])):
                if not block and previous_line.split() and len(previous_line.split()) == len(row):
                    block.append(previous_line.split())   # header line without numbers
                block.append(row)
            else:
                close_block()
                block.clear()
                if row:
                    block.append(row)
            previous_line = line
        close_block()
    return tables
