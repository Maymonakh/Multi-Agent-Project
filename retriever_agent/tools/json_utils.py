import json
import re


def parse_evidence_json(text: str) -> list:
    """
    Tolerant parser for LLM-produced evidence JSON.

    Handles: markdown fences, several JSON values glued together
    (e.g. '[...][...]' or '[...]\n{...}'), trailing prose, and
    {"evidences": [...]} wrappers. Returns one flat list of dicts.
    """
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I).strip()

    decoder = json.JSONDecoder()
    values, pos = [], 0

    while pos < len(text):
        # jump to the next JSON start
        match = re.search(r"[\[{]", text[pos:])
        if not match:
            break
        pos += match.start()
        try:
            value, end = decoder.raw_decode(text, pos)
        except json.JSONDecodeError:
            pos += 1  # skip stray bracket in prose
            continue
        values.append(value)
        pos = end

    if not values:
        raise ValueError("no valid JSON found in evidences input")

    flat = []
    for value in values:
        if isinstance(value, dict) and isinstance(value.get("evidences"), list):
            flat.extend(value["evidences"])
        elif isinstance(value, list):
            flat.extend(value)
        elif isinstance(value, dict):
            flat.append(value)
    return flat
