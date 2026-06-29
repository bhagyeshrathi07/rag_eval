"""Robust parsing helpers for LLM JSON output (ported from notebook)."""
import json


def safe_json_parse(raw_output: str, fallback: dict | None = None) -> dict:
    """Parse JSON from an LLM response that may include surrounding text."""
    if fallback is None:
        fallback = {}
    try:
        return json.loads(raw_output)
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        start = raw_output.find("{")
        end = raw_output.rfind("}") + 1
        if start == -1 or end == 0:
            return dict(fallback)
        return json.loads(raw_output[start:end])
    except Exception:
        return dict(fallback)


def normalize_doc_ids(retrieved_doc_id) -> list:
    if retrieved_doc_id is None:
        return []
    if isinstance(retrieved_doc_id, list):
        return retrieved_doc_id
    return [retrieved_doc_id]
