"""LLM-as-a-judge evaluation (ported from the notebook).

For each (question, answer) the judge first decides `is_answer`. If false, all
metric scores are forced to 0; otherwise each metric is scored 1-5 and clamped.
This gate is what prevents refusals from being scored as high-quality answers.
"""
from __future__ import annotations

import json

from . import prompts

SCORE_KEYS = ["accuracy_score", "completeness_score", "faithfulness_score",
              "relevance_score", "clarity_score", "overall_score"]


def evaluate_single_qa(question: str, answer: str, judge) -> dict:
    """Score one Q/A pair. `judge` is a rag.llm.Judge (or anything with
    .score(prompt) -> str). Returns the parsed, gated, clamped score dict."""
    if not question or not answer:
        return {"error": "Missing question or answer", "status": "skipped"}

    raw = judge.score(prompts.judge_prompt(question, answer), system="")
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        # try to salvage a JSON object embedded in extra text
        start, end = raw.find("{"), raw.rfind("}") + 1
        if start != -1 and end > 0:
            try:
                result = json.loads(raw[start:end])
            except Exception:
                return {"error": "Invalid JSON from judge", "raw_response": raw,
                        "status": "parse_failed"}
        else:
            return {"error": "Invalid JSON from judge", "raw_response": raw,
                    "status": "parse_failed"}

    # normalize is_answer to a real bool
    ia = result.get("is_answer")
    result["is_answer"] = (ia.strip().lower() == "true") if isinstance(ia, str) else bool(ia)

    if result["is_answer"] is False:
        for k in SCORE_KEYS:
            result[k] = 0
    else:
        for k in SCORE_KEYS:
            try:
                result[k] = max(1, min(5, int(result.get(k, 0))))
            except (TypeError, ValueError):
                result[k] = 0
    result["status"] = "success"
    return result
