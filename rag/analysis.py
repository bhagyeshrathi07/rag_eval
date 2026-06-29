"""Aggregate judge eval files into per-strategy metrics.

Reads results/arxiv_2025_llama_8b_I_<strategy>_rag_eval.jsonl files and computes,
per strategy and per query type (problem/method):

  - answer_rate:        % of queries with is_answer = true
  - <metric>_cond:      mean over ATTEMPTED answers (zeros/refusals excluded)
                        -> matches the paper's tables
  - <metric>_uncond:    mean over ALL queries (refusals count as 0)
                        -> fair cross-strategy comparison

Both are reported so low-answer-rate strategies (e.g. ColBERT) can't look
artificially strong by only being scored on the questions they chose to answer.
"""
from __future__ import annotations

import glob
import json
import os

import pandas as pd

from .config import cfg

METRICS = ["accuracy_score", "completeness_score", "faithfulness_score",
           "relevance_score", "clarity_score", "overall_score"]
PREFIX = "arxiv_2025_llama_8b_I_"
SUFFIX = "_rag_eval.jsonl"


def discover_strategies(results_dir: str | None = None) -> list[str]:
    results_dir = results_dir or cfg.paths.results_dir
    files = glob.glob(os.path.join(results_dir, f"{PREFIX}*{SUFFIX}"))
    return sorted(os.path.basename(f)[len(PREFIX):-len(SUFFIX)] for f in files)


def _rows_for_strategy(strategy: str, results_dir: str):
    """Yield (query_type, is_answer, {metric: score}) for every evaluated query."""
    path = os.path.join(results_dir, f"{PREFIX}{strategy}{SUFFIX}")
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for qtype, key in (("problem", "problem_evaluation"),
                               ("method", "method_evaluation")):
                ev = rec.get(key)
                if not ev or ev.get("status") not in (None, "success"):
                    continue
                if "is_answer" not in ev:
                    continue
                yield qtype, bool(ev["is_answer"]), {m: ev.get(m, 0) for m in METRICS}


def aggregate(results_dir: str | None = None) -> pd.DataFrame:
    """Return a tidy DataFrame: one row per (strategy, query_type) with
    answer_rate plus conditional and unconditional means for each metric."""
    results_dir = results_dir or cfg.paths.results_dir
    records = []
    for strat in discover_strategies(results_dir):
        buckets = {"problem": [], "method": []}
        for qtype, is_ans, scores in _rows_for_strategy(strat, results_dir):
            buckets[qtype].append((is_ans, scores))
        for qtype, rows in buckets.items():
            if not rows:
                continue
            n = len(rows)
            n_ans = sum(1 for is_ans, _ in rows if is_ans)
            row = {"strategy": strat, "query_type": qtype,
                   "n": n, "n_answered": n_ans,
                   "answer_rate": 100.0 * n_ans / n if n else 0.0}
            for m in METRICS:
                all_scores = [s[m] for _, s in rows]
                ans_scores = [s[m] for is_ans, s in rows if is_ans]
                row[f"{m}_uncond"] = sum(all_scores) / n if n else 0.0
                row[f"{m}_cond"] = (sum(ans_scores) / len(ans_scores)
                                    if ans_scores else 0.0)
            records.append(row)
    return pd.DataFrame(records)


def overall_table(df: pd.DataFrame, kind: str = "cond") -> pd.DataFrame:
    """Pivot to a strategy x query_type table of overall_score (kind: cond/uncond)."""
    col = f"overall_score_{kind}"
    piv = df.pivot(index="strategy", columns="query_type", values=col)
    piv["mean"] = piv.mean(axis=1)
    return piv.round(2)


def answer_rate_table(df: pd.DataFrame) -> pd.DataFrame:
    return df.pivot(index="strategy", columns="query_type",
                    values="answer_rate").round(1)
