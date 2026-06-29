"""Stage 5: judge a strategy's answers (or all strategies).

Reads results/<...>_rag.jsonl, writes parallel <...>_rag_eval.jsonl with
per-question scores. Resumable.

  python -m scripts.judge --strategy classic
  python -m scripts.judge --all
"""
import argparse
import json
import os

from tqdm import tqdm

from rag.config import cfg
from rag.llm import Judge
from rag.judge import evaluate_single_qa
from rag.strategies import available
from rag.io_utils import done_ids

PREFIX = "arxiv_2025_llama_8b_I_"


def judge_file(gen_file, judge):
    eval_file = gen_file.replace(".jsonl", "_eval.jsonl")
    already = done_ids(eval_file, key="edge")
    if not os.path.exists(gen_file):
        print(f"  skip (missing): {gen_file}")
        return
    with open(gen_file) as f:
        lines = [l for l in f if l.strip()]
    n = 0
    with open(eval_file, "a", encoding="utf-8") as out:
        for line in tqdm(lines, desc=os.path.basename(gen_file)):
            rec = json.loads(line)
            edge = str(rec.get("edge", "")).strip()
            if not edge or edge in already:
                continue
            q = rec.get("questions", {})
            a = rec.get("llm_answers", {})
            scored = {"edge": edge}
            if q.get("problem_query") and a.get("problem_answer"):
                scored["problem_evaluation"] = evaluate_single_qa(
                    q["problem_query"], a["problem_answer"], judge)
            if q.get("method_query") and a.get("method_answer"):
                scored["method_evaluation"] = evaluate_single_qa(
                    q["method_query"], a["method_answer"], judge)
            out.write(json.dumps(scored) + "\n")
            out.flush()
            n += 1
    print(f"  wrote {n} evaluations -> {eval_file}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=available())
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    judge = Judge()
    strategies = available() if args.all else [args.strategy]
    if not strategies or strategies == [None]:
        ap.error("Pass --strategy NAME or --all")
    for s in strategies:
        gen_file = os.path.join(cfg.paths.results_dir, f"{PREFIX}{s}_rag.jsonl")
        print(f"Judging {s}:")
        judge_file(gen_file, judge)


if __name__ == "__main__":
    main()
