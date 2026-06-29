"""Stage 4: run one retrieval strategy over the sampled papers that have questions.

Matches the notebook's data layout: one record per paper, with both the problem
and method query answered, written under "llm_answers". Output:
results/arxiv_2025_llama_8b_I_<strategy>_rag.jsonl

  python -m scripts.run_strategy --strategy classic
  python -m scripts.run_strategy --strategy fusion --collection arxiv_papers_5k
  python -m scripts.run_strategy --strategy query_rephrased --limit 20   # smoke test

Only papers that already have generated questions are processed; the progress
bar reflects that count (not the full sample), so the time estimate is honest.
Resumable: skips papers already written to the strategy's results file.
Use --limit N to process only the first N (after skipping done ones) for testing.
"""
import argparse
import json
import os

from tqdm import tqdm

from rag.config import cfg
from rag.llm import Generator
from rag.vectorstore import get_collection
from rag.strategies import get_strategy, available
from rag.io_utils import append_jsonl, done_ids


def has_question(rec):
    q = rec.get("questions") or {}
    return bool(q.get("problem_query") or q.get("method_query"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", required=True, choices=available())
    ap.add_argument("--collection", default=cfg.embeddings.collection_name)
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only the first N papers-with-questions (testing).")
    args = ap.parse_args()

    strat = get_strategy(args.strategy)
    generator = Generator()
    collection = get_collection(args.collection)

    sample_path = os.path.join(cfg.paths.base_dir, cfg.paths.sample_json)
    with open(sample_path) as f:
        sample = json.load(f)

    out_path = os.path.join(cfg.paths.results_dir,
                            f"arxiv_2025_llama_8b_I_{args.strategy}_rag.jsonl")
    already = done_ids(out_path, key="edge")

    # Build the actual worklist UP FRONT: papers with questions, not yet done.
    worklist = []
    for rec in sample:
        gold = rec.get("edge") or rec.get("id")
        if gold in already or not has_question(rec):
            continue
        worklist.append((gold, rec))
    if args.limit is not None:
        worklist = worklist[:args.limit]

    print(f"{len(already)} already done; {len(worklist)} papers to process"
          f"{f' (limited to {args.limit})' if args.limit else ''}.")

    n = 0
    for gold, rec in tqdm(worklist, desc=args.strategy):
        questions = rec.get("questions", {})
        prob_q = questions.get("problem_query")
        meth_q = questions.get("method_query")

        llm_answers, retrieval = {}, {}
        if prob_q:
            r = strat(prob_q, generator, collection, cfg)
            llm_answers["problem_answer"] = r["answer"]
            retrieval["problem"] = {"retrieved_ids": r["retrieved_ids"],
                                    "gold_retrieved": gold in r["retrieved_ids"],
                                    "meta": r["meta"]}
        if meth_q:
            r = strat(meth_q, generator, collection, cfg)
            llm_answers["method_answer"] = r["answer"]
            retrieval["method"] = {"retrieved_ids": r["retrieved_ids"],
                                   "gold_retrieved": gold in r["retrieved_ids"],
                                   "meta": r["meta"]}

        append_jsonl(out_path, {"edge": gold, "questions": questions,
                                "llm_answers": llm_answers, "retrieval": retrieval})
        n += 1
    print(f"Wrote {n} new records to {out_path}")


if __name__ == "__main__":
    main()