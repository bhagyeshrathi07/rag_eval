"""Stage 4: run one retrieval strategy over all sampled papers.

Matches the notebook's data layout: one record per paper, with both the problem
and method query answered, written under "llm_answers" alongside the existing
"questions". Output: results/arxiv_2025_llama_8b_I_<strategy>_rag.jsonl

  python -m scripts.run_strategy --strategy classic
  python -m scripts.run_strategy --strategy fusion --collection arxiv_papers_5k

Resumable: skips papers already written to the strategy's results file.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", required=True, choices=available())
    ap.add_argument("--collection", default=cfg.embeddings.collection_name)
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

    n = 0
    for rec in tqdm(sample, desc=args.strategy):
        gold = rec.get("edge") or rec.get("id")
        if gold in already:
            continue
        questions = rec.get("questions", {})
        prob_q = questions.get("problem_query")
        meth_q = questions.get("method_query")
        if not prob_q and not meth_q:
            continue

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
