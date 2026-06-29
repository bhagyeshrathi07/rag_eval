"""Build the ColBERT document pool + PLAID index.

Pool = gold papers (the sampled eval set) + distractors sampled from the Chroma
collection. Size is controlled by --pool-size (default: config.colbert.pool_size).
The paper used 60K (10K gold + 50K distractors). Set --pool-size to 0 to use ALL
gold + ALL distractors from the collection (removes the pool-composition confound).

  python -m scripts.build_colbert --collection arxiv_papers_5k --pool-size 5000  # smoke test
  python -m scripts.build_colbert --collection arxiv_papers --pool-size 60000    # paper-faithful
"""
import argparse
import json
import os
import random

# Same AdamW shim as rag/colbert.py (needed before importing colbert here too).
import transformers as _t
import torch as _torch
if not hasattr(_t, "AdamW"):
    _t.AdamW = _torch.optim.AdamW

from rag.config import cfg
from rag.vectorstore import get_collection
from rag.io_utils import read_jsonl  # noqa: F401  (kept for parity)


def batched_get(collection, ids, batch_size=5000):
    out = {"ids": [], "documents": [], "metadatas": []}
    for i in range(0, len(ids), batch_size):
        r = collection.get(ids=ids[i:i + batch_size],
                           include=["documents", "metadatas"])
        out["ids"].extend(r["ids"])
        out["documents"].extend(r["documents"])
        out["metadatas"].extend(r["metadatas"])
    return out


def build_pool(collection, sample_path, pool_size, seed=42):
    random.seed(seed)
    with open(sample_path) as f:
        eval_data = json.load(f)

    # Gold docs: the sampled papers (use their own title+abstract text).
    gold = []
    for rec in eval_data:
        edge = rec.get("edge") or rec.get("id")
        title = rec.get("title") or rec.get("attrs", {}).get("title", "")
        abstract = rec.get("abstract") or rec.get("attrs", {}).get("abstract", "")
        if not edge or not abstract:
            continue
        gold.append({"doc_id": edge, "text": f"Title: {title}\nAbstract: {abstract}"})
    gold_ids = {d["doc_id"] for d in gold}
    print(f"Gold docs: {len(gold):,}")

    # Distractors: sample from the Chroma collection.
    all_ids = collection.get(include=[])["ids"]
    n_distractor = (pool_size - len(gold)) if pool_size and pool_size > 0 else len(all_ids)
    n_distractor = max(0, min(n_distractor, len(all_ids)))
    sample_ids = random.sample(all_ids, n_distractor) if n_distractor < len(all_ids) else all_ids
    raw = batched_get(collection, sample_ids)

    gold_sig = {d["text"][:500] for d in gold}
    distractors = []
    for i, did in enumerate(raw["ids"]):
        text = raw["documents"][i]
        if did in gold_ids or text[:500] in gold_sig:
            continue
        distractors.append({"doc_id": did, "text": text})
    print(f"Distractors after de-dup: {len(distractors):,}")

    all_docs = gold + distractors
    random.shuffle(all_docs)
    print(f"Total pool: {len(all_docs):,}")
    return all_docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default=cfg.embeddings.collection_name)
    ap.add_argument("--pool-size", type=int, default=cfg.colbert.pool_size,
                    help="Total pool (gold+distractors). 0 = use everything.")
    args = ap.parse_args()

    colbert_dir = cfg.paths.colbert_dir
    os.makedirs(colbert_dir, exist_ok=True)
    sample_path = os.path.join(cfg.paths.base_dir, cfg.paths.sample_json)

    collection = get_collection(args.collection)
    docs = build_pool(collection, sample_path, args.pool_size, cfg.data.seed)

    # Write the TSV + pid map.
    tsv = os.path.join(colbert_dir, "collection.tsv")
    pid_map = os.path.join(colbert_dir, "pid_to_doc_id.json")
    mapping = {}
    with open(tsv, "w", encoding="utf-8") as f:
        for pid, d in enumerate(docs):
            text = d["text"].replace("\t", " ").replace("\n", " ")
            f.write(f"{pid}\t{text}\n")
            mapping[pid] = d["doc_id"]
    with open(pid_map, "w") as f:
        json.dump(mapping, f)
    print(f"Wrote {tsv} and {pid_map}")

    # Build the PLAID index (this is where CUDA kernels JIT-compile).
    from huggingface_hub import snapshot_download
    from colbert import Indexer
    from colbert.infra import ColBERTConfig, Run, RunConfig

    ckpt = snapshot_download(repo_id=cfg.colbert.checkpoint,
                             cache_dir=os.environ.get("HF_HOME"))
    experiments_root = os.path.abspath(os.path.join(colbert_dir, "experiments"))
    config = ColBERTConfig(nbits=cfg.colbert.nbits, root=experiments_root,
                           doc_maxlen=cfg.colbert.doc_maxlen, nranks=1,
                           kmeans_niters=4)
    print("Building PLAID index (CUDA kernels compile on first run)...")
    with Run().context(RunConfig(nranks=1, root=experiments_root, experiment="arxiv")):
        indexer = Indexer(checkpoint=ckpt, config=config)
        indexer.index(name=cfg.colbert.index_name, collection=tsv, overwrite=True)
    print("Index build complete.")


if __name__ == "__main__":
    main()