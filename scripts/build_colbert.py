"""Build the ColBERT document pool + PLAID index.

Pool = all gold papers (the sampled eval set) + distractors sampled from the
Chroma collection (excluding gold). --pool-size is the TOTAL target pool size:

  distractors = max(0, pool_size - len(gold))     # if pool_size > len(gold)
  pool_size = 0   -> use ALL gold + ALL collection docs (full-corpus, no confound)

The paper used pool_size=60000 (10k gold + 50k distractors). Set 0 to remove the
pool-composition confound (paper limitation iv).

  python -m scripts.build_colbert --collection arxiv_papers_5k --pool-size 0       # smoke
  python -m scripts.build_colbert --collection arxiv_papers --pool-size 60000      # paper-faithful
  python -m scripts.build_colbert --collection arxiv_papers --pool-size 0          # full corpus
"""
import argparse
import json
import os
import random

# AdamW shim (ColBERT 0.2.21 imports it; transformers 4.x removed it).
import transformers as _t
import torch as _torch
if not hasattr(_t, "AdamW"):
    _t.AdamW = _torch.optim.AdamW
os.environ.setdefault("COLBERT_LOAD_TORCH_EXTENSION_VERBOSE", "False")

from rag.config import cfg
from rag.vectorstore import get_collection


def batched_get(collection, ids, batch_size=5000):
    out = {"ids": [], "documents": []}
    for i in range(0, len(ids), batch_size):
        r = collection.get(ids=ids[i:i + batch_size], include=["documents"])
        out["ids"].extend(r["ids"])
        out["documents"].extend(r["documents"])
    return out


def build_pool(collection, sample_path, pool_size, seed=42):
    random.seed(seed)
    with open(sample_path) as f:
        eval_data = json.load(f)

    # --- Gold: the sampled papers (always included) ---
    gold, gold_ids = [], set()
    for rec in eval_data:
        edge = rec.get("edge") or rec.get("id")
        title = rec.get("title") or rec.get("attrs", {}).get("title", "")
        abstract = rec.get("abstract") or rec.get("attrs", {}).get("abstract", "")
        if not edge or not abstract:
            continue
        gold.append({"doc_id": edge, "text": f"Title: {title}\nAbstract: {abstract}"})
        gold_ids.add(edge)
    print(f"Gold docs: {len(gold):,}")

    # --- Distractors: from the collection, EXCLUDING gold ids ---
    all_ids = collection.get(include=[])["ids"]
    distractor_candidates = [i for i in all_ids if i not in gold_ids]
    print(f"Collection size: {len(all_ids):,}  (non-gold candidates: {len(distractor_candidates):,})")

    if pool_size and pool_size > 0:
        n_distractor = max(0, pool_size - len(gold))
    else:
        n_distractor = len(distractor_candidates)   # use everything
    n_distractor = min(n_distractor, len(distractor_candidates))

    if n_distractor < len(distractor_candidates):
        chosen = random.sample(distractor_candidates, n_distractor)
    else:
        chosen = distractor_candidates
    print(f"Distractors requested: {n_distractor:,}")

    raw = batched_get(collection, chosen) if chosen else {"ids": [], "documents": []}
    distractors = [{"doc_id": did, "text": raw["documents"][i]}
                   for i, did in enumerate(raw["ids"])]
    print(f"Distractors retrieved: {len(distractors):,}")

    all_docs = gold + distractors
    random.shuffle(all_docs)
    print(f"Total pool: {len(all_docs):,} ({len(gold):,} gold + {len(distractors):,} distractors)")
    if len(distractors) == 0 and (pool_size == 0 or pool_size > len(gold)):
        print("WARNING: 0 distractors. Is the collection built and non-gold? "
              "(For a real eval you want distractors competing with gold.)")
    return all_docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default=cfg.embeddings.collection_name)
    ap.add_argument("--pool-size", type=int, default=cfg.colbert.pool_size,
                    help="Total pool (gold+distractors). 0 = gold + ALL non-gold.")
    args = ap.parse_args()

    colbert_dir = cfg.paths.colbert_dir
    os.makedirs(colbert_dir, exist_ok=True)
    sample_path = os.path.join(cfg.paths.base_dir, cfg.paths.sample_json)

    collection = get_collection(args.collection)
    docs = build_pool(collection, sample_path, args.pool_size, cfg.data.seed)

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