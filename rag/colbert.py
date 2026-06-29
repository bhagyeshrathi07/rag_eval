"""ColBERT late-interaction retrieval strategy.

ColBERT uses its OWN token-level embeddings (ColBERTv2), not SPECTER2, and its
own PLAID index over a separate document pool. Build the pool + index with
scripts/build_colbert.py first.

The pool is parameterized (config.colbert.pool_size): the paper used 60K
(10K gold + 50K distractors). Indexing on it lets you reproduce the paper, or
set pool_size to the full corpus to remove the pool-composition confound
(paper limitation iv).
"""
from __future__ import annotations

import json
import os

# --- Compatibility shim -----------------------------------------------------
# ColBERT 0.2.21 imports `AdamW` from transformers, which newer transformers
# (4.x) removed. AdamW is only used for TRAINING, which we never do (we only
# index + search a pretrained checkpoint). Alias torch's AdamW so the import
# succeeds without downgrading transformers (which SPECTER2 needs at 4.57).
import transformers as _t
import torch as _torch
if not hasattr(_t, "AdamW"):
    _t.AdamW = _torch.optim.AdamW
# ---------------------------------------------------------------------------

from . import prompts
from .parsing import normalize_doc_ids
from .config import cfg
from .strategies import register

_searcher = None
_pid_to_doc_id = None
_pid_to_text = None


def _index_paths():
    """Resolve where the index and pool data live, from config."""
    colbert_dir = cfg.paths.colbert_dir
    experiments_root = os.path.abspath(os.path.join(colbert_dir, "experiments"))
    return {
        "collection_tsv": os.path.join(colbert_dir, "collection.tsv"),
        "pid_map": os.path.join(colbert_dir, "pid_to_doc_id.json"),
        "experiments_root": experiments_root,
        "index_root": os.path.join(experiments_root, "arxiv", "indexes"),
        "index_name": cfg.colbert.index_name,
    }


def _load_searcher():
    """Lazily load the ColBERT PLAID searcher + pid mappings."""
    global _searcher, _pid_to_doc_id, _pid_to_text
    if _searcher is not None:
        return

    from colbert import Searcher
    from colbert.infra import ColBERTConfig, Run, RunConfig

    p = _index_paths()
    if not os.path.exists(p["pid_map"]):
        raise RuntimeError(
            f"ColBERT pool/index not found ({p['pid_map']}). "
            "Run: python -m scripts.build_colbert  first."
        )

    config = ColBERTConfig(
        nbits=cfg.colbert.nbits,
        doc_maxlen=cfg.colbert.doc_maxlen,
        query_maxlen=cfg.colbert.query_maxlen,
    )
    with Run().context(RunConfig(nranks=1, root=p["experiments_root"],
                                 experiment="arxiv")):
        _searcher = Searcher(index=p["index_name"],
                             index_root=p["index_root"], config=config)

    with open(p["pid_map"]) as f:
        _pid_to_doc_id = {int(k): v for k, v in json.load(f).items()}

    _pid_to_text = {}
    with open(p["collection_tsv"], encoding="utf-8") as f:
        for line in f:
            pid_str, text = line.strip().split("\t", 1)
            _pid_to_text[int(pid_str)] = text


def retrieve_context_colbert(query, top_k=None):
    """ColBERT MaxSim retrieval. Returns (context_string, doc_ids)."""
    top_k = top_k or cfg.retrieval.top_k
    _load_searcher()
    pids, ranks, scores = _searcher.search(query, k=top_k)

    parts, ids = [], []
    for pid, rank, score in zip(pids, ranks, scores):
        doc_id = _pid_to_doc_id.get(pid, f"unknown_pid_{pid}")
        text = _pid_to_text.get(pid, "")
        ids.append(doc_id)
        parts.append(f"Doc ID: {doc_id}\n{text}\nscore: {score}\n")
    return "\n".join(parts), ids


@register("colbert")
def colbert(query, generator, collection, config=cfg):
    """ColBERT strategy: retrieve via the PLAID index, answer from that context.
    `collection` (the Chroma handle) is ignored — ColBERT uses its own index."""
    context, ids = retrieve_context_colbert(query, config.retrieval.top_k)
    answer = generator.complete(prompts.answer_prompt(query, context))
    return {"answer": answer, "retrieved_ids": normalize_doc_ids(ids), "meta": {}}