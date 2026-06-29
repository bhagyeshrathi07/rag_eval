"""ColBERT late-interaction retrieval strategy.

ColBERT needs its own PLAID index over a ~60K-document pool (separate from the
Chroma collection), so it isn't built by build_index.py. Build the pool/index
with scripts/build_colbert.py first.

IMPORTANT: ColBERT on ARM64 / CUDA 13 (DGX Spark) may require source build and
config tweaks. This module is written to register the strategy but raises a
clear error until the ColBERT index is available, so the other five strategies
remain fully runnable without ColBERT installed.
"""
from __future__ import annotations

from . import prompts
from .parsing import normalize_doc_ids
from .config import cfg
from .strategies import register

_searcher = None


def _load_searcher():
    """Lazily load the ColBERT PLAID searcher. Ported hook point for the
    notebook's retrieve_context_colbert; fill in when the index exists."""
    global _searcher
    if _searcher is not None:
        return _searcher
    try:
        from colbert import Searcher
        from colbert.infra import ColBERTConfig, RunConfig, Run
    except ImportError as e:
        raise RuntimeError(
            "colbert not installed. Install colbert-ai and build the index "
            "(scripts/build_colbert.py) before using the colbert strategy."
        ) from e
    cfg_c = cfg.colbert
    with Run().context(RunConfig(nranks=1, experiment="arxiv")):
        config = ColBERTConfig(doc_maxlen=cfg_c.doc_maxlen,
                               query_maxlen=cfg_c.query_maxlen,
                               nbits=cfg_c.nbits)
        _searcher = Searcher(index=cfg_c.index_name, config=config)
    return _searcher


def retrieve_context_colbert(query, top_k=None):
    """Return (context_string, ids) using ColBERT MaxSim retrieval."""
    top_k = top_k or cfg.retrieval.top_k
    searcher = _load_searcher()
    results = searcher.search(query, k=top_k)  # (pids, ranks, scores)
    pids = results[0]
    # NOTE: map pids -> doc text + arXiv id via the pool's pid_to_doc_id map.
    # That mapping is produced by build_colbert.py; load and apply it here.
    raise NotImplementedError(
        "ColBERT pid->doc mapping not wired yet. Port from the notebook's "
        "ColBERT section (retrieve_context_colbert + pid_to_doc_id.json)."
    )


@register("colbert")
def colbert(query, generator, collection, config=cfg):
    context, ids = retrieve_context_colbert(query, config.retrieval.top_k)
    answer = generator.complete(prompts.answer_prompt(query, context))
    return {"answer": answer, "retrieved_ids": normalize_doc_ids(ids), "meta": {}}
