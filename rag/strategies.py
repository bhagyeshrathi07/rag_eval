"""The six retrieval strategies behind one interface.

Each strategy has the signature:
    strategy(query, generator, collection, config) -> dict
        returns {"answer": str, "retrieved_ids": list[str], "meta": {...}}

In the notebook these were six sections that redefined shared function names;
here they are one registry. Add a strategy with @register("name").
ColBERT lives in rag/colbert.py and registers itself when that module imports.
"""
from __future__ import annotations

from . import prompts
from .vectorstore import retrieve
from .fusion import (reciprocal_rank_fusion, format_fused_docs_as_context,
                     filter_context_by_selected_ids)
from .parsing import safe_json_parse, normalize_doc_ids
from .config import cfg

_REGISTRY: dict = {}


def register(name):
    def deco(fn):
        _REGISTRY[name] = fn
        return fn
    return deco


def get_strategy(name):
    if name not in _REGISTRY:
        raise KeyError(f"Unknown strategy '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def available():
    return sorted(_REGISTRY)


# --------------------------------------------------------------------------- #
@register("classic")
def classic(query, generator, collection, config=cfg):
    """Embed query, retrieve top-k, answer directly."""
    context, ids, _ = retrieve(collection, query, config.retrieval.top_k)
    answer = generator.complete(prompts.answer_prompt(query, context))
    return {"answer": answer, "retrieved_ids": normalize_doc_ids(ids), "meta": {}}


@register("query_rephrased")
def query_rephrased(query, generator, collection, config=cfg):
    """Rewrite the query into an academic search query, then retrieve + answer."""
    rewritten = generator.complete(prompts.rewrite_prompt(query),
                                    desired_output_tokens=128).strip()
    context, ids, _ = retrieve(collection, rewritten, config.retrieval.top_k)
    answer = generator.complete(prompts.answer_prompt(query, context))
    return {"answer": answer, "retrieved_ids": normalize_doc_ids(ids),
            "meta": {"rewritten_query": rewritten}}


@register("query_rephrased_and_reranked")
def rephrased_reranked(query, generator, collection, config=cfg):
    """Rewrite -> retrieve a larger candidate set -> LLM re-ranks -> top-k -> answer.
    (Best strategy in the paper.)"""
    final_top_k = config.retrieval.top_k
    n_candidates = config.retrieval.rerank_candidates

    rewritten = generator.complete(prompts.rewrite_prompt(query),
                                    desired_output_tokens=128).strip()
    context, ids, _ = retrieve(collection, rewritten, n_candidates)
    retrieved_ids = normalize_doc_ids(ids)

    rerank_raw = generator.complete(
        prompts.rerank_prompt(query, rewritten, context, final_top_k),
        desired_output_tokens=512)
    parsed = safe_json_parse(rerank_raw,
                             {"ranked_papers": [], "selected_paper_ids": []})

    selected = parsed.get("selected_paper_ids", [])
    if not selected:
        selected = [it.get("paper_id") for it in parsed.get("ranked_papers", [])[:final_top_k]
                    if it.get("paper_id") is not None]
    selected = [d for d in selected if d is not None][:final_top_k]
    if not selected:
        selected = retrieved_ids[:final_top_k]   # fallback

    final_context = filter_context_by_selected_ids(context, selected) or context
    answer = generator.complete(prompts.answer_prompt(query, final_context))
    return {"answer": answer, "retrieved_ids": selected,
            "meta": {"rewritten_query": rewritten, "candidates": retrieved_ids}}


@register("fusion")
def fusion(query, generator, collection, config=cfg):
    """Generate N sub-queries -> search each -> merge with RRF -> top-k -> answer."""
    num_queries = config.retrieval.fusion_num_queries
    rrf_k = config.retrieval.rrf_k
    final_top_k = config.retrieval.top_k
    per_query_k = 5

    raw = generator.complete(prompts.fusion_query_prompt(query, num_queries),
                             desired_output_tokens=256)
    parsed = safe_json_parse(raw, {"queries": []})
    search_queries = [q.strip() for q in parsed.get("queries", [])
                      if isinstance(q, str) and q.strip()][:num_queries]
    if not search_queries:
        search_queries = [query]
    if len(search_queries) < num_queries and query not in search_queries:
        search_queries.append(query)
    search_queries = search_queries[:num_queries]

    results_by_query = []
    for sq in search_queries:
        context, ids, docs = retrieve(collection, sq, per_query_k)
        results_by_query.append({"search_query": sq, "docs": docs})

    selected_docs, _ = reciprocal_rank_fusion(results_by_query, final_top_k, rrf_k)
    selected_ids = [d["doc_id"] for d in selected_docs]
    final_context = format_fused_docs_as_context(selected_docs)
    answer = generator.complete(prompts.answer_prompt(query, final_context))
    return {"answer": answer, "retrieved_ids": selected_ids,
            "meta": {"sub_queries": search_queries}}


@register("tool_call")
def tool_call(query, generator, collection, config=cfg):
    """Agentic retrieval: the generator decides whether to retrieve via a tool.

    Simplified, deterministic version of the notebook's tool loop: offer
    retrieval; if the model emits a RETRIEVE directive, run retrieval and answer
    with context; otherwise answer directly. Kept provider-agnostic (no
    function-calling API) so behavior is reproducible.
    """
    decision_prompt = (
        "You are answering a scientific question. You may retrieve papers if needed.\n"
        f"Question: {query}\n\n"
        "If you need to search for papers, reply with exactly:\n"
        "RETRIEVE: <a concise search query>\n"
        "If you can answer without retrieval, reply with:\n"
        "ANSWER: <your answer>"
    )
    decision = generator.complete(decision_prompt, desired_output_tokens=256).strip()

    if decision.upper().startswith("RETRIEVE:"):
        search_q = decision.split(":", 1)[1].strip()
        context, ids, _ = retrieve(collection, search_q, config.retrieval.top_k)
        answer = generator.complete(prompts.answer_prompt(query, context))
        return {"answer": answer, "retrieved_ids": normalize_doc_ids(ids),
                "meta": {"tool_used": True, "search_query": search_q}}
    answer = decision.split(":", 1)[1].strip() if ":" in decision else decision
    return {"answer": answer, "retrieved_ids": [], "meta": {"tool_used": False}}


# ColBERT registers itself on import (separate module: it needs its own index).
try:
    from . import colbert as _colbert  # noqa: F401
except Exception:
    pass  # ColBERT optional; only needed for the colbert strategy
