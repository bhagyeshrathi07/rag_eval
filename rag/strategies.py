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
    """Genuinely agentic: the model is given a search_papers tool and decides
    whether/when to call it. If it retrieves, we run the retrieval, feed results
    back, and let the model answer grounded in them. If it answers without
    retrieving, retrieved_ids is empty and we record tool_used=False.
    """
    tools = [{
        "type": "function",
        "function": {
            "name": "search_papers",
            "description": ("Search the scientific paper corpus for papers "
                            "relevant to a query. Use this whenever you need "
                            "evidence from the literature to answer."),
            "parameters": {
                "type": "object",
                "properties": {
                    "search_query": {
                        "type": "string",
                        "description": "The search query to find relevant papers.",
                    }
                },
                "required": ["search_query"],
            },
        },
    }]

    system = ("You are a scientific assistant. Answer the user's question. "
              "If you need evidence from the literature, call the search_papers "
              "tool. Answer only from retrieved papers when you retrieve; if you "
              "cannot find enough evidence, say the information is insufficient.")
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": query}]

    first = generator.chat(messages, tools=tools, tool_choice="auto",
                           desired_output_tokens=512)

    tool_calls = getattr(first, "tool_calls", None)
    if not tool_calls:
        # Model chose NOT to retrieve — answer is ungrounded by its own choice.
        return {"answer": (first.content or "").strip(),
                "retrieved_ids": [], "meta": {"tool_used": False}}

    # Execute the retrieval the model asked for.
    import json as _json
    call = tool_calls[0]
    try:
        args = _json.loads(call.function.arguments)
        search_q = args.get("search_query", query)
    except Exception:
        search_q = query

    context, ids, _ = retrieve(collection, search_q, config.retrieval.top_k)

    # Feed the tool result back and get the grounded final answer.
    messages.append({
        "role": "assistant",
        "content": first.content or "",
        "tool_calls": [{
            "id": call.id,
            "type": "function",
            "function": {"name": call.function.name,
                         "arguments": call.function.arguments},
        }],
    })
    messages.append({"role": "tool", "tool_call_id": call.id,
                     "content": context})
    final = generator.chat(messages, desired_output_tokens=1000)

    return {"answer": (final.content or "").strip(),
            "retrieved_ids": normalize_doc_ids(ids),
            "meta": {"tool_used": True, "search_query": search_q}}


# ColBERT registers itself on import (separate module: it needs its own index).
try:
    from . import colbert as _colbert  # noqa: F401
except Exception:
    pass  # ColBERT optional; only needed for the colbert strategy
