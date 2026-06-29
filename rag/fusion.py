"""Reciprocal Rank Fusion + context formatting (ported verbatim from notebook)."""


def reciprocal_rank_fusion(results_by_query, final_top_k=3, rrf_k=60):
    """RRF over Chroma ranked outputs. score(doc) = sum(1 / (rrf_k + rank)).
    Distance is used only as a tie-breaker (Chroma returns lower=better)."""
    doc_map = {}
    for query_index, result in enumerate(results_by_query):
        for rank, doc in enumerate(result["docs"], start=1):
            doc_id = doc["doc_id"]
            distance = doc.get("distance", float("inf"))
            if doc_id not in doc_map:
                doc_map[doc_id] = {
                    "doc_id": doc_id,
                    "document": doc.get("document", ""),
                    "metadata": doc.get("metadata", {}),
                    "rrf_score": 0.0,
                    "retrieval_frequency": 0,
                    "best_rank": rank,
                    "ranks": [],
                    "query_hits": [],
                    "best_distance": distance,
                }
            d = doc_map[doc_id]
            d["rrf_score"] += 1.0 / (rrf_k + rank)
            d["retrieval_frequency"] += 1
            d["ranks"].append(rank)
            d["query_hits"].append(query_index + 1)
            if rank < d["best_rank"]:
                d["best_rank"] = rank
            if distance < d["best_distance"]:
                d["best_distance"] = distance
                d["document"] = doc.get("document", "")
                d["metadata"] = doc.get("metadata", {})

    fused = sorted(
        doc_map.values(),
        key=lambda x: (x["rrf_score"], x["retrieval_frequency"],
                       -x["best_distance"], -x["best_rank"]),
        reverse=True,
    )
    return fused[:final_top_k], fused


def format_fused_docs_as_context(selected_docs):
    blocks = []
    for i, doc in enumerate(selected_docs, start=1):
        blocks.append(
            f"Paper {i}\n"
            f"doc_id: {doc['doc_id']}\n"
            f"{doc['document']}\n"
            f"{doc['metadata']}\n"
            f"distance: {doc['best_distance']}\n"
            f"rrf_score: {doc['rrf_score']}\n"
            f"retrieval_frequency: {doc['retrieval_frequency']}\n"
        )
    return "\n\n".join(blocks)


def filter_context_by_selected_ids(context, selected_doc_ids):
    """Best-effort filtering of retrieved context to keep only selected papers."""
    if not context or not selected_doc_ids:
        return ""
    context_str = str(context)
    for marker in ["\nPaper ", "\n\nPaper ", "\nDocument ", "\n\nDocument "]:
        if marker in context_str:
            parts = context_str.split(marker)
            selected = []
            for idx, part in enumerate(parts):
                block = part if idx == 0 else marker.strip() + " " + part
                if any(str(d) in block for d in selected_doc_ids):
                    selected.append(block.strip())
            return "\n\n".join(selected)
    paragraphs = context_str.split("\n\n")
    return "\n\n".join(p.strip() for p in paragraphs
                       if any(str(d) in p for d in selected_doc_ids))
