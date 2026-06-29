"""Chroma vector store: build (SPECTER2 document embeddings) and retrieval
(SPECTER2 query embeddings). Cosine space to match normalized SPECTER2 vectors.
"""
from __future__ import annotations

import chromadb

from .config import cfg
from .embeddings import get_embedder


def get_collection(name: str | None = None, create: bool = False):
    name = name or cfg.embeddings.collection_name
    client = chromadb.PersistentClient(path=cfg.paths.chroma_dir)
    if create:
        return client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"})
    return client.get_collection(name=name)


def doc_text(rec: dict) -> str:
    """Compose the text embedded per document, per the config template."""
    return cfg.embeddings.doc_template.format(
        title=rec.get("title", ""),
        abstract=rec.get("abstract", ""),
        categories=", ".join(rec.get("categories", []) or []),
    )


def retrieve(collection, query: str, top_k: int | None = None):
    """Embed the query with SPECTER2 (adhoc_query adapter) and search by vector.

    Returns (context_string, ids, structured_docs).
    """
    top_k = top_k or cfg.retrieval.top_k
    q_emb = get_embedder().embed_queries([query])
    res = collection.query(query_embeddings=q_emb.tolist(), n_results=top_k)

    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    ids = res["ids"][0]

    structured, parts = [], []
    for rank, (i, d, m, dist) in enumerate(zip(ids, docs, metas, dists), 1):
        structured.append({"rank": rank, "doc_id": i, "document": d,
                           "metadata": m, "distance": dist})
        parts.append(f"Doc ID: {i}\n{d}\n{m}\ndistance: {dist}\n")
    return "\n".join(parts), ids, structured
