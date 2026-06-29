"""Stage 3: build the SPECTER2 Chroma vector store over the corpus.

Streams arxiv.json, embeds Title+Abstract+Categories with SPECTER2 (proximity
adapter), adds to Chroma keyed by arXiv id. Resumable; skips ids already present.

  python -m scripts.build_index                 # full corpus
  python -m scripts.build_index --limit 5000 --collection arxiv_papers_5k
"""
import argparse
import os

from tqdm import tqdm

from rag.config import cfg
from rag.embeddings import get_embedder
from rag.io_utils import stream_arxiv
from rag.vectorstore import get_collection, doc_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="Max papers to add (for validation runs).")
    ap.add_argument("--collection", default=cfg.embeddings.collection_name)
    ap.add_argument("--add-batch", type=int, default=256)
    args = ap.parse_args()

    src = os.path.join(cfg.paths.base_dir, cfg.paths.arxiv_json)
    embedder = get_embedder()
    collection = get_collection(args.collection, create=True)

    existing = set(collection.get(include=[])["ids"]) if collection.count() else set()
    print(f"'{args.collection}' has {len(existing):,} docs. Streaming {src}...")

    ids, docs, metas = [], [], []

    def flush():
        if not ids:
            return
        embs = embedder.embed_documents(docs)
        collection.add(ids=ids, embeddings=embs.tolist(),
                       documents=docs, metadatas=metas)
        ids.clear(); docs.clear(); metas.clear()

    added = 0
    for rec in tqdm(stream_arxiv(src, tuple(cfg.data.years)), desc="Embedding"):
        if rec["id"] in existing:
            continue
        ids.append(rec["id"])
        docs.append(doc_text(rec))
        metas.append({"date": rec["date"],
                      "categories": ", ".join(rec["categories"]),
                      "submitter": rec["submitter"]})
        existing.add(rec["id"])
        added += 1
        if len(ids) >= args.add_batch:
            flush()
        if args.limit and added >= args.limit:
            break
    flush()
    print(f"Done. Added {added:,}. Collection now: {collection.count():,}")


if __name__ == "__main__":
    main()
