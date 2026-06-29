"""Validate a built collection: do papers retrieve themselves? (sanity check)."""
import argparse

from rag.config import cfg
from rag.embeddings import get_embedder
from rag.vectorstore import get_collection


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default=cfg.embeddings.collection_name)
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()

    collection = get_collection(args.collection)
    embedder = get_embedder()
    sample = collection.get(limit=args.n, include=["documents"])

    hits = 0
    for eid, doc in zip(sample["ids"], sample["documents"]):
        abs = doc.split("Abstract:", 1)[-1][:200]
        qemb = embedder.embed_queries([abs])
        res = collection.query(query_embeddings=qemb.tolist(), n_results=3)
        if eid in res["ids"][0]:
            hits += 1
    print(f"Self-retrieval: {hits}/{len(sample['ids'])} papers found themselves in top-3")
    print("High is good; confirms embed -> store -> query is wired correctly.")


if __name__ == "__main__":
    main()
