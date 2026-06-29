"""SPECTER2 document and query encoders.

SPECTER2 = a base encoder plus task-specific adapters. Documents (papers) and
queries use DIFFERENT adapters: the 'proximity' adapter for documents and the
'adhoc_query' adapter for short search queries. Using one adapter for both
silently degrades retrieval, so both are loaded and switched per call.
"""
from __future__ import annotations

import torch
from transformers import AutoTokenizer
from adapters import AutoAdapterModel

from .config import cfg


class Specter2Embedder:
    def __init__(self, config=cfg):
        e = config.embeddings
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_length = e.max_length
        self.batch_size = e.batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(e.model)
        self.model = AutoAdapterModel.from_pretrained(e.model).to(self.device).eval()
        self.model.load_adapter(e.doc_adapter, source="hf",
                                load_as="proximity", set_active=False)
        self.model.load_adapter(e.query_adapter, source="hf",
                                load_as="adhoc_query", set_active=False)

    @torch.no_grad()
    def _embed(self, texts: list[str], adapter: str):
        self.model.set_active_adapters(adapter)
        out = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True,
                                 max_length=self.max_length,
                                 return_tensors="pt").to(self.device)
            rep = self.model(**enc).last_hidden_state[:, 0, :]   # CLS pooling
            rep = torch.nn.functional.normalize(rep, p=2, dim=1)  # cosine-ready
            out.append(rep.cpu())
        return torch.cat(out).numpy()

    def embed_documents(self, texts: list[str]):
        return self._embed(texts, "proximity")

    def embed_queries(self, texts: list[str]):
        return self._embed(texts, "adhoc_query")


# Lazily instantiated shared encoder so importing this module is cheap.
_embedder: Specter2Embedder | None = None


def get_embedder() -> Specter2Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Specter2Embedder()
    return _embedder
