"""Streaming readers/writers for the corpus and result files."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import ijson


def stream_arxiv(path: str | Path, years: tuple[str, ...] | list[str]) -> Iterator[dict]:
    """Yield paper records from the raw Zenodo arxiv.json (a graph file where
    papers live under 'edges', each with an 'attrs' sub-object and a top-level
    'edge' = arXiv id). Filters to the given publication years.

    Yields flat dicts: {id, title, abstract, categories, date, submitter}.
    """
    with open(path, "rb") as f:
        for edge in ijson.items(f, "edges.item"):
            a = edge.get("attrs", {})
            date = a.get("date", "")
            if date[:4] not in years:
                continue
            yield {
                "id": edge.get("edge"),
                "title": a.get("title", ""),
                "abstract": a.get("abstract", ""),
                "categories": a.get("categories", []) or [],
                "date": date,
                "submitter": a.get("submitter", ""),
            }


def read_jsonl(path: str | Path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(path: str | Path, records: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def append_jsonl(path: str | Path, record: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def done_ids(path: str | Path, key: str = "edge") -> set:
    """Read a results jsonl and return the set of ids already processed
    (for resumable stages)."""
    p = Path(path)
    if not p.exists():
        return set()
    ids = set()
    for r in read_jsonl(p):
        if r.get(key) is not None:
            ids.add(r[key])
    return ids
