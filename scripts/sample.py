"""Stage 1: reservoir-sample N papers from the corpus (fixed seed).

Single-pass reservoir sampling over the streamed corpus, filtered to the
configured years. Writes a flat JSON list of paper records.
"""
import json
import os
import random

from rag.config import cfg
from rag.io_utils import stream_arxiv


def main():
    src = os.path.join(cfg.paths.base_dir, cfg.paths.arxiv_json)
    dst = os.path.join(cfg.paths.base_dir, cfg.paths.sample_json)
    if os.path.exists(dst):
        print(f"{dst} already exists - skipping to preserve the sample.")
        return

    random.seed(cfg.data.seed)
    size = cfg.data.sample_size
    years = tuple(cfg.data.years)

    reservoir, seen = [], 0
    for rec in stream_arxiv(src, years):
        seen += 1
        if len(reservoir) < size:
            reservoir.append(rec)
        else:
            j = random.randint(0, seen - 1)
            if j < size:
                reservoir[j] = rec

    print(f"Streamed {seen:,} papers in years {years}; sampled {len(reservoir):,}.")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(reservoir, f, indent=2)
    print(f"Wrote {dst}")


if __name__ == "__main__":
    main()
