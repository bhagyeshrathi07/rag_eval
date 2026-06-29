"""Stage 0: download the arXiv corpus from Zenodo (idempotent)."""
import os
import sys
import urllib.request

from rag.config import cfg


def main():
    dest = os.path.join(cfg.paths.base_dir, cfg.paths.arxiv_json)
    if os.path.exists(dest):
        size_gb = os.path.getsize(dest) / 1e9
        print(f"{dest} already exists ({size_gb:.1f} GB) - skipping download.")
        return

    url = cfg.data.zenodo_url
    print(f"Downloading corpus from {url}\n  -> {dest}\nThis is ~4.7 GB and may take a while...")

    def hook(block, block_size, total):
        if total > 0:
            pct = min(100, block * block_size * 100 // total)
            sys.stdout.write(f"\r  {pct}%")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest, reporthook=hook)
    print("\nDownload complete.")


if __name__ == "__main__":
    main()
