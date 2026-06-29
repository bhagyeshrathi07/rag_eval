"""Stage 2: generate synthetic problem/method queries for each sampled paper.

Parallelized: fires up to `--workers` concurrent requests to Ollama. Reads the
sample, fills in record["questions"], saves progress periodically. Resumable
and atomic.

CONCURRENCY IS A PAIR OF SETTINGS — both must be raised to get a speedup:
  1. serving.workers in configs/default.yaml (or --workers) — how many requests
     THIS script sends at once (client side).
  2. OLLAMA_NUM_PARALLEL on the machine running Ollama (server side) — how many
     requests Ollama will actually process concurrently. Default is low, so the
     server serializes requests unless you raise it:
         export OLLAMA_NUM_PARALLEL=8   # then restart Ollama
  If workers > OLLAMA_NUM_PARALLEL, the extra client requests just queue on the
  server, so set them to roughly match. On a shared GPU, start at 8 and drop to
  4 if you hit contention or OOM.
"""
import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from rag.config import cfg
from rag.llm import Generator
from rag import prompts
from rag.parsing import safe_json_parse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=cfg.serving.get("workers", 8),
                    help="Concurrent requests (defaults to serving.workers in config).")
    ap.add_argument("--save-every", type=int, default=50)
    args = ap.parse_args()

    sample_path = os.path.join(cfg.paths.base_dir, cfg.paths.sample_json)
    if not os.path.exists(sample_path):
        raise FileNotFoundError(f"{sample_path} not found. Run scripts.sample first.")

    with open(sample_path) as f:
        data = json.load(f)

    generator = Generator()
    todo = [r for r in data if not r.get("questions")
            and (r.get("abstract") or r.get("attrs", {}).get("abstract"))]
    total, done = len(data), len(data) - len(todo)
    if not todo:
        print(f"All {total} records already have questions. Nothing to do.")
        return
    print(f"{done}/{total} have questions. Generating {len(todo)} with "
          f"{args.workers} workers...")

    lock = threading.Lock()
    completed = 0

    def save():
        tmp = sample_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, sample_path)   # atomic

    def work(rec):
        abstract = rec.get("abstract") or rec.get("attrs", {}).get("abstract")
        try:
            raw = generator.complete(prompts.question_generation_prompt(abstract),
                                     desired_output_tokens=256)
            parsed = safe_json_parse(raw, {})
            rec["questions"] = parsed if parsed else {"error": "failed_to_parse"}
        except Exception as e:
            rec["questions"] = {"error": f"exception: {type(e).__name__}"}
        return rec

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(work, r): r for r in todo}
            for _ in tqdm(as_completed(futures), total=len(todo),
                          desc="Generating questions"):
                with lock:
                    completed += 1
                    if completed % args.save_every == 0:
                        save()
    except KeyboardInterrupt:
        print("\nInterrupted - saving progress...")
    finally:
        save()
    print(f"Done. Generated questions for {completed} records this run.")


if __name__ == "__main__":
    main()