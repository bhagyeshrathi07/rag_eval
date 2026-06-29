"""Stage 2: generate synthetic problem/method queries for each sampled paper.

Reads the sample, fills in record["questions"] = {problem_query, method_query}
using the generator. Resumable and atomic: safe to interrupt.
"""
import json
import os

from tqdm import tqdm

from rag.config import cfg
from rag.llm import Generator
from rag import prompts
from rag.parsing import safe_json_parse


def main():
    sample_path = os.path.join(cfg.paths.base_dir, cfg.paths.sample_json)
    if not os.path.exists(sample_path):
        raise FileNotFoundError(f"{sample_path} not found. Run scripts.sample first.")

    with open(sample_path) as f:
        data = json.load(f)

    generator = Generator()
    total = len(data)
    done = sum(1 for r in data if r.get("questions"))
    if done == total:
        print(f"All {total} records already have questions. Nothing to do.")
        return
    print(f"{done}/{total} have questions. Generating the remaining {total - done}...")

    def save():
        tmp = sample_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, sample_path)   # atomic

    new = 0
    try:
        for rec in tqdm(data, desc="Generating questions"):
            if rec.get("questions"):
                continue
            abstract = rec.get("abstract") or rec.get("attrs", {}).get("abstract")
            if not abstract:
                continue
            raw = generator.complete(prompts.question_generation_prompt(abstract),
                                     desired_output_tokens=256)
            parsed = safe_json_parse(raw, {})
            rec["questions"] = parsed if parsed else {"error": "failed_to_parse"}
            new += 1
            if new % 10 == 0:
                save()
    except KeyboardInterrupt:
        print("\nInterrupted - saving progress...")
    finally:
        save()
    print(f"Done. Added questions to {new} records this run.")


if __name__ == "__main__":
    main()
