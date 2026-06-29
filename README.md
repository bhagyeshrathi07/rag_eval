# Retrieval Pipeline Evaluation for Scientific QA

A controlled comparison of six RAG retrieval strategies (Classic, Query-Rephrased,
Rephrased+Reranked, Fusion/RRF, Tool-Call, ColBERT) over a 2024–2025 arXiv corpus,
with an LLM-as-a-judge evaluation.

This repository accompanies the paper *"A Comparative Evaluation of Retrieval
Pipelines for Large-Scale Scientific Question Answering with Open-Weight LLMs."*

---

## 1. Prerequisites

- **Python 3.11**
- **A CUDA GPU.** Embedding 460K papers and running ColBERT on CPU is impractical.
- **[Ollama](https://ollama.com)** installed and running, serving the generator and
  judge models locally (OpenAI-compatible API on `localhost:11434`).

> **PyTorch note:** the `torch` line in `requirements.txt` is **hardware-specific**.
> The pinned build targets recent NVIDIA GPUs (CUDA 13 / Blackwell). On other
> hardware, install the matching wheel from https://pytorch.org first, then
> `pip install -r requirements.txt` for the rest.

## 2. Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Pull the models used in the paper
ollama pull llama3.1            # generator
ollama pull qwen2.5:14b         # judge (or your chosen judge model)
```

All paths, model names, and toggles live in **`configs/default.yaml`**. Edit that one
file for your machine; you should not need to touch any script.

## 3. Run order

Each stage is a standalone script. Stages are resumable and skip work already done.

```bash
# Stage 0 — download the corpus (~4.7 GB, Zenodo record 15808027)
python -m scripts.download_data

# Stage 1 — reservoir-sample 10k papers (seed 42)
python -m scripts.sample

# Stage 2 — generate synthetic problem/method queries (LLM, slow)
python -m scripts.generate_questions

# Stage 3 — build the SPECTER2 Chroma vector store (GPU, slow)
python -m scripts.build_index            # add --limit 5000 to validate first

# Stage 4 — run a retrieval strategy (repeat per strategy)
python -m scripts.run_strategy --strategy classic
python -m scripts.run_strategy --strategy query_rephrased
python -m scripts.run_strategy --strategy query_rephrased_and_reranked
python -m scripts.run_strategy --strategy fusion
python -m scripts.run_strategy --strategy tool_call
# ColBERT has its own index; see scripts/build_colbert.py then:
python -m scripts.run_strategy --strategy colbert

# Stage 5 — judge every strategy's answers
python -m scripts.judge --strategy classic
# ...repeat per strategy, or: python -m scripts.judge --all

# Stage 6 — tables & figures
jupyter lab notebooks/analysis.ipynb
```

## 4. Validate before the full run

Before the multi-hour full build, prove the pipeline on a small slice:

```bash
python -m scripts.build_index --limit 5000 --collection arxiv_papers_5k
python -m scripts.validate_index --collection arxiv_papers_5k
```

A high self-retrieval rate confirms the embed → store → query loop is correct.

## 5. Layout

```
configs/default.yaml      all paths, models, hyperparameters
rag/                      shared library (imported by scripts)
  config.py               loads + validates the YAML config
  embeddings.py           SPECTER2 document/query encoders
  vectorstore.py          Chroma build + retrieval
  llm.py                  Ollama (OpenAI-compatible) clients
  strategies.py           the six retrieval strategies, one interface
  judge.py                LLM-as-a-judge scoring
  io_utils.py             streaming arxiv.json, jsonl read/write
scripts/                  one entry point per pipeline stage
notebooks/analysis.ipynb  loads results, makes tables + figures
```

## 6. Notes on reproducibility

- Sampling uses a fixed seed (42) and single-pass reservoir sampling.
- The generator runs at temperature 0; the judge gates `is_answer` before scoring.
- Documents are embedded from Title + Abstract + Categories (a deviation from the
  original paper, which embedded Title + Abstract only — see `configs/default.yaml`).
