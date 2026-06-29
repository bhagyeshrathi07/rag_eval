# Retrieval Pipeline Evaluation for Scientific QA

A controlled comparison of six RAG retrieval strategies (Classic, Query-Rephrased,
Rephrased+Reranked, Fusion/RRF, Tool-Call, ColBERT) over a 2024-2025 arXiv corpus,
with an LLM-as-a-judge evaluation.

This repository accompanies the paper *"A Comparative Evaluation of Retrieval
Pipelines for Large-Scale Scientific Question Answering with Open-Weight LLMs."*

---

## 1. Prerequisites

- **Python 3.11**
- **A CUDA GPU.** Embedding ~460K papers and running ColBERT on CPU is impractical.
- **[Ollama](https://ollama.com)** installed and running, serving the generator and
  judge models locally (OpenAI-compatible API on `localhost:11434`).
- **For ColBERT only:** the CUDA compiler `nvcc` must be present (ColBERT JIT-compiles
  CUDA kernels at index time). Check with `nvcc --version`.

> **PyTorch note:** the `torch` line in `requirements.txt` is **hardware-specific**.
> This work was run on an NVIDIA GB10 (Blackwell, CUDA 13) using the `cu130` wheel.
> On other hardware, install the matching wheel from https://pytorch.org first, then
> `pip install -r requirements.txt` for the rest.

## 2. Setup

```bash
python -m venv .venv && source .venv/bin/activate

# Install the right torch wheel for your GPU FIRST, e.g. for CUDA 13:
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt

# Pull the models (generator is fixed across all strategies; judge is separate)
ollama pull llama3.1            # generator
ollama pull qwen2.5:14b         # judge (or your chosen judge model)
```

All paths, model names, and hyperparameters live in **`configs/default.yaml`**.
Edit that one file for your machine; you should not need to touch any script.

### Concurrency (speeding up the LLM stages)

Question generation and the strategy runs can issue requests in parallel. This is a
**pair of settings that must both be raised** to get a speedup:

| Setting | Where | Meaning |
|---|---|---|
| `serving.workers` | `configs/default.yaml` (or `--workers`) | how many requests the *client* sends at once |
| `OLLAMA_NUM_PARALLEL` | environment on the Ollama server | how many requests Ollama *processes* at once |

Ollama's default parallelism is low, so it serializes requests until you raise it:

```bash
export OLLAMA_NUM_PARALLEL=8     # then restart Ollama
```

Set the two to roughly match. On a **shared GPU**, start at 8 and drop to 4 if you
see contention or out-of-memory errors. Check `nvidia-smi` before large runs.

## 3. Run order

Each stage is a standalone, resumable script (skips work already done).

```bash
# Stage 0 - download the corpus (~4.7 GB, Zenodo record 15808027)
python -m scripts.download_data

# Stage 1 - reservoir-sample 10k papers (seed 42)
python -m scripts.sample

# Stage 2 - generate synthetic problem/method queries (LLM, slow)
python -m scripts.generate_questions

# Stage 3 - build the SPECTER2 Chroma vector store (GPU, slow)
python -m scripts.build_index            # add --limit 5000 --collection arxiv_papers_5k to validate first

# Stage 4 - run each retrieval strategy
python -m scripts.run_strategy --strategy classic
python -m scripts.run_strategy --strategy query_rephrased
python -m scripts.run_strategy --strategy query_rephrased_and_reranked
python -m scripts.run_strategy --strategy fusion
python -m scripts.run_strategy --strategy tool_call

# ColBERT has its OWN index (separate document pool, ColBERTv2 embeddings).
# Build the pool + PLAID index first, then run the strategy:
python -m scripts.build_colbert --collection arxiv_papers --pool-size 60000
python -m scripts.run_strategy --strategy colbert

# Stage 5 - judge every strategy's answers
python -m scripts.judge --all            # or --strategy NAME

# Stage 6 - tables & figures
jupyter lab notebooks/analysis.ipynb
```

Every `run_strategy` accepts `--limit N` (process only the first N papers, for quick
smoke tests) and `--collection NAME` (target a specific vector store).

## 4. Running the full pipeline

The whole pipeline can be driven by `run_all.sh`, or run stage-by-stage. Long
stages should run inside `tmux` so a dropped SSH connection doesn't kill them.

### One command (all stages)

```bash
tmux new -s run
conda activate /path/to/envs/research
cd /path/to/rag_eval_complete
./run_all.sh
# detach: Ctrl-b then d   |   reattach: tmux attach -t run
```

`run_all.sh` runs: sample -> questions -> index -> 5 Chroma strategies ->
ColBERT (pool + run) -> judge -> analysis. Every stage is **resumable** and skips
work already finished, so you can stop and restart freely.

Resume from a stage, or run just one:

```bash
./run_all.sh --from strategies     # resume from the strategy runs onward
./run_all.sh --only judge          # run a single stage
```

Override pool size or target collection without editing the script:

```bash
COLBERT_POOL=0 ./run_all.sh --only colbert       # full-corpus ColBERT (no confound)
COLLECTION=arxiv_papers ./run_all.sh --from index
```

### Stage-by-stage (manual)

See "Run order" above for the individual `python -m scripts.*` commands. Useful
when you want to inspect output between stages.

### Concurrency

The LLM stages (questions, strategies, judge) issue requests in parallel
(`serving.workers` in the config). For this to actually speed things up, the
Ollama server must allow concurrency: set `OLLAMA_NUM_PARALLEL` on the server
(see Setup > Concurrency). On a single shared GPU, 4-8 is typical; higher risks
out-of-memory. Without it, requests serialize and the parallel client gives only
a small speedup.

## 5. Reproducing the paper's results

```bash
# 1. Environment (install the torch wheel matching your GPU first; see Setup)
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
ollama pull llama3.1            # generator (fixed across all strategies)
ollama pull qwen2.5:14b         # judge

# 2. Data (downloads ~4.7 GB from Zenodo record 15808027)
python -m scripts.download_data

# 3. Full pipeline, paper-faithful settings
./run_all.sh
```

Key settings that define the experiment (all in `configs/default.yaml`):

- `data.seed: 42`, `data.sample_size: 10000`, `data.years: [2024, 2025]`
- `serving.generator_model: llama3.1:latest`, `serving.temperature: 0`
- `embeddings.model: allenai/specter2_base` (proximity + adhoc_query adapters)
- `colbert.pool_size: 60000` (10k gold + 50k distractors)

**Deviations from the original paper** (deliberate, documented):

- Embeddings are SPECTER2 (science-tuned), not all-MiniLM-L6-v2.
- Documents are embedded from Title + Abstract + Categories (paper: Title + Abstract).
- `tool_call` is a genuine agentic function-calling strategy (the original code
  duplicated the rerank pipeline under this name).
- ColBERT `doc_maxlen=512` (paper used 300, which truncated abstracts).
- Analysis reports both conditional (zeros excluded) and unconditional means.

To reproduce the *original* paper exactly instead, set the embedding model back
to MiniLM, the doc template to "Title + Abstract" only, and note the tool_call
caveat above.

## 6. Validate before the full run

Before the multi-hour full build, prove the pipeline on a small slice:

```bash
python -m scripts.build_index --limit 5000 --collection arxiv_papers_5k
python -m scripts.validate_index --collection arxiv_papers_5k
python -m scripts.run_strategy --strategy classic --collection arxiv_papers_5k --limit 20
```

A high self-retrieval rate confirms the embed -> store -> query loop is correct.

## 6. Layout

```
configs/default.yaml      all paths, models, hyperparameters
rag/                      shared library (imported by scripts)
  config.py               loads the YAML config
  embeddings.py           SPECTER2 document/query encoders (two adapters)
  vectorstore.py          Chroma build + SPECTER2 retrieval
  llm.py                  Ollama (OpenAI-compatible) generator + judge clients
  strategies.py           classic, query_rephrased, reranked, fusion, tool_call
  colbert.py              ColBERT late-interaction strategy (own PLAID index)
  fusion.py               Reciprocal Rank Fusion + context formatting
  prompts.py              all LLM prompt templates in one place
  parsing.py              robust JSON parsing of LLM output
  judge.py                LLM-as-a-judge scoring (is_answer gate)
  analysis.py             aggregate eval files -> metrics
  io_utils.py             streaming arxiv.json, jsonl read/write
scripts/                  one entry point per pipeline stage
  download_data / sample / generate_questions / build_index
  validate_index / run_strategy / build_colbert / judge
notebooks/analysis.ipynb  loads results, makes tables + figures
```

## 7. Notes on reproducibility

- Sampling uses a fixed seed (42) and single-pass reservoir sampling over years
  2024-2025.
- The generator (Llama-3.1-8B-Instruct) is held fixed across all six strategies so
  differences reflect *retrieval*, not the generator. It runs at temperature 0.
- The judge decides `is_answer` before scoring; refusals are scored 0 on every
  metric. Analysis reports both **conditional** means (zeros excluded) and
  **unconditional** means (zeros included) so strategies with different answer
  rates are compared fairly.
- Documents are embedded from Title + Abstract + Categories (a deviation from the
  original paper, which embedded Title + Abstract only - see `configs/default.yaml`).
- ColBERT uses its own ColBERTv2 embeddings and a separate document pool; the pool
  size (`colbert.pool_size`) controls the gold/distractor ratio.

## 8. Large files

Data and generated artifacts are **not** committed (see `.gitignore`): the raw
corpus, the sample, the Chroma store, the ColBERT index, and all `results/`. The
corpus is on Zenodo (record 15808027); the synthetic question set and 10k sample are
released separately (see the paper's Data Availability section).

