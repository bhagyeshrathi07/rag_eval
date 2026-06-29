#!/usr/bin/env bash
# =============================================================================
# Full pipeline driver. Each stage is resumable; re-running skips finished work.
# Run the WHOLE thing:        ./run_all.sh
# Run from a specific stage:  ./run_all.sh --from strategies
# Run a single stage:         ./run_all.sh --only judge
#
# Stages: sample questions index strategies colbert judge analysis
#
# Long stages (questions, index, strategies, colbert) should run in tmux so a
# dropped connection doesn't kill them. This script does NOT background anything;
# launch it inside `tmux new -s run` and detach with Ctrl-b d.
# =============================================================================
set -euo pipefail

# --- Settings you may want to override -----------------------------------
COLLECTION="${COLLECTION:-arxiv_papers}"     # full corpus collection name
COLBERT_POOL="${COLBERT_POOL:-60000}"        # 60000 = paper-faithful; 0 = full corpus
STRATEGIES=(classic query_rephrased query_rephrased_and_reranked fusion tool_call colbert)

ALL_STAGES=(sample questions index strategies colbert judge analysis)

# --- Arg parsing ---------------------------------------------------------
FROM=""; ONLY=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --from) FROM="$2"; shift 2;;
    --only) ONLY="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

should_run() {
  local stage="$1"
  if [[ -n "$ONLY" ]]; then [[ "$stage" == "$ONLY" ]]; return; fi
  if [[ -n "$FROM" ]]; then
    local started=0
    for s in "${ALL_STAGES[@]}"; do
      [[ "$s" == "$FROM" ]] && started=1
      [[ "$s" == "$stage" && $started -eq 1 ]] && return 0
    done
    return 1
  fi
  return 0   # no filter -> run everything
}

banner() { echo; echo "============================================================"; echo ">>> $1"; echo "============================================================"; }

# --- Pre-flight ----------------------------------------------------------
banner "Pre-flight"
python -c "import torch; assert torch.cuda.is_available(), 'GPU not visible'; print('GPU OK:', torch.cuda.get_device_name(0))"
curl -s http://localhost:11434/api/tags >/dev/null && echo "Ollama reachable" || { echo "ERROR: Ollama not reachable on :11434"; exit 1; }
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader || true

# --- Stages --------------------------------------------------------------
if should_run sample; then
  banner "Stage 1: sample 10k papers (seed 42)"
  python -m scripts.sample
fi

if should_run questions; then
  banner "Stage 2: generate synthetic questions (parallel)"
  python -m scripts.generate_questions
fi

if should_run index; then
  banner "Stage 3: build full SPECTER2 index -> $COLLECTION"
  python -m scripts.build_index --collection "$COLLECTION"
fi

if should_run strategies; then
  for s in classic query_rephrased query_rephrased_and_reranked fusion tool_call; do
    banner "Stage 4: run strategy '$s'"
    python -m scripts.run_strategy --strategy "$s" --collection "$COLLECTION"
  done
fi

if should_run colbert; then
  banner "Stage 4b: build ColBERT pool (size=$COLBERT_POOL) + run"
  python -m scripts.build_colbert --collection "$COLLECTION" --pool-size "$COLBERT_POOL"
  python -m scripts.run_strategy --strategy colbert --collection "$COLLECTION"
fi

if should_run judge; then
  banner "Stage 5: judge all strategies"
  python -m scripts.judge --all
fi

if should_run analysis; then
  banner "Stage 6: analysis"
  echo "Open the notebook to generate tables + figures:"
  echo "  jupyter lab notebooks/analysis.ipynb"
  echo "(or run it headless with: jupyter nbconvert --to notebook --execute notebooks/analysis.ipynb)"
fi

banner "Done."