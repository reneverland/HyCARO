#!/bin/bash
# RepoMARO (k anchors, radius d) on SWE-bench Lite first 40.
# Reuses the cached candidate pools; Luna patches; no Docker eval.
# Usage: bash scripts/run_swebench_lite_ball40.sh
set -euo pipefail
cd "$(dirname "$0")/.."
OFFSET="${OFFSET:-0}"
LIMIT="${LIMIT:-40}"
N_SAMPLES="${N_SAMPLES:-1}"
TAG="${TAG:-ball40}"
# Pools are keyed by instance and budget only, so a representation change (the
# hypergraph scanner, max_units, class/method splitting) needs a fresh cache dir.
POOL_CACHE="${POOL_CACHE:-cache/pools}"
export LD_LIBRARY_PATH="${PWD}/.cuda_compat:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export GIT_TERMINAL_PROMPT=0
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:10809}"
export HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:10809}"
export https_proxy="${https_proxy:-$HTTPS_PROXY}"
export http_proxy="${http_proxy:-$HTTP_PROXY}"
# shellcheck disable=SC1091
source .venv/bin/activate
exec python scripts/run_swebench.py \
  --config configs/default.yaml \
  --methods repomaro \
  --budget 8192 \
  --offset "${OFFSET}" \
  --limit "${LIMIT}" \
  --generate \
  --n-samples "${N_SAMPLES}" \
  --pool-cache "${POOL_CACHE}" \
  --out "results/swebench_lite_8k_${TAG}.jsonl" \
  --predictions "results/swebench_lite_8k_${TAG}_predictions.json"
