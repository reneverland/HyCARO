#!/bin/bash
# DIAGNOSTIC UPPER BOUND -- not a method, never report as one.
#
# Identical to the frozen ball40v3 run (d=1, same pools, same budget, same model,
# same seed) except that the reference-patch units are forced into the selected
# context. It answers one question: is the context the selector returns what
# limits the downstream model? If resolved barely moves against ball40v3's 18/40,
# then redesigning the selection objective cannot pay off either.
set -euo pipefail
cd "$(dirname "$0")/.."
OFFSET="${OFFSET:-0}"
LIMIT="${LIMIT:-40}"
N_SAMPLES="${N_SAMPLES:-1}"
TAG="${TAG:-oracle40}"
# The frozen configuration, so the only difference from ball40v3 is the oracle.
CONFIG="${CONFIG:-configs/default.yaml}"
POOL_CACHE="${POOL_CACHE:-cache/pools_v3}"
export LD_LIBRARY_PATH="${PWD}/.cuda_compat:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Matrices are a few hundred rows wide; threaded BLAS only oversubscribes the
# machine when shards run in parallel.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
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
  --config "${CONFIG}" \
  --methods repomaro \
  --budget 8192 \
  --offset "${OFFSET}" \
  --limit "${LIMIT}" \
  --generate \
  --oracle-gold \
  --n-samples "${N_SAMPLES}" \
  --pool-cache "${POOL_CACHE}" \
  --out "results/swebench_lite_8k_${TAG}.jsonl" \
  --predictions "results/swebench_lite_8k_${TAG}_predictions.json"
