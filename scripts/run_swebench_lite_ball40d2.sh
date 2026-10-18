#!/bin/bash
# RepoMARO with ball_radius=2 on SWE-bench Lite first 40.
#
# Identical to the frozen ball40v3 run except for `ball_radius` in the config:
# same cached candidate pools, same budget, same downstream model, same seed.
# The d=2 value was chosen from a gold-span reachability sweep over these same
# 40 instances, so this run is exploratory -- a clean holdout sweep is still
# needed before the value can be reported as a frozen setting.
set -euo pipefail
cd "$(dirname "$0")/.."
OFFSET="${OFFSET:-0}"
LIMIT="${LIMIT:-40}"
N_SAMPLES="${N_SAMPLES:-1}"
TAG="${TAG:-ball40d2}"
CONFIG="${CONFIG:-configs/ball_d2.yaml}"
# The pool is keyed by instance and budget only, and the representation is
# unchanged, so the ball40v3 cache is reused verbatim.
POOL_CACHE="${POOL_CACHE:-cache/pools_v3}"
export LD_LIBRARY_PATH="${PWD}/.cuda_compat:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# The objective's matrices are a few hundred rows wide, far too small for
# threaded BLAS to pay off, and shards are run in parallel anyway. Left
# unbounded, each shard opened ~130 threads and oversubscribed the machine.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export GIT_TERMINAL_PROMPT=0
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
# shellcheck disable=SC1091
source .venv/bin/activate
exec python scripts/run_swebench.py \
  --config "${CONFIG}" \
  --methods repomaro \
  --budget 8192 \
  --offset "${OFFSET}" \
  --limit "${LIMIT}" \
  --generate \
  --n-samples "${N_SAMPLES}" \
  --pool-cache "${POOL_CACHE}" \
  --out "results/swebench_lite_8k_${TAG}.jsonl" \
  --predictions "results/swebench_lite_8k_${TAG}_predictions.json"
