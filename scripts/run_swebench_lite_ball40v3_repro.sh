#!/bin/bash
# Reproduce the frozen d=1 run with the current, accelerated code.
#
# The reported 18/40 was produced before the decoder/fitness batching landed.
# Those changes are meant to be evaluation-order-only, and decode output was
# verified mask-identical on 16 probes, but `_support` drifts by 1 ULP between
# the gemv and gemm paths, so the end-to-end number was never re-confirmed.
# This run closes that gap: same config, same pools, same budget, same seed.
set -euo pipefail
cd "$(dirname "$0")/.."
OFFSET="${OFFSET:-0}"
LIMIT="${LIMIT:-40}"
N_SAMPLES="${N_SAMPLES:-1}"
TAG="${TAG:-ball40v3repro}"
# The frozen configuration: ball_radius=1, untouched since the reported run.
CONFIG="${CONFIG:-configs/default.yaml}"
POOL_CACHE="${POOL_CACHE:-cache/pools_v3}"
export LD_LIBRARY_PATH="${PWD}/.cuda_compat:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# The objective's matrices are a few hundred rows wide, far too small for
# threaded BLAS to pay off, and shards already run in parallel.
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
