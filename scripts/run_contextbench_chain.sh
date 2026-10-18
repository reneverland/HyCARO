#!/usr/bin/env bash
# Serial ContextBench Lite (verified, 500) selection. No downstream LLM.
# Skips Generic-MARO and Node-RepoMARO (optimizer self-controls, not main-table baselines).
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
CONFIG="${CONFIG:-configs/ball_lam012.yaml}"
BUDGET="${BUDGET:-8192}"
POOL_CACHE="${POOL_CACHE:-cache/pools_cb_v1}"
OUTDIR="${OUTDIR:-results/contextbench_lite}"
LOGDIR="${LOGDIR:-runlogs/contextbench_lite}"
GOLD="${GOLD:-cache/contextbench_verified.parquet}"
GPU="${GPU:-0}"
mkdir -p "$OUTDIR" "$LOGDIR" "$POOL_CACHE"

export LD_LIBRARY_PATH="${ROOT}/.cuda_compat:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export GIT_TERMINAL_PROMPT=0
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"

PY="${ROOT}/.venv/bin/python"
log() { echo "[$(date '+%F %T')] $*"; }

# Main-table methods we can run without extra official stacks.
# Checkout happens inside the first method; share GPU0 with whatever else is running.
METHODS="${METHODS:-bm25 dense rrf mmr greedy binary_pso repomaro}"

for METHOD in $METHODS; do
  log "===== ${METHOD} ====="
  "$PY" scripts/run_contextbench.py \
    --config "$CONFIG" \
    --methods "$METHOD" \
    --budget "$BUDGET" \
    --pool-cache "$POOL_CACHE" \
    --gold-parquet "$GOLD" \
    --skip-log "${OUTDIR}/skipped.jsonl" \
    --out "${OUTDIR}/${METHOD}.jsonl" \
    --official-pred "${OUTDIR}/${METHOD}_pred.jsonl" \
    > "${LOGDIR}/${METHOD}.log" 2>&1
  log "${METHOD} exit $?"
done
log "contextbench chain DONE"
