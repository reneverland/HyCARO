#!/usr/bin/env bash
# Full SWE-bench Lite (300) at the frozen operating point: cost_lambda=0.12,
# ball_radius=1, seed 42, temperature 0.0. RepoMARO only.
#
# Three phases, each independently resumable, because the 40-instance sweep kept
# dying halfway on shell accidents rather than on real failures:
#
#   A  pools    GPU-bound, 4 workers. A pool depends only on the repo snapshot,
#               the issue and the budget, so this is a one-off cost shared by
#               every later run. Sequential inside a worker; the embedder holds
#               ~8GB and GPU0 belongs to another job.
#   B  generate CPU+network. Needs phase A finished, otherwise 20 shards each
#               load the embedder and the GPU dies.
#   C  evaluate official Docker harness.
#
# Resume comes from the artifacts, not from bookkeeping: phase A skips existing
# .pkl, run_swebench.py skips instance_ids already in its shard jsonl, phase C
# skips an existing report. Re-running this script is always safe.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"

CONFIG="${CONFIG:-configs/ball_lam012.yaml}"
BUDGET="${BUDGET:-8192}"
TOTAL="${TOTAL:-300}"
TAG="${TAG:-lite300_lam012}"
OUTDIR="${OUTDIR:-results/lite300}"
LOGDIR="${LOGDIR:-runlogs/lite300}"
POOL_CACHE="${POOL_CACHE:-cache/pools_v3}"
POOL_WORKERS="${POOL_WORKERS:-4}"
POOL_BATCH="${POOL_BATCH:-32}"
SHARDS="${SHARDS:-20}"
EVAL_WORKERS="${EVAL_WORKERS:-12}"
GEN_ATTEMPTS="${GEN_ATTEMPTS:-4}"
POOL_ATTEMPTS="${POOL_ATTEMPTS:-3}"

mkdir -p "$OUTDIR" "$LOGDIR" "$POOL_CACHE" results/swebench_reports

export LD_LIBRARY_PATH="${ROOT}/.cuda_compat:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export GIT_TERMINAL_PROMPT=0
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:10809}"
export HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:10809}"
export https_proxy="${https_proxy:-$HTTPS_PROXY}"
export http_proxy="${http_proxy:-$HTTP_PROXY}"

PY="${ROOT}/.venv/bin/python"

log() { echo "[$(date '+%F %T')] $*"; }

# Counting rows with `cat glob | wc -l` is what broke the sweep: an empty glob
# leaves the literal pattern and wc reports garbage that fails `[ -ge ]`.
count_rows() {
  "$PY" - "$1" <<'PYEOF'
import glob, json, sys
ids = set()
for f in glob.glob(sys.argv[1]):
    with open(f, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                try:
                    ids.add(json.loads(line)["instance_id"])
                except Exception:
                    pass
print(len(ids))
PYEOF
}

count_pools() {
  "$PY" - "$POOL_CACHE" "$BUDGET" <<'PYEOF'
import sys
from pathlib import Path
print(len(list(Path(sys.argv[1]).glob(f"*__{sys.argv[2]}.pkl"))))
PYEOF
}

log "########## LITE-300 FROZEN RUN  config=${CONFIG} tag=${TAG}"
log "  pools=${POOL_CACHE} shards=${SHARDS} eval_workers=${EVAL_WORKERS}"

########################################  A: pools
for attempt in $(seq 1 "$POOL_ATTEMPTS"); do
  have=$(count_pools)
  log "=== phase A pools: ${have}/${TOTAL} cached (attempt ${attempt}/${POOL_ATTEMPTS})"
  if [ "${have:-0}" -ge "$TOTAL" ]; then break; fi
  for w in $(seq 0 $((POOL_WORKERS - 1))); do
    "$PY" scripts/prebuild_pools.py \
      --config "$CONFIG" \
      --pool-cache "$POOL_CACHE" \
      --budget "$BUDGET" \
      --limit "$TOTAL" \
      --batch-size "$POOL_BATCH" \
      --stride "$POOL_WORKERS" \
      --worker "$w" \
      > "${LOGDIR}/pools_w${w}_a${attempt}.log" 2>&1 &
  done
  wait
done
have=$(count_pools)
log "=== phase A done: ${have}/${TOTAL} pools"
if [ "${have:-0}" -lt "$TOTAL" ]; then
  log "!!! ${have}/${TOTAL} pools only; generation will build the rest inline (slow)"
fi

########################################  B: generation
PER_SHARD=$(( (TOTAL + SHARDS - 1) / SHARDS ))
for attempt in $(seq 1 "$GEN_ATTEMPTS"); do
  rows=$(count_rows "${OUTDIR}/${TAG}_s*.jsonl")
  log "=== phase B generate: ${rows}/${TOTAL} rows (attempt ${attempt}/${GEN_ATTEMPTS})"
  if [ "${rows:-0}" -ge "$TOTAL" ]; then break; fi
  for i in $(seq 0 $((SHARDS - 1))); do
    off=$((i * PER_SHARD))
    [ "$off" -ge "$TOTAL" ] && continue
    lim=$PER_SHARD
    [ $((off + lim)) -gt "$TOTAL" ] && lim=$((TOTAL - off))
    "$PY" scripts/run_swebench.py \
      --config "$CONFIG" \
      --methods repomaro \
      --budget "$BUDGET" \
      --offset "$off" \
      --limit "$lim" \
      --generate \
      --n-samples 1 \
      --pool-cache "$POOL_CACHE" \
      --out "${OUTDIR}/${TAG}_s${i}.jsonl" \
      --predictions "${OUTDIR}/${TAG}_s${i}_predictions.json" \
      > "${LOGDIR}/gen_s${i}_a${attempt}.log" 2>&1 &
  done
  wait
done
rows=$(count_rows "${OUTDIR}/${TAG}_s*.jsonl")
log "=== phase B done: ${rows}/${TOTAL} rows"

########################################  merge
log "=== merge predictions"
MERGED="${OUTDIR}/${TAG}_predictions_RepoMARO.json"
"$PY" - "${OUTDIR}/${TAG}_s"'*'"_predictions_RepoMARO.json" "$MERGED" <<'PYEOF'
import glob, json, sys
pattern, out = sys.argv[1], sys.argv[2]
preds, seen = [], set()
for f in sorted(glob.glob(pattern)):
    data = json.load(open(f, encoding="utf-8"))
    items = data if isinstance(data, list) else [data] if "instance_id" in data else list(data.values())
    for p in items:
        if p["instance_id"] in seen or not p.get("model_patch"):
            continue
        seen.add(p["instance_id"])
        preds.append(p)
preds.sort(key=lambda p: p["instance_id"])
json.dump(preds, open(out, "w", encoding="utf-8"), indent=2)
print(f"    merged {len(preds)} -> {out}")
PYEOF

########################################  C: evaluate
REPORT="results/swebench_reports/gpt-5.6-luna+RepoMARO+s0.${TAG}.json"
if [ -s "$REPORT" ]; then
  log "=== phase C: report already exists, skipping"
else
  log "=== phase C sanitize + docker eval"
  EVDIR="${OUTDIR}/${TAG}_eval"
  SAN="${EVDIR}/sanitized/${TAG}_predictions_RepoMARO.json"
  mkdir -p "${EVDIR}/sanitized"
  "$PY" scripts/sanitize_swebench_preds.py "$MERGED" \
    --out-dir "${EVDIR}/sanitized" \
    --status-out "${EVDIR}/sanitized/apply_status_${TAG}.jsonl" \
    > "${LOGDIR}/sanitize.log" 2>&1
  sudo -n -E "$PY" -m swebench.harness.run_evaluation \
    --dataset_name SWE-bench/SWE-bench_Lite \
    --predictions_path "$SAN" \
    --max_workers "$EVAL_WORKERS" \
    --timeout 1800 \
    --run_id "${TAG}" \
    --report_dir results/swebench_reports \
    > "${LOGDIR}/eval.log" 2>&1
  grep -E 'Instances (resolved|submitted|with errors|unresolved)' "${LOGDIR}/eval.log" || :
fi

log "########## LITE-300 FROZEN RUN DONE"
