#!/usr/bin/env bash
# Serial SWE-bench Lite chain for the remaining external baselines.
# LongCodeZip ranking can overlap Node-RepoMARO (GPU0, no API).
# LocAgent search and every generate step wait for Node generate to release the API.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
LOGDIR="${LOGDIR:-runlogs/lite300_baselines}"
MANIFEST="${MANIFEST:-cache/repomap_lite300_manifest.jsonl}"
mkdir -p "$LOGDIR" cache/locagent_graphs cache/locagent_bm25

export LD_LIBRARY_PATH="${ROOT}/.cuda_compat:${LD_LIBRARY_PATH:-}"

log() { echo "[$(date '+%F %T')] $*"; }

wait_node_generate() {
  log "waiting for Node-RepoMARO generate workers to exit"
  while pgrep -f 'run_swebench.py --config configs/ball_lam012.yaml --methods node_repomaro' >/dev/null 2>&1; do
    sleep 30
  done
  log "Node-RepoMARO generate idle"
}

wait_file_rows() {
  local path="$1" need="$2" label="$3"
  while true; do
    local n=0
    if [ -s "$path" ]; then
      n=$(wc -l < "$path")
    fi
    n=${n// /}
    log "${label}: ${n}/${need}"
    [ "${n:-0}" -ge "$need" ] && return 0
    sleep 60
  done
}

######################################## LongCodeZip: ranking must already be running or we start it
LCZ_RANK="${LCZ_RANK:-cache/longcodezip_rankings_lite300.jsonl}"
if [ ! -s "$LCZ_RANK" ] || [ "$(wc -l < "$LCZ_RANK" | tr -d ' ')" -lt 300 ]; then
  if ! pgrep -f 'scripts/external/run_longcodezip.py' >/dev/null 2>&1; then
    log "starting LongCodeZip ranking on GPU0"
    CUDA_VISIBLE_DEVICES="${LCZ_GPU:-0}" \
    HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" \
    TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}" \
    nohup "$ROOT/.venv/bin/python" scripts/external/run_longcodezip.py \
      --manifest "$MANIFEST" \
      --pool-cache cache/pools_v3 \
      --out "$LCZ_RANK" \
      > "$LOGDIR/longcodezip_ranking.log" 2>&1 &
  fi
fi

######################################## LocAgent graphs (CPU, no API)
if ! pgrep -f 'scripts/external/run_locagent_graphs.py' >/dev/null 2>&1; then
  local_graphs=$(ls cache/locagent_graphs/*.pkl 2>/dev/null | wc -l)
  if [ "${local_graphs:-0}" -lt 300 ]; then
    log "starting LocAgent graph build"
    nohup "$ROOT/.venv/bin/python" scripts/external/run_locagent_graphs.py \
      --manifest "$MANIFEST" \
      --out-dir cache/locagent_graphs \
      --workers "${GRAPH_WORKERS:-4}" \
      > "$LOGDIR/locagent_graphs.log" 2>&1 &
  fi
fi

wait_file_rows "$LCZ_RANK" 300 "LongCodeZip ranking"
wait_node_generate

log "===== LongCodeZip generate + eval ====="
METHOD=longcodezip TAG=lite300_longcodezip RANKING="$LCZ_RANK" \
  ./scripts/run_lite300_method.sh

######################################## LocAgent localize (API) then generate
while [ "$(find cache/locagent_graphs -name '*.pkl' 2>/dev/null | wc -l)" -lt 300 ]; do
  log "LocAgent graphs: $(find cache/locagent_graphs -name '*.pkl' 2>/dev/null | wc -l)/300"
  sleep 60
done

LOCAGENT_PY="${LOCAGENT_PY:-$ROOT/external/venvs/locagent/bin/python}"
if [ ! -x "$LOCAGENT_PY" ]; then
  LOCAGENT_PY="$ROOT/.venv/bin/python"
fi
export GRAPH_INDEX_DIR="${GRAPH_INDEX_DIR:-$ROOT/cache/locagent_graphs}"
export BM25_INDEX_DIR="${BM25_INDEX_DIR:-$ROOT/cache/locagent_bm25}"
export PYTHONPATH="${ROOT}/external/LocAgent:${PYTHONPATH:-}"

LOC_RANK="${LOC_RANK:-cache/locagent_rankings_lite300.jsonl}"
if [ ! -s "$LOC_RANK" ] || [ "$(wc -l < "$LOC_RANK" | tr -d ' ')" -lt 300 ]; then
  log "===== LocAgent localize ====="
  "$LOCAGENT_PY" scripts/external/run_locagent_localize.py \
    --manifest "$MANIFEST" \
    --out "$LOC_RANK" \
    --locs-out cache/locagent_locs_lite300.jsonl \
    --model "${LOCAGENT_MODEL:-openai/gpt-5.6-luna}" \
    > "$LOGDIR/locagent_localize.log" 2>&1
fi

log "===== LocAgent generate + eval ====="
METHOD=locagent TAG=lite300_locagent RANKING="$LOC_RANK" \
  ./scripts/run_lite300_method.sh

log "===== CodeScout SKIP ====="
log "CodeScout upstream is training-only; official SWE eval lives in the OpenHands benchmarks fork and needs CodeScout-14B weights. Not runnable from this tree."

log "external chain DONE"
