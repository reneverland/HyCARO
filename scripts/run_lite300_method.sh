#!/usr/bin/env bash
# One SWE-bench Lite method: generate (+ optional skip of the first 40) then Docker eval.
# Resume is by artifacts. Safe to re-run.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"

METHOD="${METHOD:?METHOD=bm25|dense|rrf|mmr|greedy|binary_pso|generic_maro|node_repomaro|repomap|longcodezip|locagent}"
RANKING="${RANKING:-}"
CONFIG="${CONFIG:-configs/ball_lam012.yaml}"
BUDGET="${BUDGET:-8192}"
TOTAL="${TOTAL:-300}"
SKIP40="${SKIP40:-0}"
TAG="${TAG:-lite300_${METHOD}}"
OUTDIR="${OUTDIR:-results/lite300_baselines/${METHOD}}"
LOGDIR="${LOGDIR:-runlogs/lite300_baselines/${METHOD}}"
POOL_CACHE="${POOL_CACHE:-cache/pools_v3}"
SHARDS="${SHARDS:-20}"
EVAL_WORKERS="${EVAL_WORKERS:-8}"
GEN_ATTEMPTS="${GEN_ATTEMPTS:-4}"
GPU="${GPU:-1}"

# Latest complete Dense-40 (patches + official report): v3p5 sample 4.
DENSE40_JSONL="${DENSE40_JSONL:-results/swebench_lite_8k_skeleton40.jsonl}"
DENSE40_PREDS="${DENSE40_PREDS:-results/swebench_lite_8k_v3_pass3/Dense_s4.json}"
DENSE40_REPORT="${DENSE40_REPORT:-results/swebench_reports/gpt-5.6-luna+Dense+s4.v3p5_Dense_s4.json}"

mkdir -p "$OUTDIR" "$LOGDIR" results/swebench_reports

export LD_LIBRARY_PATH="${ROOT}/.cuda_compat:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU}"
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
# run_swebench RECORD_NAME
case "$METHOD" in
  bm25) REC="BM25" ;;
  dense) REC="Dense" ;;
  rrf) REC="RRF" ;;
  mmr) REC="MMR" ;;
  greedy) REC="Greedy-Utility" ;;
  binary_pso) REC="Binary-PSO" ;;
  generic_maro) REC="Generic-MARO" ;;
  node_repomaro) REC="Node-RepoMARO" ;;
  repomap) REC="RepoMap"; RANKING="${RANKING:-cache/repomap_rankings_lite300.jsonl}" ;;
  longcodezip) REC="LongCodeZip"; RANKING="${RANKING:-cache/longcodezip_rankings_lite300.jsonl}" ;;
  locagent) REC="LocAgent"; RANKING="${RANKING:-cache/locagent_rankings_lite300.jsonl}" ;;
  *) echo "unknown METHOD=$METHOD"; exit 1 ;;
esac
EXTRA_ARGS=()
if [ -n "$RANKING" ]; then
  EXTRA_ARGS+=(--external-ranking "$RANKING")
fi

log() { echo "[$(date '+%F %T')] [$METHOD] $*"; }

count_rows() {
  # Pattern is expanded in Python so bash never sees `[0-9]*`.
  "$PY" - "$1" <<'PYEOF'
import glob, json, sys
ids = set()
for f in glob.glob(sys.argv[1]):
    with open(f, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                ids.add(json.loads(line)["instance_id"])
            except Exception:
                pass
print(len(ids))
PYEOF
}

######################################## seed first 40 for Dense
OFFSET=0
GEN_TOTAL=$TOTAL
if [ "$SKIP40" = "1" ]; then
  OFFSET=40
  GEN_TOTAL=$((TOTAL - 40))
  log "seeding first 40 from ${DENSE40_PREDS} + ${DENSE40_JSONL}"
  "$PY" - "$DENSE40_JSONL" "$DENSE40_PREDS" "$OUTDIR/${TAG}_seed40.jsonl" \
        "$OUTDIR/${TAG}_seed40_predictions_${REC}.json" "$REC" <<'PYEOF'
import json, sys
from pathlib import Path
src_jsonl, src_preds, out_jsonl, out_preds, rec = sys.argv[1:6]
preds = json.loads(Path(src_preds).read_text())
items = preds if isinstance(preds, list) else list(preds.values())
patch = {p["instance_id"]: p.get("model_patch") or "" for p in items}
# selection metadata from the newest Dense 40 jsonl (deterministic given the pool)
meta = {}
for line in Path(src_jsonl).read_text().splitlines():
    if not line.strip():
        continue
    r = json.loads(line)
    if r.get("method") != "Dense":
        continue
    meta[r["instance_id"]] = r
rows = []
pred_out = []
for iid, p in patch.items():
    rec0 = dict(meta.get(iid) or {"instance_id": iid, "method": rec})
    rec0["method"] = rec
    rec0["generation"] = {
        "model": "gpt-5.6-luna",
        "n_samples": 1,
        "patch": p,
        "samples": [{"patch": p}],
        "reused_from": "v3p5_Dense_s4",
    }
    rows.append(rec0)
    if p:
        pred_out.append({
            "instance_id": iid,
            "model_patch": p,
            "model_name_or_path": f"gpt-5.6-luna+{rec}+s0",
        })
Path(out_jsonl).write_text("".join(json.dumps(r) + "\n" for r in rows))
json.dump(pred_out, open(out_preds, "w"), indent=2)
print(f"    seeded {len(rows)} rows, {len(pred_out)} preds", flush=True)
PYEOF
fi

######################################## B: generate remaining
PER_SHARD=$(( (GEN_TOTAL + SHARDS - 1) / SHARDS ))
for attempt in $(seq 1 "$GEN_ATTEMPTS"); do
  rows=$(count_rows "${OUTDIR}/${TAG}_s[0-9]*.jsonl")
  rows=${rows:-0}
  log "=== generate: ${rows}/${GEN_TOTAL} new rows (attempt ${attempt}/${GEN_ATTEMPTS})"
  if [ "${rows:-0}" -ge "$GEN_TOTAL" ]; then break; fi
  for i in $(seq 0 $((SHARDS - 1))); do
    off=$((OFFSET + i * PER_SHARD))
    [ "$off" -ge $((OFFSET + GEN_TOTAL)) ] && continue
    lim=$PER_SHARD
    [ $((off + lim)) -gt $((OFFSET + GEN_TOTAL)) ] && lim=$((OFFSET + GEN_TOTAL - off))
    "$PY" scripts/run_swebench.py \
      --config "$CONFIG" \
      --methods "$METHOD" \
      --budget "$BUDGET" \
      --offset "$off" \
      --limit "$lim" \
      --generate \
      --n-samples 1 \
      --pool-cache "$POOL_CACHE" \
      "${EXTRA_ARGS[@]}" \
      --out "${OUTDIR}/${TAG}_s${i}.jsonl" \
      --predictions "${OUTDIR}/${TAG}_s${i}_predictions.json" \
      > "${LOGDIR}/gen_s${i}_a${attempt}.log" 2>&1 &
  done
  wait
done
rows=$(count_rows "${OUTDIR}/${TAG}_s[0-9]*.jsonl")
rows=${rows:-0}
log "=== generate done: ${rows}/${GEN_TOTAL} new rows"

######################################## merge predictions
log "=== merge predictions"
MERGED="${OUTDIR}/${TAG}_predictions_${REC}.json"
# Pass the glob as a single literal for Python; do not let bash expand it.
"$PY" - "${OUTDIR}/${TAG}_s[0-9]*_predictions_${REC}.json" "$MERGED" \
      "${OUTDIR}/${TAG}_seed40_predictions_${REC}.json" <<'PYEOF'
import glob, json, sys
from pathlib import Path
pattern, out, seed = sys.argv[1:4]
preds, seen = [], set()
files = sorted(glob.glob(pattern))
if Path(seed).exists():
    files = [seed] + files
for f in files:
    data = json.load(open(f, encoding="utf-8"))
    items = data if isinstance(data, list) else [data] if "instance_id" in data else list(data.values())
    for p in items:
        if p["instance_id"] in seen or not p.get("model_patch"):
            continue
        seen.add(p["instance_id"])
        p["model_name_or_path"] = p.get("model_name_or_path") or "gpt-5.6-luna"
        preds.append(p)
preds.sort(key=lambda p: p["instance_id"])
json.dump(preds, open(out, "w", encoding="utf-8"), indent=2)
print(f"    merged {len(preds)} -> {out}", flush=True)
PYEOF

######################################## C: evaluate
REPORT="results/swebench_reports/gpt-5.6-luna+${REC}+s0.${TAG}.json"
if [ -s "$REPORT" ]; then
  log "=== eval report exists, skip"
else
  log "=== sanitize + docker eval"
  EVDIR="${OUTDIR}/${TAG}_eval"
  SAN="${EVDIR}/sanitized/${TAG}_predictions_${REC}.json"
  mkdir -p "${EVDIR}/sanitized"
  "$PY" scripts/sanitize_swebench_preds.py "$MERGED" \
    --out-dir "${EVDIR}/sanitized" \
    --status-out "${EVDIR}/sanitized/apply_status_${TAG}.jsonl" \
    > "${LOGDIR}/sanitize.log" 2>&1

  EVAL_IDS_ARGS=()
  if [ "$SKIP40" = "1" ]; then
    mapfile -t NEW_IDS < <("$PY" - "$SAN" "$OUTDIR/${TAG}_seed40_predictions_${REC}.json" <<'PYEOF'
import json, sys
from pathlib import Path
san, seed = sys.argv[1:3]
items = json.load(open(san))
items = items if isinstance(items, list) else list(items.values())
skip = {p["instance_id"] for p in json.load(open(seed))}
for p in items:
    if p["instance_id"] not in skip:
        print(p["instance_id"])
PYEOF
)
    if ((${#NEW_IDS[@]})); then
      EVAL_IDS_ARGS=(--instance_ids "${NEW_IDS[@]}")
      log "    eval ${#NEW_IDS[@]} new instances; reuse Dense-40 report"
    fi
  fi

  sudo -n -E "$PY" -m swebench.harness.run_evaluation \
    --dataset_name SWE-bench/SWE-bench_Lite \
    --predictions_path "$SAN" \
    "${EVAL_IDS_ARGS[@]}" \
    --max_workers "$EVAL_WORKERS" \
    --timeout 1800 \
    --run_id "${TAG}" \
    --report_dir results/swebench_reports \
    > "${LOGDIR}/eval.log" 2>&1
  grep -E 'Instances (resolved|submitted|with errors|unresolved)' "${LOGDIR}/eval.log" || :

  if [ "$SKIP40" = "1" ] && [ -s "$DENSE40_REPORT" ] && [ -s "$REPORT" ]; then
    log "=== merge Dense-40 report into ${REPORT}"
    sudo -n "$PY" - "$DENSE40_REPORT" "$REPORT" <<'PYEOF'
import json, sys
from pathlib import Path
old_p, new_p = Path(sys.argv[1]), Path(sys.argv[2])
old, new = json.loads(old_p.read_text()), json.loads(new_p.read_text())
ids40 = set(old.get("submitted_ids") or [])
list_keys = [
    "resolved_ids", "unresolved_ids", "error_ids", "completed_ids",
    "incomplete_ids", "empty_patch_ids", "infra_failure_ids",
    "ambiguous_failure_ids", "submitted_ids",
]
count_keys = {
    "resolved_ids": "resolved_instances",
    "unresolved_ids": "unresolved_instances",
    "error_ids": "error_instances",
    "completed_ids": "completed_instances",
    "incomplete_ids": "incomplete_instances",
    "empty_patch_ids": "empty_patch_instances",
    "infra_failure_ids": "infra_failure_instances",
    "ambiguous_failure_ids": "ambiguous_failure_instances",
    "submitted_ids": "submitted_instances",
}
for key in list_keys:
    keep = [i for i in (new.get(key) or []) if i not in ids40]
    extra = list(old.get(key) or [])
    new[key] = sorted(set(keep + extra))
for src, dst in count_keys.items():
    if dst in new:
        new[dst] = len(new.get(src) or [])
fr = dict(new.get("failure_reasons") or {})
for iid, reason in (old.get("failure_reasons") or {}).items():
    if iid in ids40:
        fr[iid] = reason
new["failure_reasons"] = fr
new_p.write_text(json.dumps(new, indent=2))
res, sub, err = len(new["resolved_ids"]), len(new["submitted_ids"]), len(new["error_ids"])
print(f"    merged 40+new -> resolved {res} submitted {sub} errors {err}", flush=True)
PYEOF
  fi
fi

log "########## ${METHOD} DONE"
