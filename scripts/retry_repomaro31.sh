#!/usr/bin/env bash
# Re-generate the 31 RepoMARO Lite-300 instances that died on API
# "Connection error", evaluate them, fold into the frozen report.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"

export LD_LIBRARY_PATH="${ROOT}/.cuda_compat:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export GIT_TERMINAL_PROMPT=0
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

PY="${ROOT}/.venv/bin/python"
OUTDIR="results/lite300"
LOGDIR="runlogs/lite300"
TAG="lite300_lam012_retry31"
CONFIG="${CONFIG:-configs/ball_lam012.yaml}"
POOL_CACHE="${POOL_CACHE:-cache/pools_v3}"
ORIG_PRED="${OUTDIR}/lite300_lam012_predictions_RepoMARO.json"
ORIG_REPORT="results/swebench_reports/gpt-5.6-luna+RepoMARO+s0.lite300_lam012.json"
mkdir -p "$OUTDIR" "$LOGDIR" results/swebench_reports

IDS=(
  django__django-11099
  django__django-11910
  django__django-11964
  django__django-11999
  django__django-12856
  django__django-12908
  django__django-13590
  django__django-14155
  django__django-14382
  django__django-15213
  django__django-15851
  django__django-17051
  django__django-17087
  matplotlib__matplotlib-23913
  matplotlib__matplotlib-23964
  matplotlib__matplotlib-25498
  pytest-dev__pytest-7373
  pytest-dev__pytest-7432
  scikit-learn__scikit-learn-13496
  sphinx-doc__sphinx-10325
  sphinx-doc__sphinx-10451
  sphinx-doc__sphinx-8801
  sympy__sympy-11400
  sympy__sympy-12481
  sympy__sympy-15308
  sympy__sympy-15345
  sympy__sympy-17655
  sympy__sympy-18057
  sympy__sympy-20442
  sympy__sympy-20590
  sympy__sympy-23262
)

log() { echo "[$(date '+%F %T')] $*"; }
log "########## retry ${#IDS[@]} empty-patch RepoMARO instances"

SHARDS=4
n=${#IDS[@]}
per=$(( (n + SHARDS - 1) / SHARDS ))
for i in $(seq 0 $((SHARDS - 1))); do
  start=$((i * per))
  [ "$start" -ge "$n" ] && continue
  chunk=("${IDS[@]:start:per}")
  "$PY" scripts/run_swebench.py \
    --config "$CONFIG" \
    --methods repomaro \
    --budget 8192 \
    --instance-ids "${chunk[@]}" \
    --generate \
    --n-samples 1 \
    --pool-cache "$POOL_CACHE" \
    --out "${OUTDIR}/${TAG}_s${i}.jsonl" \
    --predictions "${OUTDIR}/${TAG}_s${i}_predictions.json" \
    > "${LOGDIR}/${TAG}_s${i}.log" 2>&1 &
done
wait

log "=== merge retry predictions into ${ORIG_PRED}"
RETRY_MERGED="${OUTDIR}/${TAG}_predictions_RepoMARO.json"
"$PY" - "${OUTDIR}/${TAG}_s*_predictions_RepoMARO.json" "$RETRY_MERGED" "$ORIG_PRED" <<'PY'
import glob, json, sys
from pathlib import Path
pattern, retry_out, orig = sys.argv[1:4]
preds, seen = [], set()
for f in sorted(glob.glob(pattern)):
    data = json.load(open(f))
    items = data if isinstance(data, list) else list(data.values())
    for p in items:
        if p["instance_id"] in seen or not p.get("model_patch"):
            continue
        seen.add(p["instance_id"])
        preds.append(p)
preds.sort(key=lambda p: p["instance_id"])
json.dump(preds, open(retry_out, "w"), indent=2)
print(f"    retry preds {len(preds)} -> {retry_out}", flush=True)
if Path(orig).exists() and preds:
    old = json.load(open(orig))
    items = old if isinstance(old, list) else list(old.values())
    by = {p["instance_id"]: p for p in items}
    for p in preds:
        by[p["instance_id"]] = p
    merged = sorted(by.values(), key=lambda p: p["instance_id"])
    json.dump(merged, open(orig, "w"), indent=2)
    print(f"    combined {len(merged)} -> {orig}", flush=True)
PY

n_ok=$("$PY" -c "import json; print(len(json.load(open('${RETRY_MERGED}'))))")
log "    got ${n_ok}/${#IDS[@]} new patches"
if [ "${n_ok:-0}" -eq 0 ]; then
  log "!!! no new patches, skip eval"
  exit 1
fi

log "=== sanitize + docker eval retry"
EVDIR="${OUTDIR}/${TAG}_eval"
SAN="${EVDIR}/sanitized/${TAG}_predictions_RepoMARO.json"
mkdir -p "${EVDIR}/sanitized"
"$PY" scripts/sanitize_swebench_preds.py "$RETRY_MERGED" \
  --out-dir "${EVDIR}/sanitized" \
  --status-out "${EVDIR}/sanitized/apply_status_${TAG}.jsonl" \
  > "${LOGDIR}/${TAG}_sanitize.log" 2>&1

mapfile -t EVAL_IDS < <("$PY" -c "import json; [print(p['instance_id']) for p in json.load(open('${SAN}'))]")
sudo -n -E "$PY" -m swebench.harness.run_evaluation \
  --dataset_name SWE-bench/SWE-bench_Lite \
  --predictions_path "$SAN" \
  --instance_ids "${EVAL_IDS[@]}" \
  --max_workers 8 \
  --timeout 1800 \
  --run_id "${TAG}" \
  --report_dir results/swebench_reports \
  > "${LOGDIR}/${TAG}_eval.log" 2>&1
grep -E 'Instances (resolved|submitted|with errors|unresolved)' "${LOGDIR}/${TAG}_eval.log" || :

NEW_REPORT="results/swebench_reports/gpt-5.6-luna+RepoMARO+s0.${TAG}.json"
if [ -s "$NEW_REPORT" ] && [ -s "$ORIG_REPORT" ]; then
  log "=== merge into original report"
  sudo -n "$PY" - "$NEW_REPORT" "$ORIG_REPORT" <<'PY'
import json, sys
from pathlib import Path
new_p, orig_p = Path(sys.argv[1]), Path(sys.argv[2])
new, orig = json.loads(new_p.read_text()), json.loads(orig_p.read_text())
bak = orig_p.with_suffix(".json.bak_before_retry31")
if not bak.exists():
    bak.write_text(json.dumps(orig, indent=2))
touched = set()
for key in ("resolved_ids","unresolved_ids","error_ids","completed_ids",
            "incomplete_ids","empty_patch_ids","infra_failure_ids",
            "ambiguous_failure_ids","submitted_ids"):
    touched |= set(new.get(key) or [])
# only IDs that actually appear in the retry report buckets besides the
# harness copying the full prediction file into submitted_ids
retry_ids = set()
for key in ("resolved_ids","unresolved_ids","error_ids","completed_ids",
            "incomplete_ids","empty_patch_ids","infra_failure_ids",
            "ambiguous_failure_ids"):
    retry_ids |= set(new.get(key) or [])
if not retry_ids:
    retry_ids = set(new.get("submitted_ids") or [])
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
    keep = [i for i in (orig.get(key) or []) if i not in retry_ids]
    extra = [i for i in (new.get(key) or []) if i in retry_ids]
    orig[key] = sorted(set(keep + extra))
# these IDs are no longer incomplete once submitted
orig["incomplete_ids"] = [i for i in (orig.get("incomplete_ids") or []) if i not in retry_ids]
orig["submitted_ids"] = sorted(set(orig.get("submitted_ids") or []) | retry_ids)
orig["completed_ids"] = sorted(set(orig.get("completed_ids") or []) | (retry_ids - set(orig.get("error_ids") or [])))
for src, dst in count_keys.items():
    if dst in orig:
        orig[dst] = len(orig.get(src) or [])
fr = dict(orig.get("failure_reasons") or {})
for iid in retry_ids:
    fr.pop(iid, None)
for iid, reason in (new.get("failure_reasons") or {}).items():
    if iid in retry_ids:
        fr[iid] = reason
orig["failure_reasons"] = fr
orig_p.write_text(json.dumps(orig, indent=2))
res, sub, err = len(orig["resolved_ids"]), len(orig["submitted_ids"]), len(orig["error_ids"])
print(f"    resolved {res} submitted {sub} errors {err}  raw {100*res/sub:.1f}%  res/(sub-err) {res}/{sub-err}", flush=True)
PY
fi

log "########## retry31 DONE"
