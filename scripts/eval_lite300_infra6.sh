#!/usr/bin/env bash
# Re-evaluate the 6 Lite-300 instances that died on image pull / Docker Hub,
# then fold any new verdicts back into the frozen-run report.
set -uo pipefail
cd "$(dirname "$0")/.."

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:10809}"
export HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:10809}"
export https_proxy="${https_proxy:-$HTTPS_PROXY}"
export http_proxy="${http_proxy:-$HTTP_PROXY}"
export PYTHONUNBUFFERED=1

PY="/home/yuyang/repomaro/.venv/bin/python"
SAN="results/lite300/lite300_lam012_eval/sanitized/lite300_lam012_predictions_RepoMARO.json"
ORIG="results/swebench_reports/gpt-5.6-luna+RepoMARO+s0.lite300_lam012.json"
NEW_RUN="lite300_lam012_infra6"
REPORT_DIR="results/swebench_reports"

IDS=(
  matplotlib__matplotlib-24265
  mwaskom__seaborn-2848
  mwaskom__seaborn-3407
  scikit-learn__scikit-learn-13779
  scikit-learn__scikit-learn-14983
  scikit-learn__scikit-learn-15512
)

echo "[$(date '+%F %T')] infra6 start: ${#IDS[@]} instances"

echo "=== docker eval infra6 ==="
sudo -n -E "$PY" -m swebench.harness.run_evaluation \
  --dataset_name SWE-bench/SWE-bench_Lite \
  --predictions_path "$SAN" \
  --instance_ids "${IDS[@]}" \
  --max_workers 4 \
  --timeout 1800 \
  --run_id "${NEW_RUN}" \
  --report_dir "$REPORT_DIR"

echo "=== merge into original report ==="
"$PY" - "$ORIG" "$REPORT_DIR/gpt-5.6-luna+RepoMARO+s0.${NEW_RUN}.json" <<'PY'
import json, sys
from pathlib import Path

orig_path, new_path = Path(sys.argv[1]), Path(sys.argv[2])
if not new_path.exists():
    print(f"no new report {new_path}", flush=True)
    sys.exit(1)
orig = json.loads(orig_path.read_text())
new = json.loads(new_path.read_text())
backup = orig_path.with_suffix(".json.bak_before_infra6")
if not backup.exists():
    backup.write_text(json.dumps(orig, indent=2))
    print(f"backed up -> {backup}", flush=True)

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

touched = set()
for key in list_keys:
    touched |= set(new.get(key) or [])

# drop these ids from every bucket, then add back from the retry report
for key in list_keys:
    keep = [i for i in (orig.get(key) or []) if i not in touched]
    extra = [i for i in (new.get(key) or []) if i in touched]
    orig[key] = sorted(set(keep + extra))

for src, dst in count_keys.items():
    if dst in orig:
        orig[dst] = len(orig.get(src) or [])

# failure_reasons: replace entries for retried ids
fr = dict(orig.get("failure_reasons") or {})
for iid in touched:
    fr.pop(iid, None)
for iid, reason in (new.get("failure_reasons") or {}).items():
    fr[iid] = reason
orig["failure_reasons"] = fr

orig_path.write_text(json.dumps(orig, indent=2))
res, sub, err = len(orig["resolved_ids"]), len(orig["submitted_ids"]), len(orig["error_ids"])
print(
    f"merged {sorted(touched)} -> {orig_path}\n"
    f"resolved {res} submitted {sub} errors {err}\n"
    f"resolved/(sub-err) {res}/{sub-err} = {100*res/max(sub-err,1):.1f}%",
    flush=True,
)
PY

echo "[$(date '+%F %T')] infra6 done"
