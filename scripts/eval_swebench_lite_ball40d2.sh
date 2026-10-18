#!/usr/bin/env bash
# Official Docker eval for the ball_radius=2 run on SWE-bench Lite first 40.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export PYTHONUNBUFFERED=1
PY="${PY:-python}"
TAG="${TAG:-ball40d2}"
OUT="results/swebench_lite_8k_${TAG}_eval"
REPORT_DIR="results/swebench_reports"
PRED="results/swebench_lite_8k_${TAG}_predictions_RepoMARO.json"
SAN="$OUT/sanitized/swebench_lite_8k_${TAG}_predictions_RepoMARO.json"
mkdir -p "$OUT/sanitized" "$REPORT_DIR"

if [ ! -s "$SAN" ]; then
  echo "=== sanitize ==="
  "$PY" scripts/sanitize_swebench_preds.py "$PRED" \
    --out-dir "$OUT/sanitized" \
    --status-out "$OUT/sanitized/apply_status_${TAG}.jsonl"
fi

echo "=== docker eval ${TAG} ==="
sudo -n -E "$PY" -m swebench.harness.run_evaluation \
  --dataset_name SWE-bench/SWE-bench_Lite \
  --predictions_path "$SAN" \
  --max_workers 4 \
  --timeout 1800 \
  --run_id "${TAG}" \
  --report_dir "$REPORT_DIR"
echo "ALL_EVALS_DONE"
