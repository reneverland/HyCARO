#!/usr/bin/env bash
set -euo pipefail
PRED="${1:-results/swebench_predictions.json}"
RUN_ID="${2:-repomaro}"
WORKERS="${3:-8}"
run_python() {
  if docker info >/dev/null 2>&1; then
    "$@"
  elif sg docker -c 'docker info' >/dev/null 2>&1; then
    sg docker -c "$(printf '%q ' "$@")"
  else
    sudo -n -E "$@"
  fi
}
run_python python -m swebench.harness.run_evaluation \
  --dataset_name SWE-bench/SWE-bench_Lite \
  --predictions_path "$PRED" \
  --max_workers "$WORKERS" \
  --run_id "$RUN_ID"
