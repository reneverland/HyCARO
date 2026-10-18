#!/usr/bin/env bash
# After RepoMARO retry generation finishes, run the remaining built-in
# optimization baselines on SWE-bench Lite (300). Selection is CPU-only
# once cache/pools_v3 exists; generation uses the same API as the retry.
set -uo pipefail
cd "$(dirname "$0")/.."
LOGDIR="${LOGDIR:-runlogs/lite300_baselines}"
mkdir -p "$LOGDIR"

wait_retry_gen() {
  echo "[$(date '+%F %T')] waiting for retry31 generation workers to exit"
  while pgrep -f 'run_swebench.py --config configs/ball_lam012.yaml --methods repomaro --budget 8192 --instance-ids' >/dev/null 2>&1; do
    sleep 20
  done
  echo "[$(date '+%F %T')] retry31 generation idle; starting optimization baselines"
}

wait_retry_gen
for METHOD in greedy binary_pso generic_maro; do
  echo "[$(date '+%F %T')] ===== ${METHOD} ====="
  METHOD="$METHOD" TAG="lite300_${METHOD}" \
    ./scripts/run_lite300_method.sh
done
echo "[$(date '+%F %T')] opt chain DONE"
