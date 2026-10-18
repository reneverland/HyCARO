#!/usr/bin/env bash
# One GPU: run METHOD, then NEXT_METHOD. Both full Lite-300 pipelines.
set -uo pipefail
cd "$(dirname "$0")/.."
GPU="${GPU:?}"
METHOD="${METHOD:?}"
NEXT_METHOD="${NEXT_METHOD:?}"
SKIP40="${SKIP40:-0}"

echo "[$(date '+%F %T')] GPU${GPU} chain ${METHOD} -> ${NEXT_METHOD} SKIP40=${SKIP40}"
GPU="$GPU" METHOD="$METHOD" SKIP40="$SKIP40" TAG="lite300_${METHOD}" \
  ./scripts/run_lite300_method.sh
GPU="$GPU" METHOD="$NEXT_METHOD" SKIP40=0 TAG="lite300_${NEXT_METHOD}" \
  ./scripts/run_lite300_method.sh
echo "[$(date '+%F %T')] GPU${GPU} chain done"
