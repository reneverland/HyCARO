#!/usr/bin/env bash
set -euo pipefail
repomaro select \
  --repo examples/toy_repo \
  --issue "Coupon is not restored after a refund." \
  --instance-id toy-refund-1 \
  --config configs/pilot.yaml \
  --method repomaro \
  --budget 80 \
  --out results/toy_repomaro.json
cat results/toy_repomaro.json
