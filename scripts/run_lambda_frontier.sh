#!/bin/bash
# Sweep cost_lambda across the token/resolved frontier, one grid point at a time.
#
# 0.15 came in at 40.0% for 3948 tokens against the frozen 0.25's 41.7% for 3080,
# so the curve is flat or slightly backwards between them. The information is at
# the ends: whether cheap tokens buy resolved rate below 0.08, and where the
# efficiency claim collapses above 0.35.
#
# Each point is a full 40-instance generation pass (8 shards) followed by the
# official Docker evaluation. Points run sequentially: 8 shards already saturate
# the downstream API, and the evaluation wants the machine to itself.
set -euo pipefail
cd "$(dirname "$0")/.."

LAMBDAS="${LAMBDAS:-005 008 035 050}"
SHARDS="${SHARDS:-8}"
PER_SHARD="${PER_SHARD:-5}"

for v in ${LAMBDAS}; do
  tag="lam${v}"
  cfg="configs/ball_${tag}.yaml"
  if [ ! -f "${cfg}" ]; then
    echo "!!! missing ${cfg}, skipping"
    continue
  fi
  echo "############ ${tag} ($(grep -E '^\s+cost_lambda:' "${cfg}" | tr -d ' ')) $(date '+%F %T')"

  if [ -f "results/swebench_reports/gpt-5.6-luna+RepoMARO+s0.${tag}.json" ]; then
    echo "    report already exists, skipping"
    continue
  fi

  echo "=== generate ${tag} ==="
  existing=0
  shopt -s nullglob
  _jsonls=(results/swebench_lite_8k_${tag}_s*.jsonl)
  shopt -u nullglob
  if ((${#_jsonls[@]})); then
    existing=$(cat "${_jsonls[@]}" | wc -l | tr -d '[:space:]')
  fi
  existing=${existing:-0}
  if [ "${existing:-0}" -ge $((SHARDS * PER_SHARD)) ]; then
    echo "    generation already complete (${existing} rows), skip"
  else
    for i in $(seq 0 $((SHARDS - 1))); do
      off=$((i * PER_SHARD))
      CONFIG="${cfg}" OFFSET="${off}" LIMIT="${PER_SHARD}" TAG="${tag}_s$i" \
        ./scripts/run_swebench_lite_ball40v3_repro.sh > "runlogs/${tag}_s$i.log" 2>&1 &
    done
    wait
    echo "=== generation done ${tag} $(date '+%F %T') ==="
    total=$(cat results/swebench_lite_8k_${tag}_s*.jsonl 2>/dev/null | wc -l)
    echo "    rows: ${total}/$((SHARDS * PER_SHARD))"
  fi

  echo "=== merge ${tag} ==="
  python3 - "${tag}" <<'PY'
import json, glob, sys
tag = sys.argv[1]
preds, seen = [], set()
for f in sorted(glob.glob(f"results/swebench_lite_8k_{tag}_s*_predictions_RepoMARO.json")):
    data = json.load(open(f))
    items = data if isinstance(data, list) else [data] if "instance_id" in data else list(data.values())
    for p in items:
        if p["instance_id"] in seen:
            continue
        seen.add(p["instance_id"])
        preds.append(p)
preds.sort(key=lambda p: p["instance_id"])
out = f"results/swebench_lite_8k_{tag}_predictions_RepoMARO.json"
json.dump(preds, open(out, "w"), indent=2)
print(f"    merged {len(preds)} -> {out}")
PY

  echo "=== evaluate ${tag} ==="
  TAG="${tag}" ./scripts/eval_swebench_lite_ball40d2.sh > "runlogs/eval_${tag}.log" 2>&1 || {
    echo "!!! eval failed for ${tag}, see runlogs/eval_${tag}.log"
    continue
  }
  tail -n 12 "runlogs/eval_${tag}.log" | grep -E 'Instances (resolved|submitted|with errors)' || :
done

echo "############ FRONTIER SWEEP DONE $(date '+%F %T')"
