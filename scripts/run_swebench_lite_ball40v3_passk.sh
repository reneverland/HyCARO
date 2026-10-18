#!/usr/bin/env bash
# RepoMARO (hypergraph + Cov/Supp) pass@3 / pass@5: extra Luna samples s1–s4
# on the same 40 selected contexts as ball40v3 s0.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT="results/swebench_lite_8k_ball40v3_passk"
mkdir -p "$OUT" results/swebench_reports
STATUS="$OUT/STATUS.txt"
status() { echo "[$(date '+%F %T')] $*" | tee -a "$STATUS"; }

export PYTHONUNBUFFERED=1
export GIT_TERMINAL_PROMPT=0
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:10809}"
export HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:10809}"
export https_proxy="${https_proxy:-$HTTPS_PROXY}"
export http_proxy="${http_proxy:-$HTTP_PROXY}"
export CUDA_VISIBLE_DEVICES=""
PY="${ROOT}/.venv/bin/python"

status "generate s1-s4 for RepoMARO (temperature=0.4, resumable)"
"$PY" scripts/gen_swebench_passk_samples.py \
  --jsonl results/swebench_lite_8k_ball40v3.jsonl \
  --methods RepoMARO \
  --out-dir "$OUT" \
  --sample-start 1 \
  --sample-end 4 \
  --temperature 0.4 \
  | tee -a "$OUT/gen.log"
n_extra=$(grep -c . "$OUT/extra_samples.jsonl" 2>/dev/null || echo 0)
status "extra_samples.jsonl lines=$n_extra"
if [ "${n_extra:-0}" -lt 160 ]; then
  status "incomplete extra samples ($n_extra/160); stop before eval"
  exit 1
fi

status "sanitize s1-s4"
mkdir -p "$OUT/sanitized"
shopt -s nullglob
pred_files=("$OUT"/RepoMARO_s{1,2,3,4}.json)
if ((${#pred_files[@]} == 0)); then
  status "no extra prediction files"
  exit 1
fi
"$PY" scripts/sanitize_swebench_preds.py "${pred_files[@]}" \
  --out-dir "$OUT/sanitized" \
  --status-out "$OUT/sanitized/apply_status.jsonl" \
  | tee "$OUT/sanitize.log"

run_eval() {
  local pred="$1"
  local run_id="$2"
  status "eval $run_id"
  sudo -n -E "$PY" -m swebench.harness.run_evaluation \
    --dataset_name SWE-bench/SWE-bench_Lite \
    --predictions_path "$pred" \
    --max_workers 4 \
    --timeout 1800 \
    --run_id "$run_id" \
    --report_dir results/swebench_reports
}

for si in 1 2 3 4; do
  pred="$OUT/sanitized/RepoMARO_s${si}.json"
  [ -f "$pred" ] || continue
  run_eval "$pred" "ball40v3_s${si}" || status "eval failed s${si}"
done

status "summarize pass@1 / pass@3 / pass@5"
mapfile -t reports < <(ls \
  results/swebench_reports/*RepoMARO+s0.ball40v3.json \
  results/swebench_reports/*RepoMARO+s{1,2,3,4}.ball40v3_s{1,2,3,4}.json \
  2>/dev/null || true)
"$PY" scripts/summarize_swebench_passk.py "${reports[@]}" --out "$OUT/passk.json" \
  | tee "$OUT/passk.txt"

status "ALL_PASSK_DONE"
