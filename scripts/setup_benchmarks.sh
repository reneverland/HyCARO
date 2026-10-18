#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-external}"
mkdir -p "$ROOT"

clone_or_update() {
  local url="$1" dir="$2"
  if [[ -d "$dir/.git" ]]; then
    echo "[exists] $dir"
  else
    git clone "$url" "$dir"
  fi
}

clone_or_update https://github.com/eyuansu62/agent-retrieval-bench.git "$ROOT/agent-retrieval-bench"
clone_or_update https://github.com/EuniAI/ContextBench.git "$ROOT/ContextBench"
clone_or_update https://github.com/SWE-bench/SWE-bench.git "$ROOT/SWE-bench"

python -m pip install -e "$ROOT/agent-retrieval-bench"
python -m pip install -e "$ROOT/SWE-bench"
# ContextBench is used through its own requirements/evaluator.
python -m pip install -r "$ROOT/ContextBench/requirements.txt"

echo "Benchmark sources installed under $ROOT"
echo "ARB data: cd $ROOT/agent-retrieval-bench && arb download-benchmark --all --local-dir data --force"
