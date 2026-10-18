#!/usr/bin/env bash
# LocAgent search stack. Graph build uses the main venv; this venv is for localize.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="${VENV:-$ROOT/external/venvs/locagent}"
python3 -m venv "$VENV"
"$VENV/bin/pip" install -U pip
"$VENV/bin/pip" install \
  'litellm>=1.52' \
  'openai>=1.54' \
  'llama-index>=0.11' \
  'llama-index-retrievers-bm25>=0.4' \
  'bm25s' \
  'RapidFuzz' \
  'toml' \
  'datasets' \
  'networkx' \
  'matplotlib' \
  'libcst' \
  'tree-sitter==0.21.3' \
  'tree-sitter-languages' \
  'PyStemmer' \
  'tqdm' \
  'python-dotenv' \
  'httpx' \
  'tiktoken'
echo "locagent venv -> $VENV"
