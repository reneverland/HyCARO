# Benchmark setup

## 1. Agent Retrieval Bench (ARB)

Official repository: `https://github.com/eyuansu62/agent-retrieval-bench`

The released benchmark has 427 samples; the four positive-retrieval workflows contain 345 samples. Its canonical BCY metric uses 4K/8K/16K/32K token budgets. HyCARO uses file-level units on ARB to match this protocol.

```bash
bash scripts/setup_benchmarks.sh
cd external/agent-retrieval-bench
arb download-benchmark --all --local-dir data --force
cd ../..

python scripts/run_arb.py \
  --samples external/agent-retrieval-bench/data/benchmark/v2_code2test/samples.jsonl \
            external/agent-retrieval-bench/data/benchmark/v2_comment2context/samples.jsonl \
            external/agent-retrieval-bench/data/benchmark/v2_trace2code/samples.jsonl \
            external/agent-retrieval-bench/data/benchmark/v2_edit2ripple/samples.jsonl \
  --budget 8192 \
  --config configs/hycaro.yaml
```

Release folder names can change; use `find external/agent-retrieval-bench/data/benchmark -name samples.jsonl` to inspect the installed release.

## 2. ContextBench Verified (500 tasks)

Official repository: `https://github.com/EuniAI/ContextBench`
Dataset: `Contextbench/ContextBench`, configuration `contextbench_verified`.

```bash
pip install -e '.[benchmarks]'
python scripts/run_contextbench.py --method hycaro --budget 8192
```

The script writes the unified ContextBench prediction format:

```json
{
  "instance_id": "...",
  "traj_data": {
    "pred_steps": [{"files": [], "spans": {}, "symbols": {}}],
    "pred_files": [],
    "pred_spans": {}
  },
  "model_patch": ""
}
```

Evaluate it with the official ContextBench package using `python -m contextbench.evaluate` and the verified gold parquet.

## 3. SWE-bench Lite (300 tasks)

Official evaluation dataset: `princeton-nlp/SWE-bench_Lite`.

Selection-only pilot:

```bash
python scripts/run_swebench.py --method hycaro --limit 30 --budget 8192
```

Downstream patch generation (requires `OPENAI_API_KEY`):

```bash
export OPENAI_API_KEY=...
python scripts/run_swebench.py --method hycaro --limit 30 --budget 8192 --generate
```

Official evaluation:

```bash
bash scripts/evaluate_swebench_official.sh results/swebench_predictions.json hycaro_pilot 8
```

The paper-scale run should first validate a 30-task pilot, then expand to the planned controlled set.

## 4. Dense retrieval

The paper configuration `configs/hycaro.yaml`, which every benchmark script uses by default, enables `Qwen/Qwen3-Embedding-4B` on CUDA. Install the extras with:

```bash
pip install -e '.[dense,tokens,benchmarks]'
```

To run without a GPU during development, set `candidate_pool.dense.enabled: false`; the resulting hashing fallback does not reproduce the paper's candidate pools.
