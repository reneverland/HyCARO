# Ablations reported in the manuscript

Table 4 of the manuscript compares full HyCARO with four variants on ContextBench. All variants share the candidate pools, token costs, seed 11, the 8,192-token budget, and the 20-second limit of `configs/hycaro.yaml`.

| Variant | What changes | How to run |
| --- | --- | --- |
| Full HyCARO | nothing | `--method hycaro --config configs/hycaro.yaml` |
| No support reward | `fitness.ball.beta = 0`; coverage, token cost, and relation-guided search stay | `--method hycaro --config configs/ablation_no_support.yaml` |
| Seed-only | the shared initial contexts are packed and scored; no preference updates or replacements | `--method seed_only --config configs/hycaro.yaml` |
| Pairwise proposals | pair-based proposals replace hyperedge-group proposals; witnesses, surrogate, and population search stay | `--method hycaro --config configs/ablation_pairwise.yaml` |
| Relation-guided greedy | a single incumbent takes the best improving relation action at each step | `--method relation_greedy --config configs/hycaro.yaml` |

Each row is one call of `scripts/run_contextbench.py`, for example:

```bash
python scripts/run_contextbench.py --method hycaro --config configs/ablation_pairwise.yaml \
  --budget 8192 --out results/cb_ablation_pairwise.jsonl
```

The two config-based variants write records under the method name `HyCARO`, so give each its own `--out` file. Score a file with:

```bash
python scripts/analysis/contextbench_f1.py results/cb_ablation_pairwise.jsonl
```

`scripts/run_ablation.py` runs a broader set of knockouts on a single local repository and issue; those knockouts are development aids and are not reported in the manuscript.
