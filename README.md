# HyCARO: Hypergraph-Catalyzed Search for Budgeted Repository Context Selection

HyCARO selects code context for repository-level tasks under a token budget. It combines a typed repository hypergraph, relation-guided bounded search, and a fixed utility surrogate to explore the trade-off between context quality, token consumption, and selection overhead.

The context selector requires no additional training and makes no coding-model calls during search. It uses pretrained retrieval representations.

This repository accompanies the research manuscript by Renda Shi, School of Business, Macau University of Science and Technology.

Manuscript status: In preparation for submission to IEEE Access.

## Method overview

HyCARO has three components plus an acceleration mechanism:

1. **Typed repository hypergraph.** The repository is scanned into selectable units (functions, methods, class shells, length-capped module spans). Each software relation becomes one typed hyperedge that keeps the whole relation group together: class containment, imports, invocation, inheritance, tests, decorators, references, and state. Candidates come from lexical and dense retrieval plus bounded structural expansion. They are restricted to the units within radius `d` of the top-`k` relevance anchors.
2. **ARO-inspired search.** An Artificial Rabbits Optimization–style population, with differential-evolution preference updates, applies bounded joint replacements `S' = (S \ D) ∪ A` with `|A| ≤ a_max` and `|D| ≤ d_max`. Proposals cycle through three modes: exploration, support completion (finishing a witness inside one hyperedge), and deletion. A trial is accepted only if it improves the full surrogate.
3. **Training-free surrogate.**
   `F(S) = α·Cov(q,S) + β·Supp(S|q) − η·C(S)/B − γ·R_red(S)`.
   - Coverage scores the issue's identifier needs and a semantic need.
   - Support uses the maximum role-compatible witness, so `0 ≤ Supp ≤ Cov ≤ 1`.
   - The token price rewards savings below the budget.
4. **Acceleration.** Exact incremental evaluation recomputes only the affected region of each joint move, and a query-local subset cache stores scored selections. Together they give fast selection with the same acceptance decisions as full rescoring.

## Installation

Python 3.10+:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
pytest -q
```

Dense retrieval uses `Qwen/Qwen3-Embedding-4B` (enabled in `configs/hycaro.yaml`). If dense retrieval is disabled, a deterministic hashing fallback is used; it is only for development.

To try the selector without a GPU or a model download, use `configs/hycaro_cpu_demo.yaml`. It is `configs/hycaro.yaml` with dense retrieval switched off, so its candidate pools and numbers differ from the paper.

## Usage

Select context for one issue:

```bash
repomaro select \
  --repo examples/toy_repo \
  --issue "Refunds do not reactivate used coupons" \
  --config configs/hycaro.yaml \
  --method hycaro \
  --budget 8192
```

Benchmark adapters (selection only, no coding model):

```bash
python scripts/run_contextbench.py --method hycaro --config configs/hycaro.yaml --budget 8192 --out results/cb_hycaro.jsonl
python scripts/run_swebench.py     --method hycaro --config configs/hycaro.yaml --budget 8192 --out results/swe_hycaro.jsonl
python scripts/run_arb.py          --methods hycaro --config configs/hycaro.yaml --budget 8192 --out results/arb_hycaro.jsonl
```

For SWE-bench downstream patch generation, add `--generate`. The API key is read only from the `OPENAI_API_KEY` environment variable.

## Code layout

```text
configs/hycaro.yaml          configuration reported in the paper (seed 11, budget 8192)
src/repomaro/
  hycaro.py                  HyCARO: witness index, surrogate, ARO-inspired search,
                             incremental evaluation, relation-guided greedy control
  graph.py                   AST scanner and typed hypergraph
  candidate_pool.py          lexical / dense / structural candidate pool
  backbone.py                anchor balls and hop distances
  fitness.py                 coverage needs and surrogate utilities
  controlled.py              controlled selectors (RepoGraph, Greedy+1swap, Two-unit, DE+repair, ...)
  baselines.py               retrieval baselines (BM25, dense, RRF, MMR)
  baselines_external.py      wrappers for external baselines (e.g. LongCodeZip coarse)
  runner.py                  method dispatch
  adapters/                  ARB / ContextBench / SWE-bench adapters
  downstream.py              downstream patch generation
tests/                       unit tests, including incremental-vs-full evaluation checks
```

The package name `repomaro` is kept from an earlier project name. The selector reported in the paper is `--method hycaro`.

## Key configuration

These are the HyCARO search parameters in `configs/hycaro.yaml`:

| key | value |
| --- | --- |
| `hycaro_population` | 6 |
| `hycaro_iterations` | 3 |
| `hycaro_proposals` (attempts per rabbit, `P`) | 3 |
| `hycaro_a_max` / `hycaro_d_max` | 3 / 3 |
| `hycaro_stop` (`K_stop`) | 2 |
| `hycaro_crossover` | 0.8 |
| `hycaro_lambda_min` / `hycaro_lambda_max` | 0.2 / 0.8 |
| `hycaro_time_budget_s` | 20 |

The surrogate coefficients are in `fitness.ball` (`alpha=1.0`, `beta=0.35`, `gamma=0`, `cost_lambda=0.25`).

## Relation types

The scanner in `graph.py` uses its own relation names. They map onto the relation types in the paper as follows:

| paper | code | support witness |
| --- | --- | --- |
| class containment | `encloses` | yes |
| imports | `imports` | yes |
| invocation | `calls` | yes |
| inheritance | `inherits` | yes |
| tests | `tests` | yes |
| decorators | `decorates` | yes |
| references | `references` | yes |
| state | `state` | no: the edge does not record which unit reads and which writes |
| (none) | `contains` | no: file-level packaging; not walked when measuring hop distance |

## Ablations

The variants in the paper's ablation table are run with `scripts/run_contextbench.py` and either a method name (`seed_only`, `relation_greedy`) or a one-line config variant (`configs/ablation_no_support.yaml`, `configs/ablation_pairwise.yaml`). See `docs/ABLATION.md` for the commands.

## Benchmarks

Third-party benchmark data and repository snapshots are not bundled. Install the official implementations with `bash scripts/setup_benchmarks.sh`; see `docs/BENCHMARK_SETUP.md`. Benchmark results are not included in this repository.

## License

Apache License 2.0; see `LICENSE`.
