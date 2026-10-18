# Paper-to-code mapping

This package ships the frozen manuscript method: RepoMARO searches token-feasible
subsets inside a **k-anchor radius-d ball** of the repository hypergraph and
optimizes \(F = \alpha\,\mathrm{Cov} + \beta\,\mathrm{Supp} - \gamma\,\mathrm{Red} - \lambda\,\ell/B\).
Downstream coding models are never called inside search.

| Manuscript element | Code |
|---|---|
| Repository hypergraph over nine relations | `graph.py::Hypergraph`, `graph.py::RepositoryScanner` |
| Candidate pool C(q,R) | `candidate_pool.py::CandidatePoolBuilder` |
| Anchor-ball feasible region B(q,k,d) | `backbone.py::build_anchor_balls` |
| Relation confidence \(w_e\) | `backbone.py::RELATION_WEIGHT` |
| Token cost l_j and hard budget B | `tokenization.py`, `decoder.py` |
| Issue needs (lexical idf + dense) | `fitness.py::IssueCoverageFitness._build_needs` |
| Coverage Cov(q,S) with depth regulariser | `fitness.py::IssueCoverageFitness._coverage` |
| Structural support Supp(S\|q) | `fitness.py::IssueCoverageFitness._build_support`, `_support` |
| Redundancy Red(S) and token price | `fitness.py::IssueCoverageFitness._redundancy`, `evaluate_batch` |
| F(S) for the baselines | `fitness.py::SurrogateFitness` |
| Retrieval-seeded initialization in the ball | `optimizer.py::RepoMARO._initialize_population` |
| Ball-constrained decoding, ΔF-guided repair | `decoder.py::BallConstrainedDecoder` |
| Item-wise decoding (baselines) | `decoder.py::BudgetAwareDecoder` |
| Individual memory | `optimizer.py` (`mem_X`, `mem_masks`, `mem_fit`) |
| Adaptive differential search | `optimizer.py::_lambda_t`, `_batch_trials` |
| Add/drop/swap ranked by exact marginal ΔF per token | `optimizer.py::RepoMARO._generate_branch_proposals` |
| Fitness cache/deduplication | `fitness.py::FitnessCache` |
| Incremental local scoring | `fitness.py::FullEvalState`, `fitness.py::LocalSubsetState` |
| Downstream LLM only after selection | `downstream.py` and `scripts/run_swebench.py` |

## Method names

- `repomaro` / `RepoMARO`: anchor-ball search + Cov/Supp/Cost objective
  (`ball_repomaro` is accepted as an alias for older run scripts)
- Greedy-Utility, Binary PSO, Generic-MARO optimize the plain surrogate
  \(F_{\mathrm{node}}\) and act as optimizer-level controls

## Ablations

`run_ablation.py` removes components of the frozen method:

- no budget repair (`enable_budget_repair=false`): threshold only; infeasible portfolios receive `-inf`
- no repository-aware local search
- no incremental local scoring
- no retrieval-seeded initialization
- no individual memory
- no population-difference term in differential search
- no early stopping
- no token price (`fitness.ball.cost_lambda=0`)
- no structural support (`fitness.ball.beta=0`)
- anchors only (`ball_radius=0`), which isolates the contribution of the radius
