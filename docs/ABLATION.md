# Planned ablations

Mechanism (`docs/COST_LAMBDA_PARETO.md`) explains the frontier by varying
volume. Sensitivity (`docs/SENSITIVITY.md`) nudges the ball geometry `(k, d)`
around the frozen point. Ablation **turns a term off** and asks whether the
frozen method still works. Frozen point: `d=1`, `k=5`, `cost_lambda=0.25`,
`beta=0.35`, `B=8192`.

The three objective/structure knockouts the conversation locked are already
wired in `scripts/run_ablation.py`:

| paper name | config | script key | what is gone |
| --- | --- | --- | --- |
| Token shadow price | `fitness.ball.cost_lambda=0` | `no_cost_term` | F no longer pays for tokens; S should swell toward B |
| Query-conditioned support | `fitness.ball.beta=0` | `no_support_term` | F keeps Cov and cost only; isolated gold is no longer penalized for lacking neighbors |
| Anchors only | `optimizer.ball_radius=0` | `radius_zero` | feasible set is the k anchors; radius contribution is isolated |

`d=0` is not the same as Node-RepoMARO. Node-RepoMARO searches the
**unrestricted pool** (no ball). `d=0` still has a ball, of radius zero: only
the k anchors are admissible. The manuscript table already has
"Anchor-ball → unrestricted pool (Node-RepoMARO)" as its own row; keep
`radius_zero` as a separate row or it double-counts.

Do not fold these into the lambda or `(k, d)` grids. `lambda=0` is the
endpoint of the mechanism sweep, but as a *result* it is reported as an
ablation (term absent), not as a point on the frontier. `d=0` is the endpoint
of the d-sensitivity, but as a *result* it is "radius removed." Same
experiment can feed both displays; the captions differ.

## Other knockouts already in the script

Optimizer plumbing, not model claims. Report in the paper table only if they
move resolved or tokens by a meaningful amount; otherwise appendix.

- `no_budget_repair`
- `no_repo_local_search`
- `no_retrieval_seeding`
- `no_memory`
- `no_population_difference`
- `no_incremental_local` (engineering; should be bit-identical)
- `no_early_stopping`

Manuscript rows not yet a one-flag ablation: hypergraph→pairwise graph,
method units→monolithic classes, query needs→additive relevance,
exact-ΔF→relevance ordering. Those need code paths, not just a yaml toggle.

## Protocol

Same 40 instances, same pools, temperature 0.0, one generation pass each.
Record tokens, gold spans/files, and resolved. `lambda=0` will look like the
cheap-lambda mechanism points (high tokens, possible gold eviction); the
ablation caption is "without a token price," not "another frontier coordinate."
