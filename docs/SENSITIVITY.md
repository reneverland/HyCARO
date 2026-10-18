# Planned sensitivity analysis

Mechanism (token volume, information density, jumpy resolve under the 8192 cap)
lives in `docs/COST_LAMBDA_PARETO.md`. Ablations that *remove* a term live in
the paper's ablation table (`cost_lambda=0`, `beta=0`, `ball_radius=0`).
This note is only the leftover question: around the frozen point, does the
method fall off a cliff?

Frozen point: `ball_anchors=5`, `ball_radius=1`, `cost_lambda=0.25`,
`fitness.ball.beta=0.35`, `budget=8192`.

## What belongs here

One-at-a-time perturbations of the **structural feasible region** — the
anchor-ball `(k, d)`. Candidate pool and selector seed stay fixed. Downstream
temperature stays 0.0. Gold-in-ball / gold-in-selected can be measured without
a model and should be plotted next to resolved so a dip is interpretable.

### 1. Ball radius `d` (flagship)

This is the namesake hyperparameter. The ball is the feasible set; `d` is how
far it extends from the anchors.

| d | role | already have |
| --- | --- | --- |
| 0 | ablation (anchors only), not sensitivity | planned in PAPER_MAPPING |
| 1 | frozen | 40-instance resolved, several generations |
| 2 | one step out | 15/40 resolved, 6 apply fails, *worse* than d=1 |
| 3 | bound the other side | gold-in-ball only so far (`ball_recall_sweep.py`) |

d=2 already falsifies "bigger ball is better." Sensitivity has to show that,
not hide it: a larger feasible set lets F spend the extra volume on cheap
non-gold units (the same displacement the mechanism documents for cheap
`cost_lambda`). Expected pattern: gold-*reachable* in the ball rises with d,
gold-*selected* and resolved do not. If that holds, d=1 is a sweet point of
the feasible region, distinct from the token-price sweet point.

Cheap first pass: reuse `scripts/analysis/ball_recall_sweep.py` (feasible
region only, minutes). Downstream only for d that actually change reachability.

### 2. Anchor count `k`

Paired with d: k seeds the ball, d grows it. Frozen k=5. Sweep 3, 5, 8, 12
at d=1. Too small k and gold never enters the ball; too large k and the ball
approaches the whole pool and the method collapses toward unconstrained MARO.
Same two curves as d: gold-in-ball vs gold-in-selected.

## What does not belong here

- `cost_lambda` grid and the budget cap `B` (4K/8K/16K): both move token
  *volume*. They are the two controls of the mechanism frontier
  (`docs/COST_LAMBDA_PARETO.md`). `B` is the right wall, `cost_lambda` is where
  we sit under it. The manuscript still lists budget under sensitivity and the
  token shadow price as panel (c) of `fig:sensitivity`; both should move.
- `beta=0` / `cost_lambda=0` / `d=0`: ablations (term off).
- A ±20% nudge of `beta` around 0.35 is optional appendix material if we need
  to show F is not knife-edge; a full beta grid is composition-mechanism.
- Selector seeds: error bars, not a hyperparameter the user tunes.
- DE population / iterations: appendix if anything.

## Order

Finish the in-flight gold-coverage lambda grid (mechanism panel). Then the
cheap (k, d) reachability sweep. Downstream sensitivity only on d (and k if
reachability actually moves).
