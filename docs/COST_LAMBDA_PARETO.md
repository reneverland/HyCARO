# Mechanism: token volume, information density, and a jumpy resolve frontier

Status: mechanism sketched from existing runs; gold-coverage grid in flight;
resolved-rate grid still planned. This is not a hyperparameter hunt. The frozen
operating point stays at `cost_lambda = 0.25` until the mechanism says otherwise.

## What this analysis is

It is a mechanism that explains the shape of a two-objective frontier, not a
third objective and not a one-knob sensitivity table.

The two objectives a user actually trades are **cost** (selected tokens, or the
prompt tokens they pay for) and **utility** (resolved rate). `cost_lambda` is the
control that slides us along that plane. What it does *not* control directly is
resolved rate. The causal chain is:

```
cost_lambda  -->  volume of S          (how many tokens we emit)
F = Cov+Supp-Red  -->  composition of S  (which units fill that volume)
composition        -->  information density  (gold-relevant mass / tokens)
information + noise -->  resolved           (binary, per instance, stochastic)
```

Information density is a **mediating variable**, not a third axis you can pick.
A three-dimensional Pareto would require three independent objectives. Here the
third quantity *causes* the second: you cannot choose high information and low
resolved, or low information and high resolved, except by noise. Plotting
(tokens, gold coverage, resolved) as a surface would double-count the same
effect and invite a reader to treat gold coverage as a knob they can turn.

The decision problem is therefore the ordinary two-objective frontier
(tokens, resolved) under a hard cap. Two controls move us on that plane:
`cost_lambda` is the token price (where we sit under the wall); the budget `B`
is the wall itself. Sweeping 4K / 8K / 16K at frozen lambda is the same
mechanism as sweeping lambda at frozen B, not a stability check of the ball.
Jumps and reversals are sampling artefacts of this frontier, concentrated near
the cap: once gold-file recall has saturated, extra volume is distractors (and
can evict gold), so a 40-instance sample can show more tokens with fewer
resolved. The sweet spot we need is the leftmost point on this capped frontier
that still matches Dense — empirically already around 3000 tokens at B=8192,
not 8000.

What *is* worth plotting as two panels of one figure:

1. Mechanism panel: tokens vs gold coverage (or gold-free proxies: Cov mass,
   support mass, fraction of selected units that are ball-anchors / need-bearers).
   This curve can be measured without a downstream model.
2. Frontier panel: tokens vs resolved. This curve is the claim. It will look
   jumpy on 40 instances and only settle on 300, because the production function
   in panel 2 is discrete.

Calling the whole thing "sensitivity analysis" undersells it: a sensitivity
table asks how much resolved moves per 0.05 of lambda. The interesting fact is
*why* that derivative is near zero over a wide interval and then jumps. Calling
it a "3D Pareto" oversells it: the third coordinate is the explanation, not a
competing goal. The paper language should be **mechanism-informed frontier**.

## Why extra tokens help, but not continuously

A selected unit is in or out. A gold span is covered or not. An instance is
resolved or not. Nothing in that stack is a slope.

Lowering `cost_lambda` expands the feasible volume of S. The extra slots are
filled by whatever F ranks next: high coverage, cheap, well-supported units
that often are *not* the gold span. File-level gold recall saturates early
(30/40 at lambda 0.25 versus 31/40 for Dense). Span-level recall still creeps
up (26/60 -> 28/60 -> 33/60 from 0.25 to 0.15 to Dense). So the extra tokens
are mostly distractors, with a thin tail of additional gold.

Resolved only moves when one of those extra gold units is *decisive* for an
instance the downstream model would otherwise miss. That is a rare event on 40
instances, so a single sample can easily show resolved falling while tokens
rise (0.25 -> 0.15 was 17/40 -> 16/40, a three-instance flip). That is not
evidence that tokens do not help. It is evidence that the help is a step
function plus downstream noise, and 40 instances cannot resolve a one-instance
step.

The smoke test makes the displacement failure mode concrete: `astropy-14182`
covers 3/3 gold spans at lambda 0.25 (4965 tokens) and only 2/3 at lambda 0.05
(8144 tokens). Expanding S can *evict* gold when F's next-best units crowd it
out under a near-zero token price. The frontier is therefore not guaranteed to
be monotone even in expectation; the mechanism has to show whether eviction is
rare or systematic.

## Claim to support later

There exists a region of `cost_lambda` where information density is high and
token volume is well below Dense, and in that region resolved matches or beats
Dense. Moving left (cheaper tokens, larger S) should raise resolved in
expectation, in jumps of one instance at a time, with diminishing hits per
token as gold file recall saturates. Moving right (expensive tokens, smaller S)
should eventually drop gold spans that were load-bearing and resolved should
collapse.

Dense is a single point that spends the whole budget at lower density. If the
mechanism is right, some of our operating points sit to its left (fewer
tokens) at comparable resolved, and the points that spend toward its token
count should not lose to it on resolved once the jumps are averaged out.

## Already measured (same 40 instances, budget 8192, d=1, seed 42)

Gold coverage does not need a model. Resolved does. The two must not be mixed
in one table as if they were the same experiment.

Full selector grid (`results/lambda_gold_sweep.jsonl`, 40 x 8 lambdas, no
downstream). File-level gold recall is **bit-identical at 30/40 for every
lambda**. The 10 missed gold files are not a token-price problem.

| cost_lambda | mean tokens | budget | gold spans | gold files | full-cover inst |
| --- | --- | --- | --- | --- | --- |
| 0.05 | 5772 | 70.5% | 29/60 | 30/40 | 17/40 |
| 0.08 | 5052 | 61.7% | **30/60** | 30/40 | **18/40** |
| 0.12 | 4336 | 52.9% | 26/60 | 30/40 | 15/40 |
| 0.15 | 3948 | 48.2% | 28/60 | 30/40 | 17/40 |
| 0.20 | 3466 | 42.3% | 27/60 | 30/40 | 16/40 |
| 0.25 | 3080 | 37.6% | 26/60 | 30/40 | 15/40 |
| 0.35 | 2570 | 31.4% | 26/60 | 30/40 | 15/40 |
| 0.50 | 1939 | 23.7% | 26/60 | 30/40 | 15/40 |
| Dense (reference) | 8169 | 99.7% | 33/60 | 31/40 | 20/40 |

Versus frozen 0.25, cheaper lambda changes gold spans on at most 4/40
instances (0.08: +3/0; 0.05: +3/−1). The −1 is `astropy-14182`: 3/3 gold
spans at 0.25–0.50, 2/3 at 0.05 with 8144 tokens — eviction at the cap.
From 0.25 through 0.50 every instance has the same span-hit count; 0.50
buys the same gold for 37% fewer tokens.

Downstream resolved, still not a grid:

| cost_lambda | resolved | note |
| --- | --- | --- |
| 0.08 | never evaluated | killed at 25/40 |
| 0.15 | 16/40 (one sample) | 38/40 gold unchanged vs 0.25 |
| 0.25 | 17/40 temp=0.0 repro; 18/40 original s0 | do not mix with temp=0.4 pass@k |
| Dense | 15.75/40 mean over 4 samples | file recall almost identical to us |

The 0.25 "six samples" were not six selector seeds. s0 and the accelerated
repro are two temp=0.0 generations on the same (or bit-identical) selection;
s1-s4 reused that selection at temperature 0.4. Report them separately.

Selector seed *does* change the set: 2 of 3 probe instances differed between
seed 42 and 43, by a couple of units, with fitness agreeing to 1e-4. That
variance belongs in a search-stability check, not in the frontier mean.

## What still needs running

1. Gold-coverage grid: **done** (40 x 8, `results/lambda_gold_sweep.jsonl`).
   File recall is invariant in lambda. Span recall peaks at 0.08 (30/60)
   then is flat 26/60 from 0.25 to 0.50. Next downstream spend, if any,
   is on 0.08 vs 0.25 vs 0.50 — the only three points whose gold differs
   or whose cost differs at matched gold.
2. Only after (1) shows a knee, spend downstream samples on two or three lambdas
   (the knee, one cheaper, one more expensive) at temperature 0.0, several
   independent generations, same selector seed. Resolved will jump; the test is
   whether the sign of the jump, averaged, matches the gold-coverage slope.
3. Optional: a gold-free density proxy on the same grid (mean relevance of
   selected units, Cov of the issued need, fraction of tokens in gold files
   *without looking at gold* — e.g. share of tokens in the top-k files by
   need). Needed if we want the mechanism panel in the paper without using the
   answer.
4. Later: move the wall. At frozen lambda, run B in {4K, 8K, 16K} with the
   same gold-coverage panel then a few downstream points. This asks whether
   the sweet point is a fraction of B or an absolute token count. Dense/BM25
   will sit on the new wall; we should still sit left of it.
5. Do not draw the resolved curve from mixed temperatures or mixed selector
   seeds. Do not treat a one-instance drop as a refutation of monotonicity in
   expectation.

## Risks

On 40 instances a true +2 gold-span gain can be erased by one unlucky apply
failure (`django-12125` at 0.15). The mechanism panel is what makes the
frontier interpretable at this scale; the resolved panel is what makes it a
result, and it wants 300 instances or many repeats.

Apply-fail rate is higher for us than for Dense (roughly 5 vs 2). If cheaper
lambda fixes apply by accident of resampling rather than by covering more gold
context lines, split that out.

The 19/40 oracle forced gold into a still-sparse S (lambda was 0.25). It is an
upper bound for that operating point, not for "perfect context."

## Reproducing

Lambda configs are copies of `configs/default.yaml` with one line changed,
`configs/ball_lam<value>.yaml`. Overrides `--seed` and `--cost-lambda` now exist
on `scripts/run_swebench.py` so a grid does not need one file per cell.

Generation: `scripts/run_swebench_lite_ball40v3_repro.sh` with `CONFIG` /
`OFFSET` / `LIMIT` / `TAG`. Evaluation: `scripts/eval_swebench_lite_ball40d2.sh`
with `TAG`, after merging shard predictions.

Gold-coverage sweep: `scripts/analysis/lambda_gold_sweep.py`.
