from __future__ import annotations

import numpy as np

from repomaro.optimizer import RepoMARO

FITNESS_CFG = {
    "alpha": 0.45,
    "beta": 0.35,
    "gamma": 0.2,
    "sigma": 2.0,
    "d0": 2,
    "redundancy_delta": 0.3,
    "ball": {"objective": "coverage_support", "alpha": 1.0, "beta": 0.35, "cost_lambda": 0.25},
}
OPTIMIZER_CFG = {
    "population_size": 8,
    "max_iterations": 6,
    "threshold": 0.5,
    "lambda_min": 0.15,
    "lambda_max": 0.8,
    "crossover_rate": 0.7,
    "elite_fraction": 0.5,
    "local_proposals": 9,
    "local_search_patience": 2,
    "stopping_patience": 5,
    "epsilon": 1e-6,
    "seed_high": 0.9,
    "seed_low": 0.1,
    "seed_noise": 0.02,
    "ball_anchors": 5,
    "ball_radius": 2,
}


def _opt(pool, budget=260):
    return RepoMARO(pool, budget, FITNESS_CFG, OPTIMIZER_CFG, seed=3)


def test_prune_proposals_follow_exact_delta_fitness(toy_pool):
    opt = _opt(toy_pool)
    mask = opt.member.copy()
    proposals = opt._generate_branch_proposals(mask, 9)
    pruned = [int(p[1][0]) for p in proposals if p[0] == "prune"]
    assert pruned

    base = float(opt.fitness.evaluate(mask))
    costs = toy_pool.token_costs
    loss = {}
    for j in np.flatnonzero(mask).tolist():
        trial = mask.copy()
        trial[j] = False
        loss[j] = (base - float(opt.fitness.evaluate(trial))) / costs[j]
    # The cheapest unit to give up is proposed first, whatever its relevance rank.
    assert pruned[0] == min(loss, key=lambda j: (loss[j], j))


def test_add_proposals_follow_exact_delta_fitness(toy_pool):
    opt = _opt(toy_pool)
    mask = np.zeros(len(toy_pool.units), dtype=bool)
    mask[np.flatnonzero(opt.member)[:2]] = True
    proposals = opt._generate_branch_proposals(mask, 9)
    added = [int(p[1][0]) for p in proposals if p[0] == "reconnect"]
    assert added

    base = float(opt.fitness.evaluate(mask))
    costs = toy_pool.token_costs
    used = int(costs[mask].sum())
    gain = {}
    for j in np.flatnonzero(opt.member & ~mask).tolist():
        if used + int(costs[j]) > opt.budget:
            continue
        trial = mask.copy()
        trial[j] = True
        gain[j] = (float(opt.fitness.evaluate(trial)) - base) / costs[j]
    assert added[0] == max(gain, key=lambda j: (gain[j], -j))


def test_optimize_still_returns_a_feasible_context(toy_pool):
    result = _opt(toy_pool).optimize()
    assert result.mask.any()
    assert int(toy_pool.token_costs[result.mask].sum()) <= 260
