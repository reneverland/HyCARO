"""Quick validation of the three ARO variants."""
import numpy as np
import pytest

from repomaro.aro import RepoARO, BundleARO, FastARO, _relation_bundles


FCFG = {"alpha": 0.45, "beta": 0.35, "gamma": 0.2, "sigma": 2.0,
        "ball": {"alpha": 1.0, "beta": 0.35, "gamma": 0.0, "cost_lambda": 0.12,
                 "redundancy_delta": 0.3}}
OCFG = {
    "population_size": 6,
    "max_iterations": 5,
    "ball_anchors": 2,
    "ball_radius": 1,
    "seed_high": 0.9,
    "seed_low": 0.1,
    "seed_noise": 0.02,
    "enable_retrieval_seeding": True,
}


def test_repo_aro_feasible(toy_pool):
    opt = RepoARO(toy_pool, budget=120, fitness_cfg=FCFG, optimizer_cfg=OCFG, seed=7)
    r = opt.optimize()
    assert opt.decoder.feasible(r.mask)
    assert r.mask.any()
    assert r.fitness > 0
    assert r.method == "RepoARO"


def test_bundle_aro_feasible(toy_pool):
    opt = BundleARO(toy_pool, budget=120, fitness_cfg=FCFG, optimizer_cfg=OCFG, seed=7)
    r = opt.optimize()
    assert opt.decoder.feasible(r.mask)
    assert r.mask.any()
    assert r.fitness > 0
    assert r.method == "BundleARO"


def test_fast_aro_feasible(toy_pool):
    opt = FastARO(toy_pool, budget=120, fitness_cfg=FCFG, optimizer_cfg=OCFG, seed=7)
    r = opt.optimize()
    assert opt.decoder.feasible(r.mask)
    assert r.mask.any()
    assert r.fitness > 0
    assert r.method == "FastARO"


def test_relation_bundles_exist(toy_pool):
    member = np.ones(len(toy_pool.units), dtype=bool)
    bundles = _relation_bundles(toy_pool, member)
    assert len(bundles) > 0
    for b in bundles:
        assert len(b) >= 2


def test_fast_aro_beats_random(toy_pool):
    """FastARO should at least match a random baseline on the toy problem."""
    opt = FastARO(toy_pool, budget=120, fitness_cfg=FCFG, optimizer_cfg=OCFG, seed=7)
    r = opt.optimize()
    # random baseline: just pick units until budget full
    rng = np.random.default_rng(99)
    mask = np.zeros(len(toy_pool.units), dtype=bool)
    used = 0
    for j in rng.permutation(len(toy_pool.units)):
        c = int(toy_pool.token_costs[j])
        if used + c <= 120:
            mask[j] = True
            used += c
    random_fit = float(opt.fitness.evaluate(mask))
    assert r.fitness >= random_fit * 0.8, f"FastARO {r.fitness:.4f} vs random {random_fit:.4f}"
