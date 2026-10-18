"""Main-table controls stay inside the ball and the token budget."""
import numpy as np

from repomaro.controlled import run_controlled
from repomaro.runner import run_selector


FCFG = {
    "alpha": 0.45,
    "beta": 0.35,
    "gamma": 0.2,
    "sigma": 2.0,
    "ball": {
        "alpha": 1.0,
        "beta": 0.35,
        "gamma": 0.0,
        "cost_lambda": 0.25,
        "semantic_need_depth": 1,
        "lexical_need_depth": 1,
    },
}
OCFG = {
    "restrict_to_ball": True,
    "ball_anchors": 2,
    "ball_radius": 1,
    "selection_deadline_s": 2.0,
    "de_population": 4,
    "de_iterations": 2,
    "hycaro_d_max": 2,
    "hycaro_population": 4,
    "hycaro_crossover": 0.8,
    "hycaro_lambda_min": 0.2,
    "hycaro_lambda_max": 0.8,
    "hycaro_eps_f": 1e-9,
    "hycaro_eps_x": 1e-3,
}


def _cfg():
    return {"fitness": FCFG, "optimizer": OCFG, "seed": 11}


def _assert_feasible(pool, result, budget=120):
    mask = np.asarray(result["mask"], dtype=bool)
    assert int(pool.token_costs[mask].sum()) <= budget
    member = np.ones(len(pool.units), dtype=bool)
    from repomaro.controlled import admissible_member

    member = admissible_member(pool, OCFG)
    assert not (mask & ~member).any()
    assert result["fitness"] >= -1e-8


def test_retrieval_controls_stay_in_ball(toy_pool):
    cfg = _cfg()
    for method in ("bm25", "dense", "rrf", "mmr", "repograph"):
        result = run_selector(method, toy_pool, 120, cfg)
        _assert_feasible(toy_pool, result)


def test_search_controls_stay_feasible(toy_pool):
    cfg = _cfg()
    for method in ("greedy_1swap", "two_unit", "de_repair"):
        result = run_controlled(method, toy_pool, 120, cfg["fitness"], cfg["optimizer"], 11)
        _assert_feasible(toy_pool, result)


def test_deadline_zero_keeps_a_feasible_incumbent(toy_pool):
    ocfg = {**OCFG, "selection_deadline_s": 0.0}
    for method in ("greedy_1swap", "two_unit", "de_repair"):
        result = run_controlled(method, toy_pool, 120, FCFG, ocfg, 11)
        _assert_feasible(toy_pool, result)
        assert result["metadata"]["stopped"] in {"time", "few-rabbits", "optimum"}
