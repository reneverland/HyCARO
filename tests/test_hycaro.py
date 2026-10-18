"""HyCARO keeps a feasible incumbent and only scores role-compatible witnesses."""
from pathlib import Path

import numpy as np

from repomaro.graph import Hypergraph
from repomaro.hycaro import HyCARO, MaxWitnessFitness, build_witness_index, project_preferences
from repomaro.schema import CandidatePool, ContextUnit


def _unit(i, cost, rel, dense, text):
    return ContextUnit(
        uid=f"u{i}",
        path=f"f{i}.py",
        text=text,
        kind="function",
        symbol=f"f{i}",
        start_line=1,
        end_line=2,
        token_cost=cost,
        node_id=f"u{i}",
        relevance=rel,
        lexical_score=rel,
        dense_score=dense,
        structural_score=rel,
        embedding=np.eye(3)[i % 3],
    )


def _pool(units, edges, query="unmatchedquerytoken"):
    g = Hypergraph()
    for u in units:
        g.add_node(u.uid)
    for spec in edges:
        g.add_hyperedge(*spec[0], **spec[1])
    return CandidatePool(query, Path("."), units, g, [units[0].uid])


FCFG = {
    "ball": {
        "alpha": 1.0,
        "beta": 0.35,
        "gamma": 0.0,
        "cost_lambda": 0.25,
        "semantic_need_depth": 1,
        "lexical_need_depth": 1,
        "semantic_need_share": 0.5,
    }
}
OCFG = {
    "hycaro_population": 6,
    "hycaro_iterations": 4,
    "hycaro_proposals": 3,
    "hycaro_a_max": 3,
    "hycaro_d_max": 3,
    "ball_anchors": 2,
    "ball_radius": 1,
}


def test_projection_reproduces_membership():
    member = np.array([True, True, True, False])
    mask = np.array([True, False, True, False])
    u = np.array([0.2, 0.8, 0.0, 0.9])
    x = project_preferences(u, mask, member, eps_x=1e-3)
    assert np.array_equal(x >= 0.5, mask)
    assert x[3] == 0.0


def test_same_role_units_are_not_witnesses(tmp_path):
    units = [
        _unit(0, 20, 0.9, 0.9, "def caller(): pass"),
        _unit(1, 20, 0.4, 0.4, "def callee_a(): pass"),
        _unit(2, 20, 0.4, 0.4, "def callee_b(): pass"),
    ]
    pool = _pool(units, [(("calls",), {"sources": ["u0"], "targets": ["u1", "u2"]})])
    pool = CandidatePool(pool.query, tmp_path, pool.units, pool.graph, pool.anchor_node_ids)
    member = np.ones(3, dtype=bool)
    index = build_witness_index(pool, member)
    pairs = {tuple(sorted(p)) for p in zip(index.pair_j.tolist(), index.pair_k.tolist())}
    assert pairs == {(0, 1), (0, 2)}
    fit = MaxWitnessFitness(pool, budget=100, alpha=1, beta=1, gamma=0, cost_lambda=0, semantic_need_depth=1)
    fit.attach_witnesses(index)
    both_callees = np.array([False, True, True])
    with_caller = np.array([True, True, False])
    assert fit.evaluate(both_callees, return_components=True).structural == 0.0
    assert fit.evaluate(with_caller, return_components=True).structural > 0.0


def test_file_contains_is_not_support_and_encloses_is(tmp_path):
    units = [
        _unit(0, 10, 0.8, 0.8, "class Box: pass"),
        _unit(1, 10, 0.8, 0.8, "def method(self): pass"),
    ]
    pool = _pool(units, [
        (("contains",), {"members": ["u0", "u1"]}),
        (("encloses",), {"sources": ["u0"], "targets": ["u1"]}),
    ])
    pool = CandidatePool(pool.query, tmp_path, pool.units, pool.graph, pool.anchor_node_ids)
    index = build_witness_index(pool, np.ones(2, dtype=bool))
    assert index.pair_j.size == 1
    assert len(index.groups) == 1


def test_hycaro_stays_feasible_and_matches_its_score(toy_pool):
    opt = HyCARO(toy_pool, budget=120, fitness_cfg=FCFG, optimizer_cfg=OCFG, seed=11)
    result = opt.optimize()
    assert int(toy_pool.token_costs[result.mask].sum()) <= 120
    assert not (result.mask & ~opt.member).any()
    fitness = MaxWitnessFitness(
        toy_pool, budget=120,
        alpha=1, beta=0.35, gamma=0, cost_lambda=0.25, semantic_need_depth=1, lexical_need_depth=1,
    )
    fitness.attach_witnesses(build_witness_index(toy_pool, opt.member))
    assert abs(float(fitness.evaluate(result.mask)) - result.fitness) < 1e-8
    assert result.fitness >= 0.0
    again = HyCARO(toy_pool, budget=120, fitness_cfg=FCFG, optimizer_cfg=OCFG, seed=11).optimize()
    assert np.array_equal(result.mask, again.mask)


def test_joint_replacement_picks_up_the_helper(tmp_path):
    units = [
        _unit(0, 50, 0.95, 0.95, "def alpha(): pass"),
        _unit(1, 50, 0.01, 0.01, "def helper(): pass"),
        _unit(2, 50, 0.80, 0.80, "def distractor(): pass"),
    ]
    pool = _pool(units, [(("calls",), {"sources": ["u0"], "targets": ["u1"]})])
    pool = CandidatePool(pool.query, tmp_path, pool.units, pool.graph, ["u0"])
    opt = HyCARO(pool, budget=100, fitness_cfg=FCFG, optimizer_cfg=OCFG, seed=11)
    result = opt.optimize()
    assert result.mask[1]
    assert result.fitness > 1.0
    assert int(pool.token_costs[result.mask].sum()) <= 100


def test_relation_greedy_uses_the_same_actions_and_stays_feasible(tmp_path):
    units = [
        _unit(0, 50, 0.95, 0.95, "def alpha(): pass"),
        _unit(1, 50, 0.01, 0.01, "def helper(): pass"),
        _unit(2, 50, 0.80, 0.80, "def distractor(): pass"),
    ]
    pool = _pool(units, [(("calls",), {"sources": ["u0"], "targets": ["u1"]})])
    pool = CandidatePool(pool.query, tmp_path, pool.units, pool.graph, ["u0"])
    cfg = {**OCFG, "hycaro_search": "greedy", "hycaro_time_budget_s": 5.0}
    result = HyCARO(pool, budget=100, fitness_cfg=FCFG, optimizer_cfg=cfg, seed=11).optimize()
    assert result.method == "Relation-guided greedy"
    assert result.mask[1]
    assert int(pool.token_costs[result.mask].sum()) <= 100
    assert not (result.mask & ~np.ones(3, dtype=bool)).any()


def test_deadline_before_init_returns_empty(toy_pool):
    cfg = {**OCFG, "hycaro_time_budget_s": 0.0}
    result = HyCARO(toy_pool, budget=120, fitness_cfg=FCFG, optimizer_cfg=cfg, seed=1).optimize()
    assert result.stopped == "time"
    assert result.iterations == 0
    assert not result.mask.any()
    assert result.fitness == 0.0
