from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from repomaro.graph import Hypergraph
from repomaro.fitness import IssueCoverageFitness
from repomaro.schema import CandidatePool, ContextUnit


def _mask(pool, *idx):
    m = np.zeros(len(pool.units), dtype=bool)
    m[list(idx)] = True
    return m


@pytest.fixture
def lexical_pool(tmp_path: Path):
    """An issue naming three identifiers, each defined in a different unit."""
    texts = [
        "def apply_coupon(order): return order",       # covers 'coupon'
        "def apply_coupon_v2(order): return order",    # duplicate need coverage
        "def compute_refund(order): return order",     # covers 'refund'
        "def render_invoice(order): return order",
    ]
    embs = np.array([[1.0, 0.0], [1.0, 0.02], [0.0, 1.0], [0.3, 0.3]])
    units = [
        ContextUnit(
            uid=f"u{i}", path=f"f{i}.py", text=t, kind="function", symbol=f"s{i}",
            token_cost=40, node_id=f"u{i}", relevance=0.9 - 0.2 * i,
            lexical_score=0.9 - 0.2 * i, dense_score=0.9 - 0.2 * i, embedding=embs[i],
        )
        for i, t in enumerate(texts)
    ]
    g = Hypergraph()
    for u in units:
        g.add_node(u.uid)
    g.add_hyperedge("calls", ["u0", "u2"])
    return CandidatePool("apply_coupon returns a wrong compute_refund amount", tmp_path, units, g, ["u0"])


def test_terms_are_bounded_and_decomposed(toy_pool):
    f = IssueCoverageFitness(toy_pool, budget=200, alpha=1.0, beta=0.5, gamma=0.0, cost_lambda=0.25)
    comp = f.evaluate(_mask(toy_pool, 0, 2), return_components=True)
    assert 0.0 <= comp.relevance <= 1.0
    assert 0.0 <= comp.structural <= 1.0
    cost = f.cost(_mask(toy_pool, 0, 2)) / f.budget
    assert comp.total == np.float64(comp.relevance + 0.5 * comp.structural - 0.25 * cost)


def test_coverage_follows_the_query_not_the_ranking(lexical_pool):
    """u1 outranks u2 but re-covers a need u0 already answered, so it gains less."""
    f = IssueCoverageFitness(lexical_pool, budget=400, alpha=1.0, beta=0.0, cost_lambda=0.0)
    first = f.evaluate(_mask(lexical_pool, 0))
    duplicate = f.evaluate(_mask(lexical_pool, 0, 1)) - first
    distinct = f.evaluate(_mask(lexical_pool, 0, 2)) - first
    assert lexical_pool.units[1].relevance > lexical_pool.units[2].relevance
    assert duplicate < distinct


def test_depth_regulariser_saves_coverage_when_the_issue_names_nothing(toy_pool):
    """toy_pool's query shares no token with any unit, so only the semantic need exists."""
    plain = IssueCoverageFitness(toy_pool, budget=400, alpha=1.0, beta=0.0, cost_lambda=0.0, semantic_need_depth=1)
    assert plain.need_affinity.shape[0] == 1
    # A single unit answers the only need in full: alpha is inert beyond the first pick.
    assert plain.evaluate(_mask(toy_pool, 0, 2)) == plain.evaluate(_mask(toy_pool, 0))

    reg = IssueCoverageFitness(toy_pool, budget=400, alpha=1.0, beta=0.0, cost_lambda=0.0, semantic_need_depth=4)
    assert reg.evaluate(_mask(toy_pool, 0, 2)) > reg.evaluate(_mask(toy_pool, 0))


def test_depth_one_reduces_to_the_max_operator(lexical_pool):
    deep = IssueCoverageFitness(lexical_pool, budget=400, alpha=1.0, beta=0.0, cost_lambda=0.0)
    flat = IssueCoverageFitness(
        lexical_pool, budget=400, alpha=1.0, beta=0.0, cost_lambda=0.0, semantic_need_depth=1
    )
    full = _mask(lexical_pool, 0, 1, 2, 3)
    sub = flat.need_affinity[:, full]
    assert flat.evaluate(full) == float(
        (flat.need_w * sub.max(axis=1)).sum() / flat.z_need
    )
    # The regulariser only ever withholds credit, never invents it.
    assert deep.evaluate(full) <= flat.evaluate(full)


def test_support_rewards_completing_a_relevant_unit(toy_pool):
    """u2 is a neighbour of the high-prize u0; u5 hangs off the low-prize tail."""
    f = IssueCoverageFitness(toy_pool, budget=400, alpha=0.0, beta=1.0, cost_lambda=0.0)
    assert f.evaluate(_mask(toy_pool, 0)) == 0.0
    assert f.evaluate(_mask(toy_pool, 0, 2)) > f.evaluate(_mask(toy_pool, 0, 3))


def test_cost_makes_the_objective_stop_early(toy_pool):
    everything = _mask(toy_pool, 0, 1, 2, 3, 4, 5)
    priced = IssueCoverageFitness(toy_pool, budget=260, alpha=1.0, beta=0.35, cost_lambda=1.0)
    free = IssueCoverageFitness(toy_pool, budget=260, alpha=1.0, beta=0.35, cost_lambda=0.0)
    assert free.evaluate(everything) > free.evaluate(_mask(toy_pool, 0, 2))
    assert priced.evaluate(everything) < priced.evaluate(_mask(toy_pool, 0, 2))


def test_local_state_matches_full_evaluation(toy_pool):
    f = IssueCoverageFitness(toy_pool, budget=260, alpha=1.0, beta=0.35, cost_lambda=0.25)
    base = _mask(toy_pool, 0, 2)
    state = f.build_local_state(base)
    assert state.base_score == f.evaluate(base)
    assert state.score_add(np.array([4])) == f.evaluate(_mask(toy_pool, 0, 2, 4))
    assert state.score_remove(np.array([2])) == f.evaluate(_mask(toy_pool, 0))
