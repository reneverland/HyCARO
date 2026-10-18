from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from repomaro.graph import Hypergraph
from repomaro.schema import CandidatePool, ContextUnit


@pytest.fixture
def toy_pool(tmp_path: Path):
    units = []
    costs = [40, 50, 35, 60, 30, 45]
    rel = [0.9, 0.75, 0.65, 0.2, 0.55, 0.1]
    embs = np.array([
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.9, 0.1],
        [0.0, 0.0, 1.0],
        [0.1, 0.0, 0.9],
    ])
    for i in range(6):
        units.append(ContextUnit(
            uid=f"u{i}", path=f"f{i}.py", text=f"def f{i}(): pass", kind="function", symbol=f"f{i}",
            start_line=1, end_line=1, token_cost=costs[i], node_id=f"u{i}", relevance=rel[i],
            lexical_score=rel[i], dense_score=rel[i] * 0.9, structural_score=rel[i] * 0.8, embedding=embs[i],
        ))
    g = Hypergraph()
    for u in units: g.add_node(u.uid)
    for a, b in [("u0", "u2"), ("u0", "u4"), ("u1", "u2"), ("u2", "u3"), ("u4", "u5")]:
        g.add_hyperedge("edge", [a, b])
    return CandidatePool("fix refund coupon", tmp_path, units, g, ["u0", "u2"])
