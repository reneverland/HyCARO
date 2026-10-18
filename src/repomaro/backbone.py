from __future__ import annotations

import numpy as np

from .schema import CandidatePool

# Confidence of each hyperedge relation, used by the structural support term.
RELATION_WEIGHT = {
    "tests": 0.95,
    "inherits": 0.88,
    "calls": 0.85,
    "encloses": 0.80,
    "imports": 0.75,
    "decorates": 0.70,
    "contains": 0.65,
    "references": 0.60,
    "state": 0.55,
}
DEFAULT_EDGE_WEIGHT = 0.50


def build_anchor_balls(pool: CandidatePool, k: int, d: int) -> np.ndarray:
    """Feasible set of the k-anchor radius-d constraint.

    Anchors are the k highest-relevance units; a unit is admissible when it lies
    within d hops of some anchor on the repository graph. Admission costs
    nothing, so a unit's own token cost is the only price the decoder pays for it.
    """
    units = pool.units
    m = len(units)
    member = np.zeros(m, dtype=bool)
    if m == 0:
        return member
    relevance = np.asarray(
        pool.relevance if pool.relevance is not None else [u.relevance for u in units],
        dtype=np.float64,
    )
    anchors = np.argsort(-relevance, kind="mergesort")[: max(int(k), 1)].tolist()
    member[anchors] = True
    graph = pool.graph
    if graph is None or int(d) <= 0:
        return member
    uid_to_idx = {u.uid: i for i, u in enumerate(units)}
    for a in anchors:
        if units[a].uid not in graph:
            continue
        for uid in graph.hop_lengths(units[a].uid, cutoff=int(d)):
            j = uid_to_idx.get(uid)
            if j is not None:
                member[j] = True
    return member
