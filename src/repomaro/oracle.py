"""Oracle context construction -- diagnostic upper bounds only.

Everything here consults the reference patch, so no number produced with it can
be reported as a method result. Its only job is to answer whether the context
the selector returns is what limits the downstream model: if handing the model
the reference regions does not move the resolved rate, then improving the
selection objective cannot either.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from .schema import CandidatePool


def gold_unit_indices(
    pool: CandidatePool, gold_spans: dict[str, list[tuple[int, int]]]
) -> tuple[list[int], int, int]:
    """Units overlapping a reference hunk, plus span counts for reporting.

    Returns (unit indices, total spans, spans with at least one covering unit).
    A span with no covering unit is unreachable for any selector at this pool
    granularity, which is why it is counted separately rather than hidden.
    """
    chosen: set[int] = set()
    total = 0
    covered = 0
    for path, ranges in (gold_spans or {}).items():
        for lo, hi in ranges:
            total += 1
            hits = [
                j
                for j, u in enumerate(pool.units)
                if getattr(u, "path", None) == path
                and (getattr(u, "start_line", 0) or 0) <= hi
                and lo <= (getattr(u, "end_line", 0) or 0)
            ]
            if hits:
                covered += 1
                # Cheapest covering unit: the oracle should buy the gold region,
                # not the largest container that happens to include it.
                chosen.add(min(hits, key=lambda j: int(pool.token_costs[j])))
    return sorted(chosen), total, covered


def force_gold_mask(
    pool: CandidatePool,
    mask: np.ndarray,
    gold_spans: dict[str, list[tuple[int, int]]],
    budget: int,
    protect: Iterable[int] = (),
) -> tuple[np.ndarray, dict[str, Any]]:
    """Add the gold units to `mask` and shed others until the budget holds again.

    The budget is kept identical to the reported runs, so the comparison isolates
    *which* units the context spends its tokens on rather than how many it gets.
    """
    mask = np.asarray(mask, dtype=bool).copy()
    costs = pool.token_costs
    relevance = pool.relevance
    gold, n_spans, n_covered = gold_unit_indices(pool, gold_spans)
    protected = set(int(j) for j in protect) | set(gold)

    added = [j for j in gold if not mask[j]]
    mask[gold] = True

    gold_tokens = int(costs[gold].sum()) if gold else 0
    truncated = []
    if gold_tokens > budget:
        # Pathological but must be handled explicitly: keep the cheapest gold
        # units that fit and record what fell off.
        order = sorted(gold, key=lambda j: int(costs[j]))
        keep, used = [], 0
        for j in order:
            c = int(costs[j])
            if used + c > budget:
                truncated.append(j)
                continue
            keep.append(j)
            used += c
        mask[:] = False
        mask[keep] = True
        protected = set(keep)

    dropped = []
    used = int(costs[mask].sum())
    if used > budget:
        # Shed the least relevant token first; gold units are never candidates.
        candidates = [j for j in np.flatnonzero(mask).tolist() if j not in protected]
        candidates.sort(key=lambda j: (relevance[j] / max(int(costs[j]), 1), -j))
        for j in candidates:
            if used <= budget:
                break
            mask[j] = False
            dropped.append(j)
            used -= int(costs[j])

    info = {
        "gold_spans": n_spans,
        "gold_spans_coverable": n_covered,
        "gold_units": len(gold),
        "gold_units_added": len(added),
        "gold_tokens": gold_tokens,
        "gold_units_truncated": len(truncated),
        "nongold_units_dropped": len(dropped),
        "tokens_after": int(costs[mask].sum()),
        "budget": int(budget),
    }
    return mask, info
