#!/usr/bin/env python
"""A/B the batched repair loops against a literal transcription of the scalar ones.

Both variants run inside one process against the same fitness object, so the only
difference under test is scalar-vs-batched evaluation of the one-unit moves.
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from repomaro.config import load_config  # noqa: E402
from repomaro.optimizer import RepoMARO  # noqa: E402

BUDGET = 8192
INSTANCES = [
    "astropy__astropy-12907",
    "django__django-10914",
    "django__django-10924",
    "astropy__astropy-6938",
]
N_VECTORS = 4


# ---- verbatim copies of the pre-optimisation scalar loops --------------------

def ref_trim(dec, selected):
    selected = selected.copy()
    while int(selected.sum()) > 1:
        current = float(dec.fitness.evaluate(selected))
        best = None
        for j in np.flatnonzero(selected).tolist():
            trial = selected.copy()
            trial[j] = False
            gain = float(dec.fitness.evaluate(trial)) - current
            if gain <= 0:
                continue
            if best is None or gain > best[0]:
                best = (gain, j)
        if best is None:
            break
        selected[best[1]] = False
    return selected


def ref_shed(dec, selected):
    selected = selected.copy()
    used = int(dec.costs[selected].sum())
    while used > dec.budget:
        heads = np.flatnonzero(selected)
        if len(heads) == 0:
            break
        current = float(dec.fitness.evaluate(selected))
        best_j = None
        best_key = None
        for j in heads.tolist():
            trial = selected.copy()
            trial[j] = False
            delta = current - float(dec.fitness.evaluate(trial))
            key = (delta / int(dec.costs[j]), j)
            if best_key is None or key < best_key:
                best_key = key
                best_j = j
        if best_j is None:
            break
        selected[best_j] = False
        used -= int(dec.costs[best_j])
    return selected


def ref_absorb(dec, selected):
    selected = selected.copy()
    used = int(dec.costs[selected].sum())
    while used < dec.budget:
        current = float(dec.fitness.evaluate(selected))
        best = None
        for j in np.flatnonzero(dec.member & ~selected).tolist():
            c = int(dec.costs[j])
            if used + c > dec.budget:
                continue
            trial = selected.copy()
            trial[j] = True
            delta = float(dec.fitness.evaluate(trial)) - current
            if delta <= 0:
                continue
            key = (delta / c, -j)
            if best is None or key > best[0]:
                best = (key, j, c)
        if best is None:
            break
        _, j, c = best
        selected[j] = True
        used += c
    return selected


def ref_decode(dec, x):
    selected = (np.asarray(x, dtype=np.float64) >= dec.threshold) & dec.member
    if not dec.repair:
        return selected
    if int(dec.costs[selected].sum()) > dec.budget:
        selected = ref_shed(dec, selected)
    selected = ref_trim(dec, selected)
    return ref_absorb(dec, selected)


def main() -> None:
    cfg = load_config("configs/ball_d2.yaml")
    print(f"{'instance':26s} {'vec':>3} {'ref s':>8} {'new s':>8} {'speedup':>8} "
          f"{'sel_ref':>7} {'sel_new':>7} {'mask':>6} {'|dF|':>10}")
    print("-" * 100)
    all_same = True
    speedups = []
    for inst in INSTANCES:
        pool = pickle.loads(Path(f"cache/pools_v3/{inst}__{BUDGET}.pkl").read_bytes())
        opt = RepoMARO(pool, BUDGET, cfg["fitness"], cfg["optimizer"], seed=42)
        dec, m = opt.decoder, len(pool.units)
        rng = np.random.default_rng(1234)
        X = rng.uniform(0.0, 1.0, size=(N_VECTORS, m))
        X[:, ~opt.member] = 0.0
        for i, x in enumerate(X):
            t0 = time.perf_counter()
            a = ref_decode(dec, x)
            ta = time.perf_counter() - t0
            t0 = time.perf_counter()
            b = dec.decode(x)
            tb = time.perf_counter() - t0
            same = bool(np.array_equal(a, b))
            all_same &= same
            speedups.append(ta / max(tb, 1e-9))
            df = abs(float(opt.fitness.evaluate(a)) - float(opt.fitness.evaluate(b)))
            print(f"{inst:26s} {i:>3} {ta:>8.2f} {tb:>8.2f} {ta/max(tb,1e-9):>7.1f}x "
                  f"{int(a.sum()):>7} {int(b.sum()):>7} {'SAME' if same else '**DIFF**':>6} {df:>10.2e}",
                  flush=True)
    print("-" * 100)
    print(f"median speedup: {np.median(speedups):.1f}x   min: {np.min(speedups):.1f}x   max: {np.max(speedups):.1f}x")
    print("masks identical on every probe" if all_same else "*** MASK DIVERGENCE -- inspect above ***")


if __name__ == "__main__":
    main()
