#!/usr/bin/env python
"""Bit-exactness harness for the decoder/fitness acceleration work.

The acceleration is meant to change only how the objective is evaluated, never
which subset the search returns. This script captures a reference fingerprint
with one revision of the code and compares it against another, so any drift is
caught as a mask difference rather than a timing difference.

    python scripts/analysis/decoder_equivalence.py --dump  ref_before.npz
    # ... apply the optimisation ...
    python scripts/analysis/decoder_equivalence.py --compare ref_before.npz
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from repomaro.config import load_config  # noqa: E402
from repomaro.optimizer import RepoMARO  # noqa: E402

POOL_DIR = Path("cache/pools_v3")
BUDGET = 8192

# Two tiers: many cheap single-decode probes, a few full searches.
DECODE_POOLS = [
    "astropy__astropy-12907",
    "django__django-10914",
    "django__django-10924",
    "astropy__astropy-14182",
    "astropy__astropy-6938",
    "astropy__astropy-7746",
]
OPTIMIZE_POOLS = ["astropy__astropy-12907", "django__django-10914"]
N_DECODE_VECTORS = 8
OPTIMIZE_TMAX = 8


def _load(instance: str):
    path = POOL_DIR / f"{instance}__{BUDGET}.pkl"
    return pickle.loads(path.read_bytes())


def _make_optimizer(instance: str, tmax: int | None = None) -> RepoMARO:
    cfg = load_config("configs/ball_d2.yaml")
    ocfg = dict(cfg["optimizer"])
    if tmax is not None:
        ocfg["max_iterations"] = tmax
    return RepoMARO(_load(instance), BUDGET, cfg["fitness"], ocfg, seed=42)


def collect() -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}

    for instance in DECODE_POOLS:
        opt = _make_optimizer(instance)
        m = len(opt.pool.units)
        # Fixed probe vectors: independent of the optimizer's own RNG stream so
        # they stay identical even if population init were ever reordered.
        rng = np.random.default_rng(1234)
        X = rng.uniform(0.0, 1.0, size=(N_DECODE_VECTORS, m))
        X[:, ~opt.member] = 0.0
        t0 = time.perf_counter()
        masks = np.stack([opt.decoder.decode(x) for x in X], axis=0)
        dt = time.perf_counter() - t0
        fits = np.asarray(opt.fitness.evaluate_batch(masks), dtype=np.float64)
        out[f"decode_mask/{instance}"] = masks
        out[f"decode_fit/{instance}"] = fits
        out[f"decode_time/{instance}"] = np.asarray([dt])
        print(
            f"[decode] {instance:34s} m={m:4d} ball={int(opt.member.sum()):4d} "
            f"sel={masks.sum(axis=1).tolist()} {dt:7.1f}s",
            flush=True,
        )

    for instance in OPTIMIZE_POOLS:
        opt = _make_optimizer(instance, tmax=OPTIMIZE_TMAX)
        t0 = time.perf_counter()
        res = opt.optimize()
        dt = time.perf_counter() - t0
        out[f"opt_mask/{instance}"] = res.mask
        out[f"opt_fit/{instance}"] = np.asarray([res.fitness])
        out[f"opt_iters/{instance}"] = np.asarray([res.iterations])
        out[f"opt_hist/{instance}"] = np.asarray(
            [h["best_fitness"] for h in res.history], dtype=np.float64
        )
        out[f"opt_time/{instance}"] = np.asarray([dt])
        print(
            f"[optimize] {instance:32s} fit={res.fitness:.12f} iters={res.iterations} "
            f"sel={int(res.mask.sum())} {dt:7.1f}s",
            flush=True,
        )
    return out


def compare(ref_path: Path) -> int:
    ref = np.load(ref_path)
    cur = collect()
    print()
    bad = 0
    speedups = []
    for key in sorted(k for k in ref.files if "_time/" not in k):
        a, b = ref[key], cur.get(key)
        if b is None:
            print(f"MISSING  {key}")
            bad += 1
            continue
        if a.dtype == bool:
            n = int((a != b).sum())
            status = "OK " if n == 0 else "DIFF"
            if n:
                bad += 1
            print(f"{status} {key:48s} differing_bits={n}/{a.size}")
        else:
            d = float(np.abs(a.astype(float) - b.astype(float)).max()) if a.size else 0.0
            status = "OK " if d == 0.0 else ("~   " if d < 1e-9 else "DIFF")
            if d >= 1e-9:
                bad += 1
            print(f"{status} {key:48s} max_abs_diff={d:.3e}")

    print()
    for key in sorted(k for k in ref.files if "_time/" in k):
        before, after = float(ref[key][0]), float(cur[key][0])
        speedups.append(before / after if after > 0 else float("nan"))
        print(f"    {key:48s} {before:8.1f}s -> {after:8.1f}s  ({before/max(after,1e-9):5.1f}x)")
    if speedups:
        print(f"\n  median speedup: {np.nanmedian(speedups):.1f}x")

    print()
    if bad == 0:
        print("RESULT: bit-exact (or within 1e-9) on every probe -- pure engineering change confirmed.")
    else:
        print(f"RESULT: {bad} fingerprint(s) diverged -- investigate before rerunning experiments.")
    return 1 if bad else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dump", type=Path, help="write a reference fingerprint")
    g.add_argument("--compare", type=Path, help="compare against a reference fingerprint")
    args = ap.parse_args()

    if args.dump:
        data = collect()
        args.dump.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.dump, **data)
        print(f"\nwrote {args.dump}")
    else:
        raise SystemExit(compare(args.compare))


if __name__ == "__main__":
    main()
