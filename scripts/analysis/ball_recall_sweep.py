#!/usr/bin/env python
"""Measure how much gold context the anchor ball makes reachable, over (k, d).

Only the feasible region is constructed -- no population search -- so the whole
grid runs on CPU in minutes. Reports, per (k, d): the fraction of reference-patch
hunk regions that lie inside the ball, the ball size, and the token mass the ball
spans (the budget pressure the optimizer would then face).
"""
from __future__ import annotations

import argparse
import glob
import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd

from repomaro.config import load_config
from repomaro.optimizer import RepoMARO

_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? ")


def gold_spans(patch: str) -> dict[str, list[tuple[int, int]]]:
    """Old-side line ranges touched by the reference patch, keyed by file."""
    out: dict[str, list[tuple[int, int]]] = {}
    current = None
    for line in patch.splitlines():
        if line.startswith("--- a/"):
            current = line[6:].strip()
        elif line.startswith("+++ b/"):
            path = line[6:].strip()
            current = None if path == "/dev/null" else path
        elif line.startswith("@@") and current:
            m = _HUNK.match(line)
            if m:
                start = int(m.group(1))
                length = int(m.group(2) or 1)
                out.setdefault(current, []).append((start, start + max(length, 1) - 1))
    return out


def load_lite(limit: int, offset: int):
    files = glob.glob(
        "/home/yuyang/.cache/huggingface/hub/datasets--SWE-bench--SWE-bench_Lite/"
        "**/test-*.parquet",
        recursive=True,
    )
    if not files:
        raise SystemExit("no cached SWE-bench Lite parquet")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.sort_values("instance_id").reset_index(drop=True)
    return df.iloc[offset : offset + limit]


def span_indices(units, path: str, lo: int, hi: int) -> list[int]:
    out = []
    for j, u in enumerate(units):
        if getattr(u, "path", None) != path:
            continue
        a = getattr(u, "start_line", 0) or 0
        b = getattr(u, "end_line", 0) or 0
        if a <= hi and lo <= b:
            out.append(j)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--pool-cache", default="cache/pools_v3")
    ap.add_argument("--budget", type=int, default=8192)
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--anchors", default="3,5,8,10,15,20,30")
    ap.add_argument("--radii", default="1,2,3")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    ks = [int(x) for x in args.anchors.split(",")]
    ds = [int(x) for x in args.radii.split(",")]

    rows = []
    instances = load_lite(args.limit, args.offset)
    for n, r in enumerate(instances.itertuples(), 1):
        iid = r.instance_id
        pkl = Path(args.pool_cache) / f"{iid}__{args.budget}.pkl"
        if not pkl.exists():
            print(f"[{n}] {iid}: no cached pool, skipped", flush=True)
            continue
        pool = pickle.loads(pkl.read_bytes())
        units = list(pool.units)
        costs = np.asarray([getattr(u, "token_cost", 0) or 0 for u in units], dtype=float)
        spans = gold_spans(r.patch)
        targets = [
            (path, lo, hi, span_indices(units, path, lo, hi))
            for path, ranges in spans.items()
            for (lo, hi) in ranges
        ]
        n_spans = len(targets)
        n_in_pool = sum(1 for *_, idx in targets if idx)

        for k in ks:
            for d in ds:
                ocfg = dict(cfg["optimizer"])
                ocfg["ball_anchors"] = k
                ocfg["ball_radius"] = d
                opt = RepoMARO(pool, args.budget, cfg["fitness"], ocfg, seed)
                member = np.asarray(opt.member).astype(bool)
                in_ball = sum(1 for *_, idx in targets if any(member[j] for j in idx))
                rows.append(
                    {
                        "instance_id": iid,
                        "k": k,
                        "d": d,
                        "spans": n_spans,
                        "in_pool": n_in_pool,
                        "in_ball": in_ball,
                        "ball_units": int(member.sum()),
                        "pool_units": len(units),
                        "ball_tokens": float(costs[member].sum()),
                    }
                )
        print(f"[{n}/{len(instances)}] {iid}: {n_spans} spans, {n_in_pool} in pool", flush=True)

    out = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    agg = out.groupby(["k", "d"]).agg(
        spans=("spans", "sum"),
        in_pool=("in_pool", "sum"),
        in_ball=("in_ball", "sum"),
        ball_units=("ball_units", "mean"),
        ball_tokens=("ball_tokens", "mean"),
    )
    agg["ball_recall"] = (agg.in_ball / agg.spans).round(3)
    agg["recall_of_available"] = (agg.in_ball / agg.in_pool).round(3)
    print()
    print("=== gold-span reachability inside the anchor ball ===")
    print(agg.round(1).to_string())
    print()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
