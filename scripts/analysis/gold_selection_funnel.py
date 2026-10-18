#!/usr/bin/env python
"""Where do the reference-patch regions get lost: the pool, the ball, or the choice?

The k/d sweep only measured whether gold regions were *reachable* inside the
anchor ball. Reachable is not chosen: the objective still has to prefer those
units over everything else the ball offers. This walks the full funnel

    gold span -> present in candidate pool -> inside the anchor ball -> selected

per instance, and splits the result by whether the produced patch applied, so
the passing instances act as a control for the failing ones.
"""

from __future__ import annotations

import argparse
import glob
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? ")


def gold_spans(patch: str) -> list[tuple[str, int, int]]:
    """Old-side line ranges touched by the reference patch."""
    out: list[tuple[str, int, int]] = []
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
                out.append((current, start, start + max(length, 1) - 1))
    return out


def load_lite() -> pd.DataFrame:
    files = glob.glob(
        "/home/yuyang/.cache/huggingface/hub/datasets--SWE-bench--SWE-bench_Lite/"
        "**/test-*.parquet",
        recursive=True,
    )
    if not files:
        raise SystemExit("no cached SWE-bench Lite parquet")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.sort_values("instance_id").reset_index(drop=True)


def overlaps(units, path: str, lo: int, hi: int) -> bool:
    for u in units:
        if u["path"] != path:
            continue
        if (u["start_line"] or 0) <= hi and lo <= (u["end_line"] or 0):
            return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results/swebench_lite_8k_ball40d2_s*.jsonl")
    ap.add_argument("--status", default="results/swebench_lite_8k_ball40d2_eval/sanitized/apply_status_ball40d2.jsonl")
    ap.add_argument("--config", default="configs/ball_d2.yaml")
    ap.add_argument("--pool-cache", default="cache/pools_v3")
    ap.add_argument("--budget", type=int, default=8192)
    ap.add_argument("--out", default="results/equiv/gold_funnel_d2.csv")
    args = ap.parse_args()

    selected: dict[str, list[dict]] = {}
    for path in sorted(glob.glob(args.results)):
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            selected[row["instance_id"]] = row["selected_units"]

    apply_ok = {}
    for line in Path(args.status).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            apply_ok[r["instance_id"]] = bool(r.get("apply_ok"))

    lite = load_lite().set_index("instance_id")
    from repomaro.config import load_config
    from repomaro.optimizer import RepoMARO

    cfg = load_config(args.config)
    rows = []
    for iid, units in sorted(selected.items()):
        spans = gold_spans(lite.loc[iid, "patch"])
        pkl = Path(args.pool_cache) / f"{iid}__{args.budget}.pkl"
        pool = pickle.loads(pkl.read_bytes())
        opt = RepoMARO(pool, args.budget, cfg["fitness"], cfg["optimizer"], int(cfg.get("seed", 42)))
        member = np.asarray(opt.member, dtype=bool)
        pool_units = [
            {"path": getattr(u, "path", None), "start_line": getattr(u, "start_line", 0),
             "end_line": getattr(u, "end_line", 0)}
            for u in pool.units
        ]
        ball_units = [pu for pu, inb in zip(pool_units, member) if inb]

        n = len(spans)
        in_pool = sum(1 for p, lo, hi in spans if overlaps(pool_units, p, lo, hi))
        in_ball = sum(1 for p, lo, hi in spans if overlaps(ball_units, p, lo, hi))
        in_sel = sum(1 for p, lo, hi in spans if overlaps(units, p, lo, hi))
        rows.append(
            {
                "instance_id": iid,
                "apply_ok": apply_ok.get(iid),
                "spans": n,
                "in_pool": in_pool,
                "in_ball": in_ball,
                "selected": in_sel,
                "ball_units": int(member.sum()),
                "n_selected": len(units),
            }
        )
        print(
            f"{iid:32s} apply={'OK ' if apply_ok.get(iid) else 'FAIL'} "
            f"spans={n:2d} pool={in_pool:2d} ball={in_ball:2d} selected={in_sel:2d}",
            flush=True,
        )

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print("\n=== funnel, aggregated by whether the patch applied ===")
    agg = df.groupby("apply_ok").agg(
        instances=("instance_id", "count"),
        spans=("spans", "sum"),
        in_pool=("in_pool", "sum"),
        in_ball=("in_ball", "sum"),
        selected=("selected", "sum"),
    )
    agg["pool_rate"] = (agg.in_pool / agg.spans).round(3)
    agg["ball_rate"] = (agg.in_ball / agg.spans).round(3)
    agg["selected_rate"] = (agg.selected / agg.spans).round(3)
    agg["kept_from_ball"] = (agg.selected / agg.in_ball.replace(0, np.nan)).round(3)
    print(agg.to_string())

    print("\n=== instances where the ball had the gold region but the optimizer dropped it ===")
    lost = df[(df.in_ball > df.selected)]
    print(lost.to_string(index=False) if len(lost) else "  none")

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
