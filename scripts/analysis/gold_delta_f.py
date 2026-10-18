#!/usr/bin/env python
"""Does the objective reject the gold units, or did the search simply miss them?

For every reference-patch region that the anchor ball made reachable but the
final subset left out, this forces the covering units into the returned subset
and reports the exact change in F and in each of its terms.

The sign of dF is the whole point:

* dF < 0 -- F genuinely prefers the subset without the gold units. The objective
  is mis-specified (the token price outweighs what coverage pays, or coverage
  saturates before the gold unit can earn anything).
* dF > 0 -- F would have scored the gold-bearing subset higher. The objective is
  fine and the search failed to reach it.
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


def span_unit_indices(units, path: str, lo: int, hi: int) -> list[int]:
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
    ap.add_argument("--results", default="results/swebench_lite_8k_ball40d2_s*.jsonl")
    ap.add_argument("--status", default="results/swebench_lite_8k_ball40d2_eval/sanitized/apply_status_ball40d2.jsonl")
    ap.add_argument("--config", default="configs/ball_d2.yaml")
    ap.add_argument("--pool-cache", default="cache/pools_v3")
    ap.add_argument("--budget", type=int, default=8192)
    ap.add_argument("--out", default="results/equiv/gold_delta_f_d2.csv")
    args = ap.parse_args()

    selected: dict[str, list[dict]] = {}
    for path in sorted(glob.glob(args.results)):
        for line in Path(path).read_text().splitlines():
            if line.strip():
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
    lam = float(cfg["fitness"].get("cost_lambda", cfg["fitness"].get("ball", {}).get("cost_lambda", 0.25)))

    rows = []
    for iid, sel_units in sorted(selected.items()):
        pkl = Path(args.pool_cache) / f"{iid}__{args.budget}.pkl"
        pool = pickle.loads(pkl.read_bytes())
        opt = RepoMARO(pool, args.budget, cfg["fitness"], cfg["optimizer"], int(cfg.get("seed", 42)))
        fit, member, costs = opt.fitness, opt.member, pool.token_costs
        uid_to_idx = {u.uid: i for i, u in enumerate(pool.units)}

        mask = np.zeros(len(pool.units), dtype=bool)
        for u in sel_units:
            j = uid_to_idx.get(u["uid"])
            if j is not None:
                mask[j] = True

        missing = set()
        for path, lo, hi in gold_spans(lite.loc[iid, "patch"]):
            idxs = span_unit_indices(pool.units, path, lo, hi)
            if not idxs or any(mask[j] for j in idxs):
                continue
            in_ball = [j for j in idxs if member[j]]
            if in_ball:
                # cheapest covering unit inside the ball
                missing.add(min(in_ball, key=lambda j: int(costs[j])))
        if not missing:
            continue

        add = sorted(missing)
        forced = mask.copy()
        forced[add] = True

        before = fit.evaluate(mask, return_components=True)
        after = fit.evaluate(forced, return_components=True)
        tok_before = int(costs[mask].sum())
        tok_add = int(costs[add].sum())

        rows.append(
            {
                "instance_id": iid,
                "apply_ok": apply_ok.get(iid),
                "n_forced": len(add),
                "tok_before": tok_before,
                "tok_add": tok_add,
                "tok_after": tok_before + tok_add,
                "fits_budget": tok_before + tok_add <= args.budget,
                "F_before": before.total,
                "F_after": after.total,
                "dF": after.total - before.total,
                "dCov": after.relevance - before.relevance,
                "dSupp": after.structural - before.structural,
                "dCost": -lam * tok_add / float(args.budget),
            }
        )
        print(
            f"{iid:32s} apply={'OK ' if apply_ok.get(iid) else 'FAIL'} "
            f"forced={len(add)} +{tok_add:5d}tok  dF={rows[-1]['dF']:+.5f} "
            f"(dCov={rows[-1]['dCov']:+.5f} dSupp={rows[-1]['dSupp']:+.5f} dCost={rows[-1]['dCost']:+.5f})"
            f"{'' if rows[-1]['fits_budget'] else '  [OVER BUDGET]'}",
            flush=True,
        )

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"\n=== verdict over {len(df)} instances with reachable-but-unselected gold ===")
    worse = df[df.dF < 0]
    better = df[df.dF > 0]
    print(f"  dF < 0  (objective rejects the gold units): {len(worse)}")
    print(f"  dF > 0  (search missed a better subset)   : {len(better)}")
    print(f"\n  mean dF    = {df.dF.mean():+.5f}")
    print(f"  mean dCov  = {df.dCov.mean():+.5f}   (alpha-weighted gain from covering the issue)")
    print(f"  mean dSupp = {df.dSupp.mean():+.5f}")
    print(f"  mean dCost = {df.dCost.mean():+.5f}   (lambda = {lam})")
    print(f"\n  instances where the added tokens still fit the budget: {int(df.fits_budget.sum())}/{len(df)}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
