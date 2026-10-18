#!/usr/bin/env python
"""Sweep cost_lambda and measure what the tokens actually buy.

Resolved rate needs a downstream call and a Docker evaluation per grid point,
which is hours. Gold coverage needs neither: the pool is cached, so a whole grid
costs one selector run per (instance, lambda). It cannot tell us the resolved
rate, but it can tell us whether extra tokens are reaching the lines the
reference patch touches, and that is what the sweet-spot question turns on.

Comparing 0.15 against 0.25 showed the context growing 28% while gold span
coverage moved 26/60 to 28/60 and gold file hits stayed at 30/40. If that holds
across the grid, the knee is wherever coverage saturates and everything to the
left is paying for distractors.

Reads reference patches, so this is diagnostic only: nothing here may feed a
selector or a prompt on a reported run.
"""
from __future__ import annotations

import argparse
import copy
import json
import pickle
import statistics as st
import time
from pathlib import Path

from repomaro.adapters.swebench import load_swebench_lite
from repomaro.candidate_pool import CandidatePoolBuilder
from repomaro.config import load_config
from repomaro.io_utils import append_jsonl
from repomaro.runner import run_selector


def _build_pool(builder, inst, budget: int, cfg: dict, cache_dir: Path | None):
    if cache_dir is None:
        return builder.build(inst.query, inst.repo_root, budget, cfg["fitness"])
    path = cache_dir / f"{inst.instance_id}__{budget}.pkl"
    if path.exists():
        try:
            return pickle.loads(path.read_bytes())
        except Exception as exc:
            print(f"  pool cache unreadable for {inst.instance_id}: {exc}", flush=True)
    pool = builder.build(inst.query, inst.repo_root, budget, cfg["fitness"])
    tmp = path.with_suffix(".pkl.tmp")
    tmp.write_bytes(pickle.dumps(pool))
    tmp.replace(path)
    return pool


def _units(pool, mask) -> list[dict]:
    out = []
    for unit, keep in zip(pool.units, mask):
        if not keep:
            continue
        out.append(
            {
                "path": unit.path,
                "start_line": unit.start_line,
                "end_line": unit.end_line,
                "token_cost": unit.token_cost,
            }
        )
    return out


def _coverage(units: list[dict], spans: dict[str, list[tuple[int, int]]]) -> tuple[int, int, int, int]:
    """(spans hit, spans total, gold files hit, gold files total)."""
    hit = total = 0
    for path, ranges in spans.items():
        for gs, ge in ranges:
            total += 1
            if any(
                u["path"] == path and not (u["end_line"] < gs or u["start_line"] > ge)
                for u in units
            ):
                hit += 1
    sel_paths = {u["path"] for u in units}
    return hit, total, len(set(spans) & sel_paths), len(spans)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--checkout-cache", default="cache/checkouts")
    ap.add_argument("--pool-cache", default="cache/pools_v3")
    ap.add_argument("--budget", type=int, default=8192)
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument(
        "--lambdas",
        default="0.0,0.02,0.05,0.08,0.12,0.15,0.20,0.25,0.30,0.40,0.50,0.70",
    )
    ap.add_argument("--seeds", default="42", help="comma separated; the DE search is stochastic")
    ap.add_argument("--out", default="results/lambda_gold_sweep.jsonl")
    args = ap.parse_args()

    lambdas = [float(x) for x in args.lambdas.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    base = copy.deepcopy(load_config(args.config))
    base["candidate_pool"]["granularity"] = "symbol"

    print("Loading SWE-bench Lite...", flush=True)
    instances = load_swebench_lite(args.checkout_cache, args.limit, offset=args.offset)
    gold = {inst.instance_id: (inst.gold_spans or {}) for inst in instances}
    missing = [i for i, s in gold.items() if not s]
    if missing:
        print(f"!! no gold spans for {len(missing)} instances: {missing[:5]}", flush=True)

    builder = CandidatePoolBuilder(base["candidate_pool"], (base.get("downstream") or {}).get("model"))
    pool_cache = Path(args.pool_cache) if args.pool_cache else None
    if pool_cache:
        pool_cache.mkdir(parents=True, exist_ok=True)
    out = Path(args.out)
    done: set[tuple[str, float, int]] = set()
    if out.exists():
        for line in out.open(encoding="utf-8"):
            if not line.strip():
                continue
            rec = json.loads(line)
            done.add((rec["instance_id"], float(rec["cost_lambda"]), int(rec["seed"])))
        print(f"resuming, {len(done)} rows already in {out}", flush=True)

    print(f"{len(instances)} instances x {len(lambdas)} lambdas x {len(seeds)} seeds", flush=True)
    rows: list[dict] = []
    for n, inst in enumerate(instances, 1):
        spans = gold.get(inst.instance_id) or {}
        pending = [(lam, seed) for lam in lambdas for seed in seeds if (inst.instance_id, lam, seed) not in done]
        if not pending:
            print(f"[{n}/{len(instances)}] {inst.instance_id} skip (done)", flush=True)
            continue
        pool = _build_pool(builder, inst, args.budget, base, pool_cache)
        for lam, seed in pending:
            cfg = copy.deepcopy(base)
            cfg["seed"] = seed
            cfg.setdefault("fitness", {}).setdefault("ball", {})["cost_lambda"] = lam
            t0 = time.perf_counter()
            result = run_selector("repomaro", pool, args.budget, cfg)
            elapsed = time.perf_counter() - t0
            units = _units(pool, result["mask"])
            hit, total, fhit, ftotal = _coverage(units, spans)
            row = {
                "instance_id": inst.instance_id,
                "cost_lambda": lam,
                "seed": seed,
                "tokens": sum(u["token_cost"] for u in units),
                "n_units": len(units),
                "fitness": result.get("fitness"),
                "gold_spans_hit": hit,
                "gold_spans_total": total,
                "gold_files_hit": fhit,
                "gold_files_total": ftotal,
                "seconds": elapsed,
            }
            rows.append(row)
            append_jsonl(out, row)
        print(f"[{n}/{len(instances)}] {inst.instance_id}", flush=True)

    print(f"\n{'lambda':>7} {'tokens':>8} {'util':>7} {'units':>6} {'gold_spans':>12} {'gold_files':>12} {'full_cov':>9}")
    for lam in lambdas:
        sel = [r for r in rows if r["cost_lambda"] == lam]
        if not sel:
            continue
        tok = st.mean(r["tokens"] for r in sel)
        hit = sum(r["gold_spans_hit"] for r in sel)
        tot = sum(r["gold_spans_total"] for r in sel)
        fhit = sum(r["gold_files_hit"] for r in sel)
        ftot = sum(r["gold_files_total"] for r in sel)
        full = sum(
            1 for r in sel if r["gold_spans_total"] and r["gold_spans_hit"] == r["gold_spans_total"]
        )
        print(
            f"{lam:7.2f} {tok:8.0f} {tok / args.budget:7.1%} "
            f"{st.mean(r['n_units'] for r in sel):6.1f} "
            f"{hit:5}/{tot:<6} {fhit:5}/{ftot:<6} {full:4}/{len(sel):<4}"
        )
    print(f"\nrows -> {out}")


if __name__ == "__main__":
    main()
