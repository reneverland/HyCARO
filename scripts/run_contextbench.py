#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import pickle
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from repomaro.adapters.contextbench import (
    load_contextbench_verified,
    selection_to_contextbench_record,
    write_contextbench_predictions,
)
from repomaro.candidate_pool import CandidatePoolBuilder
from repomaro.checkout import ensure_checkout
from repomaro.config import load_config
from repomaro.io_utils import append_jsonl, selector_record
from repomaro.runner import BUILTIN_METHODS, run_selector

RECORD_NAME = {
    "bm25": "BM25",
    "dense": "Dense",
    "rrf": "RRF",
    "mmr": "MMR",
    "greedy": "Greedy-Utility",
    "repomap": "RepoMap",
    "longcodezip": "LongCodeZip (coarse)",
    "locagent": "LocAgent",
    "binary_pso": "Binary-PSO",
    "generic_maro": "Generic-MARO",
    "repomaro": "RepoMARO",
    "ball_repomaro": "RepoMARO",
    "node_repomaro": "Node-RepoMARO",
    "repo_aro": "RepoARO",
    "bundle_aro": "BundleARO",
    "fast_aro": "FastARO",
    "hycaro": "HyCARO",
    "repograph": "RepoGraph",
    "greedy_1swap": "Greedy+1swap",
    "two_unit": "Two-unit",
    "de_repair": "DE+repair",
    "seed_only": "Seed-only",
    "relation_greedy": "Relation-guided greedy",
}


def _load_done(path: Path) -> dict[str, set[str]]:
    done: dict[str, set[str]] = defaultdict(set)
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            done[rec["instance_id"]].add(rec["method"])
    return done


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


def _unit_view(u: dict):
    return type("Unit", (), u)


def _record_from_saved_units(rec: dict) -> dict:
    units = [_unit_view(u) for u in rec.get("selected_units") or []]
    return selection_to_contextbench_record(rec["instance_id"], units)


def _write_official(out: Path, dest: Path, method_name: str) -> None:
    if not out.exists():
        return
    rows = []
    with out.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("method") != method_name:
                continue
            rows.append(_record_from_saved_units(rec))
    write_contextbench_predictions(rows, dest)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkout-cache", default="cache/checkouts")
    ap.add_argument("--config", default="configs/hycaro.yaml")
    ap.add_argument("--method", choices=BUILTIN_METHODS, help="single method; overrides --methods")
    ap.add_argument("--methods", default="bm25,dense,rrf,mmr,repograph,greedy_1swap,two_unit,de_repair,hycaro")
    ap.add_argument("--budget", type=int, default=8192)
    ap.add_argument("--seed", type=int, help="override config seed for this run")
    ap.add_argument(
        "--deadline",
        type=float,
        help="override selection deadline in seconds for every search method",
    )
    ap.add_argument("--limit", type=int)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--out", default="results/contextbench_selection.jsonl")
    ap.add_argument("--official-pred", default="results/contextbench_official_pred.jsonl")
    ap.add_argument("--pool-cache", default="cache/pools_cb_v1")
    ap.add_argument("--external-ranking", default="")
    ap.add_argument("--gold-parquet", default="cache/contextbench_verified.parquet")
    ap.add_argument("--skip-log", default="results/contextbench_lite/skipped.jsonl")
    ap.add_argument("--checkout-only", action="store_true")
    ap.add_argument(
        "--prefetch",
        action="store_true",
        help="build the next pool while the current instance is being selected",
    )
    args = ap.parse_args()

    cfg = copy.deepcopy(load_config(args.config))
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.deadline is not None:
        opt = cfg.setdefault("optimizer", {})
        opt["selection_deadline_s"] = float(args.deadline)
        opt["hycaro_time_budget_s"] = float(args.deadline)
    cfg["candidate_pool"]["granularity"] = "symbol"
    if args.external_ranking:
        cfg.setdefault("external", {})["ranking"] = args.external_ranking
    methods = [args.method] if args.method else [x.strip() for x in args.methods.split(",") if x.strip()]
    unknown = set(methods) - set(BUILTIN_METHODS)
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")

    print("Loading ContextBench Verified (Lite, 500)...", flush=True)
    instances = load_contextbench_verified(
        args.checkout_cache,
        None,
        gold_parquet=args.gold_parquet,
        skip_log=args.skip_log,
        defer_checkout=not args.checkout_only,
    )
    if args.offset or args.limit:
        end = len(instances) if args.limit is None else min(len(instances), args.offset + args.limit)
        instances = instances[args.offset:end]
    print(f"ready {len(instances)} instances", flush=True)
    if args.checkout_only:
        return

    out = Path(args.out)
    done = _load_done(out)
    needed = {RECORD_NAME[m] for m in methods}
    n_done = sum(1 for inst in instances if needed <= done[inst.instance_id])
    print(
        f"ContextBench @{args.budget}: {len(instances)} instances, "
        f"skip {n_done} finished, methods={methods}",
        flush=True,
    )

    builder = CandidatePoolBuilder(cfg["candidate_pool"], (cfg.get("downstream") or {}).get("model"))
    pool_cache = Path(args.pool_cache) if args.pool_cache else None
    if pool_cache:
        pool_cache.mkdir(parents=True, exist_ok=True)

    pending = [inst for inst in instances if not (needed <= done[inst.instance_id])]

    def _prepare(inst):
        repo = str((inst.metadata or {}).get("repo") or "")
        commit = str((inst.metadata or {}).get("base_commit") or "")
        print(f"  checkout {repo}@{commit[:10]} {inst.instance_id}", flush=True)
        inst.repo_root = ensure_checkout(repo, commit, args.checkout_cache)
        pool = _build_pool(builder, inst, args.budget, cfg, pool_cache)
        return inst, pool

    def _select(n: int, inst, pool) -> None:
        for method in methods:
            record_name = RECORD_NAME[method]
            if record_name in done[inst.instance_id]:
                continue
            try:
                t0 = time.perf_counter()
                pool.metadata["instance_id"] = inst.instance_id
                result = run_selector(method, pool, args.budget, cfg)
                elapsed = time.perf_counter() - t0
                elapsed += float((result.get("metadata") or {}).get("rank_seconds") or 0.0)
                rec = selector_record(inst.instance_id, pool, result)
                rec.update({
                    "benchmark": "ContextBench Lite",
                    "budget": args.budget,
                    "selector_time_seconds": elapsed,
                    "seed": int(cfg.get("seed", 42)),
                    "original_inst_id": (inst.metadata or {}).get("original_inst_id"),
                    "language": (inst.metadata or {}).get("language"),
                })
                append_jsonl(out, rec)
                done[inst.instance_id].add(rec["method"])
                print(
                    f"    {rec['method']} n={int(result['mask'].sum())} "
                    f"tok={rec['selected_tokens']} {elapsed:.2f}s",
                    flush=True,
                )
            except Exception as exc:
                print(f"  FAIL {method} {inst.instance_id}: {exc}", flush=True)
                traceback.print_exc()
                append_jsonl(args.skip_log, {
                    "instance_id": inst.instance_id,
                    "method": method,
                    "reason": f"select: {exc}",
                })
        print(f"[{n}/{len(pending)}] {inst.instance_id}", flush=True)

    if not args.prefetch or len(pending) <= 1:
        for n, inst in enumerate(pending, 1):
            try:
                inst, pool = _prepare(inst)
            except Exception as exc:
                detail = getattr(exc, "stderr", None) or str(exc)
                print(f"  FAIL prepare {inst.instance_id}: {detail}", flush=True)
                traceback.print_exc()
                append_jsonl(args.skip_log, {"instance_id": inst.instance_id, "reason": f"prepare: {detail}"})
                continue
            _select(n, inst, pool)
    else:
        with ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(_prepare, pending[0])
            for n, inst in enumerate(pending):
                try:
                    inst, pool = future.result()
                except Exception as exc:
                    detail = getattr(exc, "stderr", None) or str(exc)
                    print(f"  FAIL prepare {inst.instance_id}: {detail}", flush=True)
                    traceback.print_exc()
                    append_jsonl(args.skip_log, {"instance_id": inst.instance_id, "reason": f"prepare: {detail}"})
                    pool = None
                if n + 1 < len(pending):
                    future = ex.submit(_prepare, pending[n + 1])
                if pool is None:
                    continue
                _select(n + 1, inst, pool)

    dest = Path(args.official_pred)
    for method in methods:
        name = RECORD_NAME[method]
        method_dest = dest.with_name(f"{dest.stem}_{name}{dest.suffix}")
        _write_official(out, method_dest, name)
        print(f"official pred {name} -> {method_dest} ({_count_method(out, name)} rows)", flush=True)


def _count_method(out: Path, method_name: str) -> int:
    if not out.exists():
        return 0
    n = 0
    with out.open(encoding="utf-8") as f:
        for line in f:
            if line.strip() and json.loads(line).get("method") == method_name:
                n += 1
    return n


if __name__ == "__main__":
    main()
