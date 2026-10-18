#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import pickle
import time
from collections import defaultdict
from pathlib import Path

from repomaro.adapters.arb import load_arb_samples
from repomaro.candidate_pool import CandidatePoolBuilder
from repomaro.config import load_config
from repomaro.io_utils import append_jsonl, selector_record
from repomaro.metrics import file_prf, gold_coverage, selected_paths
from repomaro.runner import BUILTIN_METHODS, run_selector

RECORD_NAME = {
    "bm25": "BM25",
    "dense": "Dense",
    "rrf": "RRF",
    "mmr": "MMR",
    "greedy": "Greedy-Utility",
    "binary_pso": "Binary-PSO",
    "generic_maro": "Generic-MARO",
    "repomaro": "RepoMARO",
    "hycaro": "HyCARO",
    "repograph": "RepoGraph",
    "greedy_1swap": "Greedy+1swap",
    "two_unit": "Two-unit",
    "de_repair": "DE+repair",
}


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


def _ranked_files(pool, mask) -> list[str]:
    best: dict[str, float] = {}
    for unit, keep in zip(pool.units, mask.tolist() if hasattr(mask, "tolist") else list(mask)):
        if not keep:
            continue
        rel = float(getattr(unit, "relevance", 0.0))
        if unit.path not in best or rel > best[unit.path]:
            best[unit.path] = rel
    return sorted(best, key=lambda path: (-best[path], path))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", nargs="+", required=True, help="ARB samples.jsonl files")
    ap.add_argument("--checkout-cache", default="cache/checkouts")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--methods", default="bm25,dense,rrf,mmr,greedy,binary_pso,generic_maro,repomaro")
    ap.add_argument("--budget", type=int, default=8192)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--pool-cache", default="")
    ap.add_argument("--out", default="results/arb.jsonl")
    args = ap.parse_args()

    cfg = load_config(args.config)
    # ARB's canonical primary unit is file-level retrieval.
    cfg = copy.deepcopy(cfg)
    cfg["candidate_pool"]["granularity"] = "file"
    # Workflow/docs files steal budget on file-level BCY; keep them out of ARB pools.
    skip_ext = {".yml", ".yaml", ".md"}
    exts = cfg["candidate_pool"].get("include_extensions") or []
    cfg["candidate_pool"]["include_extensions"] = [e for e in exts if e.lower() not in skip_ext]
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    unknown = set(methods) - set(BUILTIN_METHODS)
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")
    missing_names = set(methods) - set(RECORD_NAME)
    if missing_names:
        raise ValueError(f"No record name for: {sorted(missing_names)}")
    needed = {RECORD_NAME[m] for m in methods}
    print(f"Loading {len(args.samples)} sample files (cloning missing snapshots)...", flush=True)
    instances = load_arb_samples(
        args.samples, args.checkout_cache, args.limit, offset=args.offset
    )
    out = Path(args.out)
    done: dict[str, set[str]] = defaultdict(set)
    if out.exists():
        with out.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                done[rec["instance_id"]].add(rec["method"])
        n_done = sum(1 for inst in instances if needed <= done[inst.instance_id])
        print(f"Resuming {out}: skipping {n_done} finished instances", flush=True)

    builder = CandidatePoolBuilder(cfg["candidate_pool"], (cfg.get("downstream") or {}).get("model"))
    pool_cache = Path(args.pool_cache) if args.pool_cache else None
    if pool_cache:
        pool_cache.mkdir(parents=True, exist_ok=True)
    pending = [inst for inst in instances if not (needed <= done[inst.instance_id])]
    print(
        f"ARB @{args.budget}: {len(pending)}/{len(instances)} instances, "
        f"offset={args.offset}, methods={methods}",
        flush=True,
    )

    for n, inst in enumerate(pending, 1):
        pool = _build_pool(builder, inst, args.budget, cfg, pool_cache)
        for method in methods:
            if RECORD_NAME[method] in done[inst.instance_id]:
                continue
            t0 = time.perf_counter()
            result = run_selector(method, pool, args.budget, cfg)
            elapsed = time.perf_counter() - t0
            rec = selector_record(inst.instance_id, pool, result)
            paths = selected_paths(pool, result["mask"])
            ranked = _ranked_files(pool, result["mask"])
            prf = file_prf(paths, inst.gold_paths)
            rec.update({
                "benchmark": "Agent Retrieval Bench",
                "budget": args.budget,
                "gold_paths": inst.gold_paths,
                "ranked_files": ranked,
                "BCY": gold_coverage(paths, inst.gold_paths),
                "file_precision": prf.precision,
                "file_recall": prf.recall,
                "file_f1": prf.f1,
                "selector_time_seconds": elapsed,
                "task_type": inst.metadata.get("task_type"),
                "repo": inst.metadata.get("repo"),
                "base_commit": inst.metadata.get("base_commit"),
            })
            append_jsonl(out, rec)
            done[inst.instance_id].add(rec["method"])
        print(f"[{n}/{len(pending)}] {inst.instance_id}", flush=True)


if __name__ == "__main__":
    main()
