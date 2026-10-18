#!/usr/bin/env python
"""Build and cache candidate pools ahead of a sweep.

A pool depends only on the repository snapshot, the issue text and the budget,
so every cost_lambda grid point reuses the same file. Building them inside the
sweep would serialise the GPU embedder against 20 generation shards; building
them once here keeps the sweep itself CPU- and network-bound.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from repomaro.adapters.swebench import load_swebench_lite
from repomaro.candidate_pool import CandidatePoolBuilder
from repomaro.config import load_config
from run_swebench import _build_pool


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--checkout-cache", default="cache/checkouts")
    ap.add_argument("--pool-cache", default="cache/pools_v3")
    ap.add_argument("--budget", type=int, default=8192)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument(
        "--batch-size",
        type=int,
        help="Override candidate_pool.dense.batch_size. The shipped value of 1 costs "
        "~20 min per pool; sentence-transformers pads and masks per batch, so a larger "
        "batch changes the embeddings only at fp noise level.",
    )
    ap.add_argument("--stride", type=int, default=1, help="Take every Nth pending instance (worker sharding)")
    ap.add_argument("--worker", type=int, default=0, help="Offset within --stride")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["candidate_pool"]["granularity"] = "symbol"
    if args.batch_size:
        cfg["candidate_pool"]["dense"]["batch_size"] = args.batch_size
    cache = Path(args.pool_cache)
    cache.mkdir(parents=True, exist_ok=True)

    instances = load_swebench_lite(args.checkout_cache, args.limit, offset=args.offset)
    todo = [i for i in instances if not (cache / f"{i.instance_id}__{args.budget}.pkl").exists()]
    todo = todo[args.worker :: args.stride]
    print(f"pools: {len(instances)} instances, {len(todo)} for this worker", flush=True)
    if not todo:
        return

    import torch

    batch = int(cfg["candidate_pool"]["dense"]["batch_size"])
    builder = CandidatePoolBuilder(cfg["candidate_pool"], (cfg.get("downstream") or {}).get("model"))
    for n, inst in enumerate(todo, 1):
        t0 = time.perf_counter()
        pool = None
        # The biggest repositories need a smaller batch than the rest, and the
        # workers share one card, so back off per instance instead of picking a
        # batch small enough for the worst case.
        for attempt_batch in (batch, max(batch // 4, 4), 1):
            try:
                builder.dense.batch_size = attempt_batch
                pool = _build_pool(builder, inst, args.budget, cfg, cache)
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"[{n}/{len(todo)}] OOM {inst.instance_id} at batch={attempt_batch}", flush=True)
            except Exception as exc:  # one broken snapshot must not stop the prebuild
                print(f"[{n}/{len(todo)}] FAIL {inst.instance_id}: {exc}", flush=True)
                break
        builder.dense.batch_size = batch
        if pool is None:
            continue
        print(
            f"[{n}/{len(todo)}] {inst.instance_id} units={len(pool.units)} "
            f"batch={attempt_batch} {time.perf_counter() - t0:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
