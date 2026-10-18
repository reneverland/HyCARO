#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import pickle
import time
from collections import defaultdict
from pathlib import Path

from repomaro.adapters.swebench import load_swebench_lite, write_swebench_predictions
from repomaro.candidate_pool import CandidatePoolBuilder
from repomaro.config import load_config
from repomaro.downstream import OpenAIPatchGenerator, serialize_context
from repomaro.io_utils import append_jsonl, selector_record
from repomaro.oracle import force_gold_mask
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
    "hycaro": "HyCARO",
    "repograph": "RepoGraph",
    "greedy_1swap": "Greedy+1swap",
    "two_unit": "Two-unit",
    "de_repair": "DE+repair",
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


def _write_predictions(out: Path, predictions_path: Path, model: str) -> None:
    by_method_sample: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    if not out.exists():
        return
    with out.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            gen = rec.get("generation") or {}
            samples = gen.get("samples")
            if not samples:
                patch = gen.get("patch")
                samples = [{"patch": patch}] if patch else []
            for si, sample in enumerate(samples):
                patch = (sample or {}).get("patch") or ""
                if not patch:
                    continue
                by_method_sample[(rec["method"], si)].append({
                    "instance_id": rec["instance_id"],
                    "model_patch": patch,
                    "model_name_or_path": f"{model}+{rec['method']}+s{si}",
                })
    root = Path(predictions_path)
    for (method, si), rows in by_method_sample.items():
        dest = root.with_name(f"{root.stem}_{method}_s{si}{root.suffix}")
        write_swebench_predictions(rows, dest)
        if si == 0:
            write_swebench_predictions(rows, root.with_name(f"{root.stem}_{method}{root.suffix}"))


def _build_pool(builder, inst, budget: int, cfg: dict, cache_dir: Path | None):
    if cache_dir is None:
        return builder.build(inst.query, inst.repo_root, budget, cfg["fitness"])
    path = cache_dir / f"{inst.instance_id}__{budget}.pkl"
    if path.exists():
        try:
            return pickle.loads(path.read_bytes())
        except Exception as exc:  # a truncated cache entry must not kill the run
            print(f"  pool cache unreadable for {inst.instance_id}: {exc}", flush=True)
    pool = builder.build(inst.query, inst.repo_root, budget, cfg["fitness"])
    tmp = path.with_suffix(".pkl.tmp")
    tmp.write_bytes(pickle.dumps(pool))
    tmp.replace(path)
    return pool


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkout-cache", default="cache/checkouts")
    ap.add_argument("--config", default="configs/hycaro.yaml")
    ap.add_argument("--method", choices=BUILTIN_METHODS, help="single method; overrides --methods")
    ap.add_argument("--methods", default="bm25,dense,rrf,mmr,repograph,greedy_1swap,two_unit,de_repair,hycaro")
    ap.add_argument("--budget", type=int, default=8192)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--offset", type=int, default=0, help="Skip the first N Lite instances before applying --limit")
    ap.add_argument(
        "--instance-ids",
        nargs="+",
        help="Only these SWE-bench instance IDs (ignores --offset/--limit when set)",
    )
    ap.add_argument("--generate", action="store_true", help="Call downstream OpenAI model and produce SWE-bench patches")
    ap.add_argument(
        "--n-samples",
        type=int,
        default=1,
        help="Independent patches per instance×method for pass@k (official harness still scores each sample as pass@1)",
    )
    ap.add_argument("--out", default="results/swebench_selection.jsonl")
    ap.add_argument("--predictions", default="results/swebench_predictions.json")
    ap.add_argument(
        "--pool-cache",
        default="",
        help="Reuse candidate pools across runs. Pools depend only on the repository, "
        "the issue text and the budget, so selector changes can be re-measured without rescoring.",
    )
    ap.add_argument(
        "--repomap-ranking",
        default="",
        help="JSONL of precomputed aider RepoMap rankings, required by --methods repomap "
        "(see scripts/external/run_repomap_ranking.py)",
    )
    ap.add_argument(
        "--external-ranking",
        default="",
        help="JSONL of precomputed external rankings (RepoMap / LongCodeZip / LocAgent). "
        "Used by whichever external method is in --methods.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        help="Override the config seed. Only the stochastic selectors (repomaro, "
        "binary_pso, generic_maro) consume it; the retrieval baselines are "
        "deterministic given the pool and ignore it.",
    )
    ap.add_argument(
        "--cost-lambda",
        type=float,
        help="Override fitness.ball.cost_lambda, the token price in F. Lets a grid "
        "sweep run from one config instead of a file per grid point.",
    )
    ap.add_argument(
        "--oracle-gold",
        action="store_true",
        help="DIAGNOSTIC UPPER BOUND, not a method: force the reference-patch units into "
        "the selected context (same budget) to test whether context is what limits the "
        "downstream model. Results from this flag must never be reported as a method.",
    )
    args = ap.parse_args()

    if args.oracle_gold:
        print(
            "!! --oracle-gold is ON: contexts are built from the reference patch. "
            "These numbers are an upper bound, not a method result.",
            flush=True,
        )

    cfg = copy.deepcopy(load_config(args.config))
    cfg["candidate_pool"]["granularity"] = "symbol"
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.cost_lambda is not None:
        cfg.setdefault("fitness", {}).setdefault("ball", {})["cost_lambda"] = args.cost_lambda
    if args.repomap_ranking:
        cfg.setdefault("external", {})["repomap_ranking"] = args.repomap_ranking
    if args.external_ranking:
        cfg.setdefault("external", {})["ranking"] = args.external_ranking
    methods = [args.method] if args.method else [x.strip() for x in args.methods.split(",") if x.strip()]
    unknown = set(methods) - set(BUILTIN_METHODS)
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")

    print("Loading SWE-bench Lite (cloning missing snapshots)...", flush=True)
    instances = load_swebench_lite(
        args.checkout_cache,
        None if args.instance_ids else args.limit,
        offset=0 if args.instance_ids else args.offset,
        instance_ids=args.instance_ids,
    )
    out = Path(args.out)
    done = _load_done(out)
    needed = {RECORD_NAME[m] for m in methods}
    n_done = sum(1 for inst in instances if needed <= done[inst.instance_id])
    print(
        f"SWE-bench Lite @{args.budget}: {len(instances)} instances, "
        f"skip {n_done} finished, methods={methods}, generate={args.generate}, n_samples={args.n_samples}",
        flush=True,
    )

    downstream_cfg = cfg.get("downstream", {}) or {}
    generator = None
    if args.generate:
        generator = OpenAIPatchGenerator(
            model=str(downstream_cfg.get("model", "gpt-5.6-luna")),
            reasoning_effort=str(downstream_cfg.get("reasoning_effort", "medium")),
            max_output_tokens=int(downstream_cfg.get("max_output_tokens", 12000)),
            apply_retries=int(downstream_cfg.get("apply_retries", 1)),
            temperature=downstream_cfg.get("temperature"),
            request_timeout=float(downstream_cfg.get("request_timeout", 900)),
            timeout_retries=int(downstream_cfg.get("timeout_retries", 2)),
        )
    model_name = str(downstream_cfg.get("model", "openai"))
    builder = CandidatePoolBuilder(cfg["candidate_pool"], downstream_cfg.get("model"))

    pool_cache = Path(args.pool_cache) if args.pool_cache else None
    if pool_cache:
        pool_cache.mkdir(parents=True, exist_ok=True)

    pending = [inst for inst in instances if not (needed <= done[inst.instance_id])]
    for n, inst in enumerate(pending, 1):
        pool = _build_pool(builder, inst, args.budget, cfg, pool_cache)
        for method in methods:
            record_name = RECORD_NAME[method]
            if record_name in done[inst.instance_id]:
                continue
            t0 = time.perf_counter()
            pool.metadata["instance_id"] = inst.instance_id
            result = run_selector(method, pool, args.budget, cfg)
            selector_time = time.perf_counter() - t0
            selector_time += float((result.get("metadata") or {}).get("rank_seconds") or 0.0)
            if args.oracle_gold:
                result["mask"], oracle_info = force_gold_mask(
                    pool, result["mask"], inst.gold_spans, args.budget
                )
                result["metadata"] = {**(result.get("metadata") or {}), "oracle_gold": oracle_info}
            rec = selector_record(inst.instance_id, pool, result)
            rec.update({
                "benchmark": "SWE-bench Lite",
                "budget": args.budget,
                "selector_time_seconds": selector_time,
                # A sweep writes one file per (cost_lambda, seed); recording both
                # here keeps a row interpretable once the files are merged.
                "seed": int(cfg.get("seed", 42)),
                "cost_lambda": float(
                    ((cfg.get("fitness") or {}).get("ball") or {}).get("cost_lambda", 0.25)
                ),
                "temperature": downstream_cfg.get("temperature"),
            })
            if generator is not None:
                context = serialize_context(pool, result["mask"])
                samples = []
                for _ in range(max(int(args.n_samples), 1)):
                    try:
                        patch, gen_meta = generator.generate(
                            inst.query, context, repo_root=inst.repo_root
                        )
                    except Exception as exc:
                        print(
                            f"  generate failed {inst.instance_id} {record_name}: {exc}",
                            flush=True,
                        )
                        patch, gen_meta = "", {
                            "model": generator.model,
                            "error": str(exc),
                        }
                    sample = dict(gen_meta)
                    sample["patch"] = patch
                    samples.append(sample)
                rec["generation"] = {
                    "model": samples[0].get("model") if samples else None,
                    "n_samples": len(samples),
                    "patch": samples[0]["patch"] if samples else "",
                    "samples": samples,
                    **({"error": samples[0].get("error")} if samples and samples[0].get("error") else {}),
                }
            append_jsonl(out, rec)
            done[inst.instance_id].add(rec["method"])
        if generator is not None:
            _write_predictions(out, Path(args.predictions), model_name)
        print(f"[{n}/{len(pending)}] {inst.instance_id}", flush=True)

    if generator is not None:
        _write_predictions(out, Path(args.predictions), model_name)
        print(f"SWE-bench predictions under {Path(args.predictions).with_name(Path(args.predictions).stem + '_*')}")


if __name__ == "__main__":
    main()
