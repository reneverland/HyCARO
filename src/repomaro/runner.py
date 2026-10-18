from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from .baselines import BinaryPSO, GenericMARO, bm25_baseline, dense_baseline, greedy_utility, mmr_baseline, rrf_baseline
from .fitness import SurrogateFitness
from .optimizer import RepoMARO
from .schema import CandidatePool


BUILTIN_METHODS = [
    "bm25",
    "dense",
    "rrf",
    "mmr",
    "greedy",
    "binary_pso",
    "generic_maro",
    "repomaro",
    "node_repomaro",
    "repo_aro",
    "bundle_aro",
    "fast_aro",
    "hycaro",
    "repograph",
    "greedy_1swap",
    "greedy_f",
    "seed_only",
    "two_unit",
    "de_repair",
    "relation_greedy",
    "repomap",
    "longcodezip",
    "locagent",
]


def _run_hycaro(method: str, pool: CandidatePool, budget: int, fcfg: dict, ocfg: dict, seed: int) -> dict[str, Any]:
    from .hycaro import HyCARO

    if method == "seed_only":
        ocfg = {**ocfg, "hycaro_iterations": 0}
    opt = HyCARO(pool, budget, fcfg, ocfg, seed)
    r = opt.optimize()
    return {
        "method": "Seed-only" if method == "seed_only" else r.method,
        "mask": r.mask,
        "fitness": r.fitness,
        "metadata": {
            "iterations": r.iterations,
            "cache_hits": r.cache_hits,
            "cache_misses": r.cache_misses,
            "n_accepted": r.n_accepted,
            "n_proposals": r.n_proposals,
            "n_selected": int(np.asarray(r.mask).sum()),
            "search_seconds": r.search_seconds,
            "fitness_init_seconds": r.fitness_init_seconds,
            "ball_size": r.ball_size,
            "n_witness_pairs": r.n_witness_pairs,
            "n_groups": r.n_groups,
            "stopped": r.stopped,
            "history": r.history,
        },
    }


def run_selector(method: str, pool: CandidatePool, budget: int, cfg: dict[str, Any]) -> dict[str, Any]:
    fcfg = cfg["fitness"]
    ocfg = cfg["optimizer"]
    seed = int(cfg.get("seed", 42))
    method = method.lower()
    controlled = {"repograph", "greedy_1swap", "greedy_f", "two_unit", "de_repair"}
    retrieval = {"bm25", "dense", "rrf", "mmr"}
    if method in controlled or (bool(ocfg.get("restrict_to_ball")) and method in retrieval):
        from .controlled import run_controlled

        return run_controlled(method, pool, budget, fcfg, ocfg, seed)
    if method in ("hycaro", "seed_only", "relation_greedy"):
        if method == "relation_greedy":
            ocfg = {**ocfg, "hycaro_search": "greedy"}
        return _run_hycaro(method, pool, budget, fcfg, ocfg, seed)
    fitness = SurrogateFitness(
        pool,
        alpha=float(fcfg.get("alpha", 0.45)),
        beta=float(fcfg.get("beta", 0.35)),
        gamma=float(fcfg.get("gamma", 0.20)),
        sigma=float(fcfg.get("sigma", 2.0)),
    )
    if method == "bm25":
        r = bm25_baseline(pool, budget, fitness)
        return {"method": r.name, "mask": r.mask, "fitness": r.fitness, "metadata": r.metadata}
    if method == "dense":
        r = dense_baseline(pool, budget, fitness)
        return {"method": r.name, "mask": r.mask, "fitness": r.fitness, "metadata": r.metadata}
    if method == "rrf":
        r = rrf_baseline(pool, budget, fitness)
        return {"method": r.name, "mask": r.mask, "fitness": r.fitness, "metadata": r.metadata}
    if method == "mmr":
        r = mmr_baseline(pool, budget, fitness)
        return {"method": r.name, "mask": r.mask, "fitness": r.fitness, "metadata": r.metadata}
    if method == "greedy":
        r = greedy_utility(pool, budget, fitness)
        return {"method": r.name, "mask": r.mask, "fitness": r.fitness, "metadata": r.metadata}
    if method == "binary_pso":
        r = BinaryPSO(pool, budget, fitness, int(ocfg.get("population_size", 20)), int(ocfg.get("max_iterations", 40)), seed).run()
        return {"method": r.name, "mask": r.mask, "fitness": r.fitness, "metadata": r.metadata}
    if method == "generic_maro":
        r = GenericMARO(pool, budget, fitness, int(ocfg.get("population_size", 20)), int(ocfg.get("max_iterations", 40)), seed).run()
        return {"method": r.name, "mask": r.mask, "fitness": r.fitness, "metadata": r.metadata}
    if method == "repomap":
        from .baselines_external import ranking_path_from_cfg, repomap_baseline

        path = ranking_path_from_cfg(cfg, "repomap_ranking")
        if not path:
            raise ValueError(
                "repomap requires --external-ranking or external.repomap_ranking; "
                "produce it with scripts/external/run_repomap_ranking.py"
            )
        r = repomap_baseline(pool, budget, fitness, path)
        return {"method": r.name, "mask": r.mask, "fitness": r.fitness, "metadata": r.metadata}
    if method == "longcodezip":
        from .baselines_external import longcodezip_baseline, ranking_path_from_cfg

        path = ranking_path_from_cfg(cfg, "longcodezip_ranking")
        if not path:
            raise ValueError(
                "longcodezip requires --external-ranking; "
                "produce it with scripts/external/run_longcodezip.py"
            )
        r = longcodezip_baseline(pool, budget, fitness, path)
        return {"method": r.name, "mask": r.mask, "fitness": r.fitness, "metadata": r.metadata}
    if method == "locagent":
        from .baselines_external import locagent_baseline, ranking_path_from_cfg

        path = ranking_path_from_cfg(cfg, "locagent_ranking")
        if not path:
            raise ValueError(
                "locagent requires --external-ranking; "
                "produce it with scripts/external/run_locagent_localize.py"
            )
        r = locagent_baseline(pool, budget, fitness, path)
        return {"method": r.name, "mask": r.mask, "fitness": r.fitness, "metadata": r.metadata}
    if method in ("node_repomaro", "node-repomaro"):
        ocfg = dict(ocfg)
        ocfg["unrestricted_pool"] = True
        opt = RepoMARO(pool, budget, fcfg, ocfg, seed)
        r = opt.optimize()
        return {
            "method": "Node-RepoMARO",
            "mask": r.mask,
            "fitness": r.fitness,
            "metadata": {
                "iterations": r.iterations,
                "cache_hits": r.cache_hits,
                "cache_misses": r.cache_misses,
                "local_search_triggers": r.local_search_triggers,
                "ball_anchors": opt.ball_k,
                "ball_radius": opt.ball_d,
                "ball_size": int(opt.member.sum()),
                "unrestricted_pool": True,
                "n_selected": int(np.asarray(r.mask).sum()),
                "history": r.history,
            },
        }
    if method in ("repo_aro", "bundle_aro", "fast_aro"):
        from .aro import BundleARO, FastARO, RepoARO as _RepoARO

        cls_map = {"repo_aro": _RepoARO, "bundle_aro": BundleARO, "fast_aro": FastARO}
        opt = cls_map[method](pool, budget, fcfg, ocfg, seed)
        r = opt.optimize()
        return {
            "method": r.method,
            "mask": r.mask,
            "fitness": r.fitness,
            "metadata": {
                "iterations": r.iterations,
                "cache_hits": r.cache_hits,
                "cache_misses": r.cache_misses,
                "n_bundles_tried": r.n_bundles_tried,
                "n_bundles_accepted": r.n_bundles_accepted,
                "n_selected": int(np.asarray(r.mask).sum()),
                "history": r.history,
            },
        }
    if method in ("repomaro", "ball_repomaro"):
        opt = RepoMARO(pool, budget, fcfg, ocfg, seed)
        r = opt.optimize()
        return {
            "method": "RepoMARO",
            "mask": r.mask,
            "fitness": r.fitness,
            "metadata": {
                "iterations": r.iterations,
                "cache_hits": r.cache_hits,
                "cache_misses": r.cache_misses,
                "local_search_triggers": r.local_search_triggers,
                "ball_anchors": opt.ball_k,
                "ball_radius": opt.ball_d,
                "ball_size": int(opt.member.sum()),
                "n_selected": int(np.asarray(r.mask).sum()),
                "history": r.history,
            },
        }
    raise ValueError(f"Unknown method {method!r}; choose from {BUILTIN_METHODS}")
