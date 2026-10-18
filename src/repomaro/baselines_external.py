"""Baselines whose ranking is produced by an external tool in its own venv.

These methods pin tree-sitter / torch versions that conflict with ours, so their
ranking is precomputed by a script under `scripts/external/` and consumed here.
Only the ranking crosses the boundary: the units, the serialization and the token
budget stay ours, so a comparison isolates the ranking signal instead of also
varying the chunking and packing scheme.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .baselines import BaselineResult
from .fitness import SurrogateFitness
from .retrieval import greedy_pack_by_ranking
from .schema import CandidatePool

# Ranked units come first; everything the external tool never mentioned is
# ordered behind them by unit index, so budget filling stays deterministic and
# does not smuggle our own relevance signal into the baseline.
_UNRANKED_OFFSET = 10**7


@lru_cache(maxsize=8)
def _load_rankings(path: str) -> dict[str, tuple[tuple[dict, ...], float]]:
    """Rankings keyed by `root::query` when both are present, else by root.

    Each value is ``(ranking, rank_seconds)``. ``rank_seconds`` is the external
    scorer's own time and is added to selector time by the benchmark runners.
    """
    out: dict[str, tuple[tuple[dict, ...], float]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        ranking = tuple(row.get("ranking") or ())
        seconds = float(row.get("rank_seconds") or 0.0)
        payload = (ranking, seconds)
        root = str(Path(row["root"]).resolve()) if row.get("root") else ""
        query = (row.get("query") or "").strip()
        if root and query:
            out[f"{root}::{query}"] = payload
        if root:
            out[root] = payload
        iid = (row.get("instance_id") or "").strip()
        if iid:
            out[f"id::{iid}"] = payload
    return out


def _lookup_ranking(pool: CandidatePool, ranking_path: str) -> tuple[tuple[dict, ...], float]:
    rankings = _load_rankings(ranking_path)
    key = str(Path(pool.repo_root).resolve())
    iid = str((pool.metadata or {}).get("instance_id") or "")
    payload = rankings.get(f"id::{iid}") if iid else None
    payload = payload or rankings.get(f"{key}::{pool.query}") or rankings.get(key)
    if payload is None:
        raise KeyError(f"no external ranking for {key} in {ranking_path}")
    return payload


def _unit_ranks(pool: CandidatePool, ranking: tuple[dict, ...]) -> np.ndarray:
    """Best external rank per unit; unmatched units get a large sentinel rank.

    A missing or non-positive `line` is a file-level hit: every unit on that
    path receives the tag's rank. LocAgent and LongCodeZip both emit that form.
    """
    m = len(pool.units)
    ranks = np.full(m, float(_UNRANKED_OFFSET), dtype=np.float64)
    by_path: dict[str, list[int]] = {}
    for j, u in enumerate(pool.units):
        path = str(getattr(u, "path", "")).replace("\\", "/")
        by_path.setdefault(path, []).append(j)
        by_path.setdefault(Path(path).name, []).append(j)

    for tag in ranking:
        raw_path = str(tag.get("path", "")).replace("\\", "/")
        line = int(tag.get("line", 0) or 0)
        r = float(tag.get("rank", _UNRANKED_OFFSET))
        hits = by_path.get(raw_path) or by_path.get(Path(raw_path).name) or ()
        for j in hits:
            if line <= 0:
                if r < ranks[j]:
                    ranks[j] = r
                continue
            u = pool.units[j]
            lo = int(getattr(u, "start_line", 0) or 0)
            hi = int(getattr(u, "end_line", 0) or 0)
            if lo <= line <= hi and r < ranks[j]:
                ranks[j] = r
    unmatched = ranks >= _UNRANKED_OFFSET
    ranks[unmatched] = _UNRANKED_OFFSET + np.flatnonzero(unmatched)
    return ranks


def _pack_external(
    name: str,
    pool: CandidatePool,
    budget: int,
    fitness: SurrogateFitness,
    ranking_path: str,
    missing_hint: str,
) -> BaselineResult:
    try:
        ranking, rank_seconds = _lookup_ranking(pool, ranking_path)
    except KeyError as exc:
        raise KeyError(f"{exc}; {missing_hint}") from exc
    ranks = _unit_ranks(pool, ranking)
    mask = greedy_pack_by_ranking(-ranks, pool.token_costs, budget)
    meta: dict[str, Any] = {
        "external_tags": len(ranking),
        "units_matched": int((ranks < _UNRANKED_OFFSET).sum()),
        "units_total": len(pool.units),
        "rank_seconds": rank_seconds,
    }
    return BaselineResult(name, mask, float(fitness.evaluate(mask)), meta)


def ranking_path_from_cfg(cfg: dict[str, Any], key: str) -> str:
    ext = cfg.get("external") or {}
    return str(ext.get(key) or ext.get("ranking") or "").strip()


def repomap_baseline(
    pool: CandidatePool,
    budget: int,
    fitness: SurrogateFitness,
    ranking_path: str,
) -> BaselineResult:
    """aider's PageRank-over-tags ranking, packed under our budget."""
    return _pack_external(
        "RepoMap",
        pool,
        budget,
        fitness,
        ranking_path,
        "run scripts/external/run_repomap_ranking.py first",
    )


def longcodezip_baseline(
    pool: CandidatePool,
    budget: int,
    fitness: SurrogateFitness,
    ranking_path: str,
) -> BaselineResult:
    """Official LongCodeZip coarse ranking, packed under our budget."""
    return _pack_external(
        "LongCodeZip (coarse)",
        pool,
        budget,
        fitness,
        ranking_path,
        "run scripts/external/run_longcodezip.py first",
    )


def locagent_baseline(
    pool: CandidatePool,
    budget: int,
    fitness: SurrogateFitness,
    ranking_path: str,
) -> BaselineResult:
    """Official LocAgent localize paths, packed under our budget."""
    return _pack_external(
        "LocAgent",
        pool,
        budget,
        fitness,
        ranking_path,
        "run scripts/external/run_locagent_localize.py first",
    )
