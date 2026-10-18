"""Artificial Rabbit Optimization (ARO) variants for repository context selection.

Three variants for fast validation:
1. RepoARO: vanilla ARO adapted to discrete subset selection with ball constraints
2. BundleARO: ARO + relation-bundle spawning ("人工催生")
3. FastARO:  BundleARO + direct set moves + incremental scoring ("最速ARO")

All operate on the same IssueCoverageFitness / anchor-ball as RepoMARO, so
surrogate quality comparisons are apples-to-apples.

Reference: L. Wang et al., "Artificial rabbits optimization: A new
meta-heuristic algorithm for solving engineering optimization problems,"
Engineering Applications of AI, 2022.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .backbone import build_anchor_balls
from .candidate_pool import seed_masks
from .decoder import BallConstrainedDecoder
from .fitness import FitnessCache, IssueCoverageFitness
from .schema import CandidatePool


@dataclass
class AROResult:
    mask: np.ndarray
    fitness: float
    iterations: int
    method: str
    history: list[dict[str, Any]] = field(default_factory=list)
    cache_hits: int = 0
    cache_misses: int = 0
    n_bundles_tried: int = 0
    n_bundles_accepted: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _relation_bundles(pool: CandidatePool, member: np.ndarray, max_bundle: int = 3) -> list[list[int]]:
    """Enumerate small relation-based unit groups within the ball.

    A "bundle" is a set of 2-3 units that share a hyperedge (e.g. a function
    and its caller, a class and its test).  Bundles whose structural support
    term only activates when ≥2 members are co-selected are the ones that
    single-unit moves cannot discover.
    """
    uid_to_idx = {u.uid: i for i, u in enumerate(pool.units)}
    graph = pool.graph
    if graph is None:
        return []
    bundles: list[list[int]] = []
    seen: set[frozenset[int]] = set()
    for edge in graph.edges:
        idxs = sorted(
            j for uid in edge.members
            if (j := uid_to_idx.get(uid)) is not None and member[j]
        )
        if len(idxs) < 2:
            continue
        # pairs
        for a in range(len(idxs)):
            for b in range(a + 1, min(a + max_bundle, len(idxs))):
                key = frozenset(idxs[a:b + 1])
                if key not in seen:
                    seen.add(key)
                    bundles.append(list(key))
        # triples (if allowed and exist)
        if max_bundle >= 3 and len(idxs) >= 3:
            for a in range(min(len(idxs) - 2, 8)):  # cap combinatorics
                key = frozenset(idxs[a:a + 3])
                if key not in seen:
                    seen.add(key)
                    bundles.append(list(key))
    return bundles


def _select_drop_candidates(
    mask: np.ndarray, costs: np.ndarray, prize: np.ndarray, n: int
) -> list[int]:
    """Indices of selected units with lowest prize/cost, for making room."""
    sel = np.flatnonzero(mask)
    if len(sel) == 0:
        return []
    ratio = prize[sel] / np.maximum(costs[sel], 1)
    order = sel[np.argsort(ratio, kind="mergesort")]
    return order[:n].tolist()


# ---------------------------------------------------------------------------
# 1. RepoARO  —  vanilla ARO in the ball
# ---------------------------------------------------------------------------

class RepoARO:
    """ARO adapted to discrete subset selection within the anchor-ball.

    Each rabbit is a continuous preference vector.  Foraging (exploitation) moves
    toward the best rabbit; hiding (exploration) makes a random detour.  Decoded
    through BallConstrainedDecoder exactly like RepoMARO.
    """

    def __init__(
        self,
        pool: CandidatePool,
        budget: int,
        fitness_cfg: dict[str, Any],
        optimizer_cfg: dict[str, Any],
        seed: int = 42,
    ):
        self.pool = pool
        self.budget = budget
        self.rng = np.random.default_rng(seed)
        self.N = int(optimizer_cfg.get("population_size", 8))
        self.Tmax = int(optimizer_cfg.get("max_iterations", 15))
        self.ball_k = int(optimizer_cfg.get("ball_anchors", 5))
        self.ball_d = int(optimizer_cfg.get("ball_radius", 1))
        self.seed_high = float(optimizer_cfg.get("seed_high", 0.90))
        self.seed_low = float(optimizer_cfg.get("seed_low", 0.10))
        self.seed_noise = float(optimizer_cfg.get("seed_noise", 0.03))
        self.enable_seeding = bool(optimizer_cfg.get("enable_retrieval_seeding", True))

        if bool(optimizer_cfg.get("unrestricted_pool", False)):
            self.member = np.ones(len(pool.units), dtype=bool)
        else:
            self.member = build_anchor_balls(pool, self.ball_k, self.ball_d)

        fcfg = {**fitness_cfg, **(fitness_cfg.get("ball") or {})}
        self.fitness = IssueCoverageFitness(
            pool, budget=budget,
            alpha=float(fcfg.get("alpha", 1.0)),
            beta=float(fcfg.get("beta", 0.35)),
            gamma=float(fcfg.get("gamma", 0.0)),
            cost_lambda=float(fcfg.get("cost_lambda", 0.12)),
            semantic_need_share=float(fcfg.get("semantic_need_share", 0.5)),
            max_lexical_needs=int(fcfg.get("max_lexical_needs", 32)),
            max_need_df_ratio=float(fcfg.get("max_need_df_ratio", 0.5)),
            semantic_need_depth=int(fcfg.get("semantic_need_depth", 4)),
            lexical_need_depth=int(fcfg.get("lexical_need_depth", 1)),
            need_depth_decay=float(fcfg.get("need_depth_decay", 0.5)),
            redundancy_delta=float(fcfg.get("redundancy_delta", 0.3)),
            support_relations=fcfg.get("support_relations"),
        )
        self.decoder = BallConstrainedDecoder(
            self.member, pool.token_costs, budget, self.fitness,
            threshold=0.5, repair=True,
        )
        self.cache = FitnessCache(self.fitness)

    def _init_population(self) -> np.ndarray:
        m = len(self.pool)
        X = self.rng.uniform(0.0, 1.0, size=(self.N, m))
        X[:, ~self.member] = 0.0
        if self.enable_seeding:
            seeds = seed_masks(self.pool, self.budget)
            for row, name in enumerate(("bm25", "dense", "struct")):
                if row >= self.N:
                    break
                vec = np.where(seeds[name] & self.member, self.seed_high, self.seed_low)
                vec = vec.astype(np.float64) + self.rng.normal(0, self.seed_noise, size=m)
                vec[~self.member] = 0.0
                X[row] = np.clip(vec, 0, 1)
        return X

    def optimize(self) -> AROResult:
        m = len(self.pool)
        X = self._init_population()
        masks = self.decoder.decode_batch(X)
        fits = self.cache.evaluate_batch(masks)

        best_idx = int(np.argmax(fits))
        best_fit = float(fits[best_idx])
        best_mask = masks[best_idx].copy()
        best_X = X[best_idx].copy()

        history = []
        for t in range(1, self.Tmax + 1):
            # energy factor decreases over time
            A = 2.0 * np.log(1.0 / self.rng.random()) * (1.0 - t / self.Tmax)
            for i in range(self.N):
                r = self.rng.random()
                if r < 0.5:
                    # Foraging: move toward best (exploitation)
                    rand_idx = self.rng.integers(0, self.N)
                    X_new = X[rand_idx] + A * (best_X - X[i])
                else:
                    # Hiding: random detour (exploration)
                    j = self.rng.integers(0, m)
                    X_new = X[i].copy()
                    R = np.zeros(m)
                    R[j] = self.rng.normal()
                    X_new = X[i] + A * R

                X_new = np.clip(X_new, 0, 1)
                X_new[~self.member] = 0.0
                mask_new = self.decoder.decode(X_new)
                fit_new = self.cache.evaluate(mask_new)

                if fit_new > fits[i]:
                    X[i] = X_new
                    masks[i] = mask_new
                    fits[i] = fit_new
                    if fit_new > best_fit:
                        best_fit = fit_new
                        best_mask = mask_new.copy()
                        best_X = X_new.copy()

            history.append({
                "iteration": t,
                "best_fitness": best_fit,
                "mean_fitness": float(np.mean(fits)),
            })

        return AROResult(
            mask=best_mask, fitness=best_fit, iterations=len(history),
            method="RepoARO", history=history,
            cache_hits=self.cache.hits, cache_misses=self.cache.misses,
        )


# ---------------------------------------------------------------------------
# 2. BundleARO  —  ARO + relation-bundle spawning
# ---------------------------------------------------------------------------

class BundleARO(RepoARO):
    """ARO with relation-bundle spawning ("人工催生").

    After each ARO iteration, the top-half rabbits try inserting a
    randomly-chosen relation bundle.  If the bundle doesn't fit, low-value
    units are dropped first.  The bundle is accepted only when the full
    surrogate improves.
    """

    def optimize(self) -> AROResult:
        m = len(self.pool)
        costs = self.pool.token_costs
        bundles = _relation_bundles(self.pool, self.member)
        n_tried = 0
        n_accepted = 0

        X = self._init_population()
        masks = self.decoder.decode_batch(X)
        fits = self.cache.evaluate_batch(masks)

        best_idx = int(np.argmax(fits))
        best_fit = float(fits[best_idx])
        best_mask = masks[best_idx].copy()
        best_X = X[best_idx].copy()

        history = []
        for t in range(1, self.Tmax + 1):
            # --- standard ARO step ---
            A = 2.0 * np.log(1.0 / self.rng.random()) * (1.0 - t / self.Tmax)
            for i in range(self.N):
                r = self.rng.random()
                if r < 0.5:
                    rand_idx = self.rng.integers(0, self.N)
                    X_new = X[rand_idx] + A * (best_X - X[i])
                else:
                    j = self.rng.integers(0, m)
                    X_new = X[i].copy()
                    R = np.zeros(m)
                    R[j] = self.rng.normal()
                    X_new = X[i] + A * R

                X_new = np.clip(X_new, 0, 1)
                X_new[~self.member] = 0.0
                mask_new = self.decoder.decode(X_new)
                fit_new = self.cache.evaluate(mask_new)

                if fit_new > fits[i]:
                    X[i] = X_new
                    masks[i] = mask_new
                    fits[i] = fit_new
                    if fit_new > best_fit:
                        best_fit = fit_new
                        best_mask = mask_new.copy()
                        best_X = X_new.copy()

            # --- bundle spawning on top half ---
            if bundles:
                elite_idx = np.argsort(-fits)[:max(1, self.N // 2)]
                for ei in elite_idx:
                    bundle = bundles[self.rng.integers(0, len(bundles))]
                    missing = [j for j in bundle if not masks[ei][j]]
                    if not missing:
                        continue
                    n_tried += 1
                    trial = masks[ei].copy()
                    need = int(sum(costs[j] for j in missing))
                    used = int(costs[trial].sum())
                    # make room by dropping low-value units
                    if used + need > self.budget:
                        drops = _select_drop_candidates(
                            trial, costs, self.fitness.prize, n=len(missing) + 2
                        )
                        for d in drops:
                            if d not in bundle:
                                trial[d] = False
                                used -= int(costs[d])
                            if used + need <= self.budget:
                                break
                    for j in missing:
                        trial[j] = True
                    # re-check feasibility
                    if int(costs[trial].sum()) > self.budget:
                        continue
                    fit_trial = self.cache.evaluate(trial)
                    if fit_trial > fits[ei]:
                        n_accepted += 1
                        masks[ei] = trial
                        fits[ei] = fit_trial
                        X[ei] = np.where(trial, self.seed_high, self.seed_low).astype(float)
                        X[ei][~self.member] = 0.0
                        if fit_trial > best_fit:
                            best_fit = fit_trial
                            best_mask = trial.copy()
                            best_X = X[ei].copy()

            history.append({
                "iteration": t,
                "best_fitness": best_fit,
                "mean_fitness": float(np.mean(fits)),
                "bundles_tried": n_tried,
                "bundles_accepted": n_accepted,
            })

        return AROResult(
            mask=best_mask, fitness=best_fit, iterations=len(history),
            method="BundleARO", history=history,
            cache_hits=self.cache.hits, cache_misses=self.cache.misses,
            n_bundles_tried=n_tried, n_bundles_accepted=n_accepted,
        )


# ---------------------------------------------------------------------------
# 3. FastARO  —  BundleARO + direct set moves + incremental scoring
# ---------------------------------------------------------------------------

class FastARO(RepoARO):
    """Fastest ARO: direct set operations + bundles + incremental ΔF.

    Instead of continuous-to-discrete decoding every step, each rabbit directly
    proposes S' = (S \\ D) ∪ A where A is a bundle or single unit and D makes
    room.  The full surrogate judges acceptance.  Foraging borrows a bundle from
    the best rabbit's neighbourhood; hiding does a local add/drop/swap.

    This skips the continuous vector entirely after initialization.
    """

    def optimize(self) -> AROResult:
        costs = self.pool.token_costs
        bundles = _relation_bundles(self.pool, self.member)
        n_tried = 0
        n_accepted = 0

        # --- initialization: start from decoded retrieval seeds ---
        X = self._init_population()
        masks = self.decoder.decode_batch(X)
        fits = self.cache.evaluate_batch(masks)

        best_idx = int(np.argmax(fits))
        best_fit = float(fits[best_idx])
        best_mask = masks[best_idx].copy()

        history = []
        for t in range(1, self.Tmax + 1):
            for i in range(self.N):
                r = self.rng.random()
                if r < 0.5 and bundles:
                    # --- Foraging: borrow a bundle from best rabbit ---
                    bundle = bundles[self.rng.integers(0, len(bundles))]
                    proposal = self._try_insert_bundle(masks[i], bundle, costs)
                    n_tried += 1
                else:
                    # --- Hiding: local add/drop/swap ---
                    proposal = self._local_move(masks[i], costs)

                if proposal is None:
                    continue
                fit_new = self.cache.evaluate(proposal)
                if fit_new > fits[i]:
                    if r < 0.5:
                        n_accepted += 1
                    masks[i] = proposal
                    fits[i] = fit_new
                    if fit_new > best_fit:
                        best_fit = fit_new
                        best_mask = proposal.copy()

            history.append({
                "iteration": t,
                "best_fitness": best_fit,
                "mean_fitness": float(np.mean(fits)),
            })

        return AROResult(
            mask=best_mask, fitness=best_fit, iterations=len(history),
            method="FastARO", history=history,
            cache_hits=self.cache.hits, cache_misses=self.cache.misses,
            n_bundles_tried=n_tried, n_bundles_accepted=n_accepted,
        )

    def _try_insert_bundle(
        self, mask: np.ndarray, bundle: list[int], costs: np.ndarray
    ) -> np.ndarray | None:
        missing = [j for j in bundle if not mask[j]]
        if not missing:
            return None
        trial = mask.copy()
        need = int(sum(costs[j] for j in missing))
        used = int(costs[trial].sum())
        if used + need > self.budget:
            drops = _select_drop_candidates(trial, costs, self.fitness.prize, n=len(missing) + 2)
            for d in drops:
                if d not in bundle:
                    trial[d] = False
                    used -= int(costs[d])
                if used + need <= self.budget:
                    break
        for j in missing:
            trial[j] = True
        if int(costs[trial].sum()) > self.budget:
            return None
        return trial

    def _local_move(self, mask: np.ndarray, costs: np.ndarray) -> np.ndarray | None:
        sel = np.flatnonzero(mask & self.member)
        unsel = np.flatnonzero(~mask & self.member)
        if len(sel) == 0 or len(unsel) == 0:
            return None
        trial = mask.copy()
        move_type = self.rng.random()
        used = int(costs[trial].sum())
        if move_type < 0.33 and len(sel) > 1:
            # drop lowest-value
            ratios = self.fitness.prize[sel] / np.maximum(costs[sel], 1)
            j = sel[int(np.argmin(ratios))]
            trial[j] = False
        elif move_type < 0.66:
            # add best affordable
            affordable = unsel[costs[unsel] + used <= self.budget]
            if len(affordable) == 0:
                return None
            j = affordable[int(np.argmax(self.fitness.prize[affordable]))]
            trial[j] = True
        else:
            # swap: drop worst, add best affordable
            ratios = self.fitness.prize[sel] / np.maximum(costs[sel], 1)
            out_j = sel[int(np.argmin(ratios))]
            trial[out_j] = False
            freed = used - int(costs[out_j])
            affordable = unsel[costs[unsel] + freed <= self.budget]
            if len(affordable) == 0:
                trial[out_j] = True  # undo
                return None
            in_j = affordable[int(np.argmax(self.fitness.prize[affordable]))]
            trial[in_j] = True
        return trial
