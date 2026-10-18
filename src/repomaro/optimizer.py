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
class RepoMAROResult:
    mask: np.ndarray
    fitness: float
    preference: np.ndarray
    iterations: int
    history: list[dict[str, Any]] = field(default_factory=list)
    cache_hits: int = 0
    cache_misses: int = 0
    local_search_triggers: int = 0


class RepoMARO:
    """Anchor-ball RepoMARO: search inside k anchor neighbourhoods of radius d.

    Population members are continuous preference vectors over the candidate
    units. A member is decoded into a token-feasible subset of the ball, scored
    by issue coverage plus structural support minus a token price, and refined
    by exact one-unit exchanges. The optimizer never invokes the downstream
    coding model.
    """

    def __init__(self, pool: CandidatePool, budget: int, fitness_cfg: dict[str, Any], optimizer_cfg: dict[str, Any], seed: int = 42):
        self.pool = pool
        self.budget = int(budget)
        self.cfg = optimizer_cfg
        self.rng = np.random.default_rng(seed)
        self.N = int(optimizer_cfg.get("population_size", 20))
        self.Tmax = int(optimizer_cfg.get("max_iterations", 40))
        self.threshold = float(optimizer_cfg.get("threshold", 0.5))
        self.lambda_min = float(optimizer_cfg.get("lambda_min", 0.15))
        self.lambda_max = float(optimizer_cfg.get("lambda_max", 0.80))
        self.cr = float(optimizer_cfg.get("crossover_rate", 0.70))
        self.elite_frac = float(optimizer_cfg.get("elite_fraction", 0.50))
        self.local_L = int(optimizer_cfg.get("local_proposals", 12))
        self.k_ls_max = int(optimizer_cfg.get("local_search_patience", 4))
        self.k_stop_max = int(optimizer_cfg.get("stopping_patience", 10))
        self.epsilon = float(optimizer_cfg.get("epsilon", 1e-4))
        self.seed_high = float(optimizer_cfg.get("seed_high", 0.90))
        self.seed_low = float(optimizer_cfg.get("seed_low", 0.10))
        self.seed_noise = float(optimizer_cfg.get("seed_noise", 0.03))
        self.enable_cache = bool(optimizer_cfg.get("enable_cache", True))
        self.enable_incremental = bool(optimizer_cfg.get("enable_incremental_local", True))
        self.enable_local = bool(optimizer_cfg.get("enable_local_search", True))
        self.enable_early_stop = bool(optimizer_cfg.get("enable_early_stopping", True))
        self.enable_seeding = bool(optimizer_cfg.get("enable_retrieval_seeding", True))
        self.enable_memory = bool(optimizer_cfg.get("enable_memory", True))
        self.enable_population_difference = bool(optimizer_cfg.get("enable_population_difference", True))
        self.enable_budget_repair = bool(optimizer_cfg.get("enable_budget_repair", True))

        if self.N < 4:
            raise ValueError("population_size must be at least 4 for differential search")
        if self.k_stop_max <= self.k_ls_max and self.enable_local:
            raise ValueError("stopping_patience should exceed local_search_patience so refinement can intervene")

        self.ball_k = int(optimizer_cfg.get("ball_anchors", 5))
        self.ball_d = int(optimizer_cfg.get("ball_radius", 1))
        self.local_screen = int(optimizer_cfg.get("local_screen_size", 64))
        # Node-RepoMARO: same search / objective, but the feasible set is the
        # whole pool rather than the k-anchor radius-d ball.
        if bool(optimizer_cfg.get("unrestricted_pool", False)):
            self.member = np.ones(len(pool.units), dtype=bool)
        else:
            self.member = build_anchor_balls(pool, self.ball_k, self.ball_d)

        # Coverage saturates and tokens carry an explicit price, so the objective
        # stops once the query is answered instead of packing the budget.
        fcfg = {**fitness_cfg, **(fitness_cfg.get("ball") or {})}
        self.fitness = IssueCoverageFitness(
            pool,
            budget=self.budget,
            alpha=float(fcfg.get("alpha", 1.0)),
            beta=float(fcfg.get("beta", 0.35)),
            gamma=float(fcfg.get("gamma", 0.0)),
            cost_lambda=float(fcfg.get("cost_lambda", 0.25)),
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
            self.member,
            pool.token_costs,
            self.budget,
            self.fitness,
            threshold=self.threshold,
            repair=self.enable_budget_repair,
        )
        self.cache = FitnessCache(self.fitness)

    def optimize(self) -> RepoMAROResult:
        X = self._initialize_population()
        current_masks = self.decoder.decode_batch(X)
        current_fit = self._evaluate_batch(current_masks)

        mem_X = X.copy()
        mem_masks = current_masks.copy()
        mem_fit = current_fit.copy()

        best_idx = int(np.argmax(mem_fit))
        best_fit = float(mem_fit[best_idx])
        best_X = mem_X[best_idx].copy()
        best_mask = mem_masks[best_idx].copy()

        k_ls = 0
        k_stop = 0
        history: list[dict[str, Any]] = []
        local_triggers = 0
        executed = 0

        for t in range(1, self.Tmax + 1):
            executed = t
            prev_best = best_fit
            lam = self._lambda_t(t)
            trials = self._batch_trials(X, best_X, lam)
            trial_masks = self.decoder.decode_batch(trials)
            trial_fit = self._evaluate_batch(trial_masks)

            accept = trial_fit > current_fit
            X[accept] = trials[accept]
            current_masks[accept] = trial_masks[accept]
            current_fit[accept] = trial_fit[accept]

            if self.enable_memory:
                mem_accept = trial_fit > mem_fit
                mem_X[mem_accept] = trials[mem_accept]
                mem_masks[mem_accept] = trial_masks[mem_accept]
                mem_fit[mem_accept] = trial_fit[mem_accept]
            else:
                mem_X[:] = X
                mem_masks[:] = current_masks
                mem_fit[:] = current_fit

            best_idx = int(np.argmax(mem_fit))
            best_fit = float(mem_fit[best_idx])
            best_X = mem_X[best_idx].copy()
            best_mask = mem_masks[best_idx].copy()

            if best_fit - prev_best > self.epsilon:
                k_ls = 0
                k_stop = 0
            else:
                k_ls += 1
                k_stop += 1

            local_triggered = False
            local_improved = False
            if self.enable_local and k_ls >= self.k_ls_max:
                local_triggered = True
                local_triggers += 1
                before_local = best_fit
                self._local_refine(mem_X, mem_masks, mem_fit)
                if not self.enable_memory:
                    X[:] = mem_X
                    current_masks[:] = mem_masks
                    current_fit[:] = mem_fit
                best_idx = int(np.argmax(mem_fit))
                best_fit = float(mem_fit[best_idx])
                best_X = mem_X[best_idx].copy()
                best_mask = mem_masks[best_idx].copy()
                local_improved = best_fit - before_local > self.epsilon
                k_ls = 0
                if local_improved:
                    k_stop = 0

            history.append(
                {
                    "iteration": t,
                    "lambda": lam,
                    "best_fitness": best_fit,
                    "mean_current_fitness": float(np.mean(current_fit)),
                    "local_search": local_triggered,
                    "local_improved": local_improved,
                    "cache_hits": self.cache.hits if self.enable_cache else 0,
                    "cache_misses": self.cache.misses if self.enable_cache else 0,
                    "unique_current_subsets": int(len({self.cache.key(m) for m in current_masks})),
                }
            )
            if self.enable_early_stop and k_stop >= self.k_stop_max:
                break

        return RepoMAROResult(
            mask=best_mask,
            fitness=best_fit,
            preference=best_X,
            iterations=executed,
            history=history,
            cache_hits=self.cache.hits if self.enable_cache else 0,
            cache_misses=self.cache.misses if self.enable_cache else 0,
            local_search_triggers=local_triggers,
        )

    def _initialize_population(self) -> np.ndarray:
        m = len(self.pool)
        X = self.rng.uniform(0.0, 1.0, size=(self.N, m))
        X[:, ~self.member] = 0.0
        if not self.enable_seeding:
            return X
        seeds = seed_masks(self.pool, self.budget)
        for row, name in enumerate(("bm25", "dense", "struct")):
            if row >= self.N:
                break
            vec = np.where(seeds[name] & self.member, self.seed_high, self.seed_low)
            vec = vec.astype(np.float64) + self.rng.normal(0.0, self.seed_noise, size=m)
            vec[~self.member] = 0.0
            X[row] = np.clip(vec, 0.0, 1.0)
        return X

    def _lambda_t(self, t: int) -> float:
        frac = 1.0 - float(t) / max(float(self.Tmax), 1.0)
        return self.lambda_min + (self.lambda_max - self.lambda_min) * frac

    def _batch_trials(self, X: np.ndarray, best_X: np.ndarray, lam: float) -> np.ndarray:
        N, m = X.shape
        r1 = np.empty(N, dtype=np.int64)
        r2 = np.empty(N, dtype=np.int64)
        all_idx = np.arange(N)
        for i in range(N):
            choices = all_idx[all_idx != i]
            pair = self.rng.choice(choices, size=2, replace=False)
            r1[i], r2[i] = int(pair[0]), int(pair[1])
        V = X + lam * (best_X[None, :] - X)
        if self.enable_population_difference:
            V = V + lam * (X[r1] - X[r2])
        V = np.clip(V, 0.0, 1.0)
        cross = self.rng.random((N, m)) < self.cr
        forced = self.rng.integers(0, m, size=N)
        cross[np.arange(N), forced] = True
        U = np.where(cross, V, X)
        return np.clip(U, 0.0, 1.0)

    def _evaluate_batch(self, masks: np.ndarray) -> np.ndarray:
        masks = np.asarray(masks, dtype=bool)
        feasible = np.asarray([self.decoder.feasible(m) for m in masks], dtype=bool)
        out = np.full(len(masks), -np.inf, dtype=np.float64)
        if not feasible.any():
            return out
        if self.enable_cache:
            out[feasible] = self.cache.evaluate_batch(masks[feasible])
        else:
            out[feasible] = np.asarray(self.fitness.evaluate_batch(masks[feasible]), dtype=np.float64)
        return out

    def _local_refine(self, mem_X: np.ndarray, mem_masks: np.ndarray, mem_fit: np.ndarray) -> None:
        n_elite = max(1, int(np.ceil(self.elite_frac * self.N)))
        elites = np.argsort(-mem_fit, kind="mergesort")[:n_elite]
        for e in elites.tolist():
            base_mask = mem_masks[e].copy()
            base_fit = float(mem_fit[e])
            proposals = self._generate_branch_proposals(base_mask, self.local_L)
            if not proposals:
                continue
            if self.enable_incremental:
                state = self.fitness.build_local_state(base_mask)
                scored = []
                for kind, nodes, mask in proposals:
                    if kind == "prune":
                        score = state.score_remove(nodes)
                    elif not np.any(base_mask & ~mask):
                        score = state.score_add(np.flatnonzero(mask & ~base_mask))
                    else:
                        # A swap both prunes a branch and attaches a node, so the
                        # incremental add-only score would not describe this mask.
                        score = float(self.fitness.evaluate(mask))
                    scored.append((float(score), mask))
                score, best_mask = max(scored, key=lambda item: item[0])
            else:
                masks = np.stack([p[2] for p in proposals], axis=0)
                vals = self._evaluate_batch(masks)
                j = int(np.argmax(vals))
                score, best_mask = float(vals[j]), masks[j]
            if score > base_fit + 1e-15:
                mem_fit[e] = score
                mem_masks[e] = best_mask
                mem_X[e] = self._vector_from_mask(best_mask, mem_X[e])
                if self.enable_cache:
                    self.cache.cache[self.cache.key(best_mask)] = score

    def _vector_from_mask(self, mask: np.ndarray, previous: np.ndarray) -> np.ndarray:
        target = np.where(mask, self.seed_high, self.seed_low)
        vec = np.clip(0.25 * previous + 0.75 * target, 0.0, 1.0)
        vec[~self.member] = 0.0
        return vec

    def _single_move_deltas(self, mask: np.ndarray, moves: np.ndarray, add: bool) -> np.ndarray:
        """Exact fitness of every one-unit variation of `mask`, in one batch."""
        if len(moves) == 0:
            return np.zeros(0, dtype=np.float64)
        trials = np.repeat(mask[None, :], len(moves), axis=0)
        trials[np.arange(len(moves)), moves] = add
        return np.asarray(self.fitness.evaluate_batch(trials), dtype=np.float64)

    def _generate_branch_proposals(self, mask: np.ndarray, L: int) -> list[tuple[str, np.ndarray, np.ndarray]]:
        """Rank prune/reconnect/swap moves by their exact marginal fitness per token.

        Ordering the proposals by retrieval relevance would answer a different
        question than the objective does: under Cov/Supp a unit is worth buying
        when it covers a need nothing else covers or completes the relations of
        a unit the issue is about, which relevance alone does not see. Both the
        drop and add rankings are therefore read off real one-unit deltas, taken
        in two batched evaluations per elite.
        """
        selected = np.flatnonzero(mask)
        if len(selected) == 0:
            return []
        costs = self.pool.token_costs
        used = int(costs[selected].sum())
        base = float(self.fitness.evaluate(mask))
        proposals: list[tuple[str, np.ndarray, np.ndarray]] = []
        target = max(1, L // 3)

        # Cheapest loss per released token first: these are the units the
        # objective is least attached to, and the ones worth freeing for a swap.
        drop_fit = self._single_move_deltas(mask, selected, add=False)
        drop_rank = np.argsort((base - drop_fit) / costs[selected], kind="mergesort")
        drops = selected[drop_rank].tolist()

        # Screening keeps the exact ranking affordable when the ball is large.
        outside = np.flatnonzero(self.member & ~mask)
        if len(outside) > self.local_screen:
            cheap = -(self.pool.relevance[outside] / costs[outside])
            outside = outside[np.argsort(cheap, kind="mergesort")[: self.local_screen]]
        add_fit = self._single_move_deltas(mask, outside, add=True)
        add_rank = np.argsort(-(add_fit - base) / costs[outside], kind="mergesort")
        adds = outside[add_rank].tolist()

        for j in drops[:target]:
            trial = mask.copy()
            trial[j] = False
            proposals.append(("prune", np.asarray([j], dtype=np.int64), trial))

        for j in adds:
            if len(proposals) >= 2 * target:
                break
            if used + int(costs[j]) > self.budget:
                continue
            trial = mask.copy()
            trial[j] = True
            proposals.append(("reconnect", np.asarray([j], dtype=np.int64), trial))

        for drop in [p for p in proposals if p[0] == "prune"]:
            freed = int(costs[drop[1]].sum())
            dropped = set(drop[1].tolist())
            for j in adds:
                if len(proposals) >= L:
                    break
                if mask[j] or j in dropped:
                    continue
                if int(costs[j]) > self.budget - (used - freed):
                    continue
                trial = mask.copy()
                trial[drop[1]] = False
                trial[j] = True
                proposals.append(("swap", drop[1], trial))
            if len(proposals) >= L:
                break
        return proposals[:L]
