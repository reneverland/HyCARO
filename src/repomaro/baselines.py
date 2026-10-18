from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .decoder import BudgetAwareDecoder
from .fitness import FitnessCache, SurrogateFitness
from .retrieval import greedy_pack_by_ranking, reciprocal_rank_fusion
from .schema import CandidatePool


@dataclass
class BaselineResult:
    name: str
    mask: np.ndarray
    fitness: float
    metadata: dict[str, Any]


def bm25_baseline(pool: CandidatePool, budget: int, fitness: SurrogateFitness) -> BaselineResult:
    mask = greedy_pack_by_ranking(pool.lexical_scores, pool.token_costs, budget)
    return BaselineResult("BM25", mask, float(fitness.evaluate(mask)), {})


def dense_baseline(pool: CandidatePool, budget: int, fitness: SurrogateFitness) -> BaselineResult:
    mask = greedy_pack_by_ranking(pool.dense_scores, pool.token_costs, budget)
    return BaselineResult("Dense", mask, float(fitness.evaluate(mask)), {})


def rrf_baseline(pool: CandidatePool, budget: int, fitness: SurrogateFitness) -> BaselineResult:
    scores = reciprocal_rank_fusion([pool.dense_scores, pool.structural_scores])
    mask = greedy_pack_by_ranking(scores, pool.token_costs, budget)
    return BaselineResult("RRF", mask, float(fitness.evaluate(mask)), {})


def mmr_baseline(pool: CandidatePool, budget: int, fitness: SurrogateFitness, lambda_rel: float = 0.7) -> BaselineResult:
    selected: list[int] = []
    remaining = set(range(len(pool.units)))
    used = 0
    relevance = pool.relevance
    sim = fitness.sim
    while remaining:
        best = None
        best_score = -np.inf
        for j in remaining:
            cost = int(pool.token_costs[j])
            if used + cost > budget:
                continue
            redundancy = max((sim[j, k] for k in selected), default=0.0)
            score = lambda_rel * relevance[j] - (1 - lambda_rel) * redundancy
            if score > best_score:
                best_score = float(score)
                best = j
        if best is None:
            break
        selected.append(best)
        remaining.remove(best)
        used += int(pool.token_costs[best])
    mask = np.zeros(len(pool.units), dtype=bool)
    mask[selected] = True
    return BaselineResult("MMR", mask, float(fitness.evaluate(mask)), {})


def greedy_utility(pool: CandidatePool, budget: int, fitness: SurrogateFitness) -> BaselineResult:
    selected = np.zeros(len(pool.units), dtype=bool)
    used = 0
    current = float(fitness.evaluate(selected))
    while True:
        best_j = None
        best_ratio = 0.0
        best_fit = current
        for j in np.flatnonzero(~selected).tolist():
            c = int(pool.token_costs[j])
            if used + c > budget:
                continue
            trial = selected.copy(); trial[j] = True
            f = float(fitness.evaluate(trial))
            gain = f - current
            ratio = gain / c
            if ratio > best_ratio + 1e-15:
                best_ratio = ratio
                best_j = j
                best_fit = f
        if best_j is None:
            break
        selected[best_j] = True
        used += int(pool.token_costs[best_j])
        current = best_fit
    return BaselineResult("Greedy-Utility", selected, current, {})


class BinaryPSO:
    def __init__(self, pool: CandidatePool, budget: int, fitness: SurrogateFitness, population_size: int = 20, iterations: int = 40, seed: int = 42):
        self.pool = pool
        self.budget = budget
        self.fit = fitness
        self.N = population_size
        self.T = iterations
        self.rng = np.random.default_rng(seed)
        self.decoder = BudgetAwareDecoder(pool.token_costs, budget, 0.5)
        self.cache = FitnessCache(fitness)

    def run(self) -> BaselineResult:
        m = len(self.pool.units)
        pos = self.rng.uniform(0, 1, size=(self.N, m))
        vel = self.rng.normal(0, 0.5, size=(self.N, m))
        masks = self.decoder.decode_batch(pos)
        scores = self.cache.evaluate_batch(masks)
        pbest = pos.copy(); pbest_scores = scores.copy(); pbest_masks = masks.copy()
        g = int(np.argmax(scores)); gbest = pos[g].copy(); gscore = float(scores[g]); gmask = masks[g].copy()
        for _ in range(self.T):
            r1 = self.rng.random((self.N, m)); r2 = self.rng.random((self.N, m))
            vel = 0.72 * vel + 1.49 * r1 * (pbest - pos) + 1.49 * r2 * (gbest[None, :] - pos)
            prob = 1.0 / (1.0 + np.exp(-vel))
            bits = self.rng.random((self.N, m)) < prob
            # Map binary proposal into continuous preferences then use common budget repair.
            pos = np.where(bits, 0.9, 0.1).astype(float)
            masks = self.decoder.decode_batch(pos)
            scores = self.cache.evaluate_batch(masks)
            better = scores > pbest_scores
            pbest[better] = pos[better]; pbest_scores[better] = scores[better]; pbest_masks[better] = masks[better]
            g = int(np.argmax(pbest_scores))
            if float(pbest_scores[g]) > gscore:
                gscore = float(pbest_scores[g]); gbest = pbest[g].copy(); gmask = pbest_masks[g].copy()
        return BaselineResult("Binary-PSO", gmask, gscore, {"cache_hits": self.cache.hits, "cache_misses": self.cache.misses})


class GenericMARO:
    """Generic population-search control without retrieval seeding or repository-aware local search.

    This is deliberately a control, not a claim to reproduce an unpublished implementation bit-for-bit.
    It keeps memory + adaptive differential search and uses the same hard-budget decoder for a fair,
    executable comparison. Local exploitation uses random coordinate swaps rather than repository edges.
    """

    def __init__(self, pool: CandidatePool, budget: int, fitness: SurrogateFitness, population_size: int = 20, iterations: int = 40, seed: int = 42):
        self.pool = pool; self.budget = budget; self.fit = fitness; self.N = population_size; self.T = iterations
        self.rng = np.random.default_rng(seed)
        self.decoder = BudgetAwareDecoder(pool.token_costs, budget, 0.5)
        self.cache = FitnessCache(fitness)

    def run(self) -> BaselineResult:
        m = len(self.pool.units)
        X = self.rng.uniform(0, 1, size=(self.N, m))
        masks = self.decoder.decode_batch(X); scores = self.cache.evaluate_batch(masks)
        memX = X.copy(); memM = masks.copy(); memF = scores.copy()
        for t in range(1, self.T + 1):
            b = int(np.argmax(memF)); best = memX[b]
            lam = 0.15 + (0.80 - 0.15) * (1 - t / self.T)
            U = X.copy()
            for i in range(self.N):
                others = np.array([j for j in range(self.N) if j != i])
                r1, r2 = self.rng.choice(others, 2, replace=False)
                v = X[i] + lam * (best - X[i]) + lam * (X[r1] - X[r2])
                v = np.clip(v, 0, 1)
                cross = self.rng.random(m) < 0.7; cross[self.rng.integers(0, m)] = True
                U[i] = np.where(cross, v, X[i])
            newM = self.decoder.decode_batch(U); newF = self.cache.evaluate_batch(newM)
            better = newF > scores
            X[better] = U[better]; masks[better] = newM[better]; scores[better] = newF[better]
            mb = newF > memF
            memX[mb] = U[mb]; memM[mb] = newM[mb]; memF[mb] = newF[mb]

            # Generic local random swap on top half memories.
            for e in np.argsort(-memF)[: max(1, self.N // 2)].tolist():
                base = memM[e].copy(); selected = np.flatnonzero(base); unselected = np.flatnonzero(~base)
                if len(selected) == 0 or len(unselected) == 0:
                    continue
                out_j = int(self.rng.choice(selected)); in_j = int(self.rng.choice(unselected))
                prop = base.copy(); prop[out_j] = False; prop[in_j] = True
                # Decoder-style repair through a preference representation.
                pref = np.where(prop, 0.9, 0.1)
                prop = self.decoder.decode(pref)
                f = self.cache.evaluate(prop)
                if f > memF[e]:
                    memF[e] = f; memM[e] = prop; memX[e] = pref
        b = int(np.argmax(memF))
        return BaselineResult("Generic-MARO", memM[b], float(memF[b]), {"cache_hits": self.cache.hits, "cache_misses": self.cache.misses})
