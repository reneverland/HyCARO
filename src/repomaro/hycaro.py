"""HyCARO: hypergraph-catalyzed search over feasible context sets.

Proposals are bounded insertions and deletions applied to a feasible incumbent.
Acceptance uses the full surrogate. Support is the maximum role-compatible
witness, not the incident-relation-mass score used by RepoMARO.

Repository relations already stored on the hypergraph map onto the manuscript
types as: encloses→class contain, imports, calls→invoke, inherits, tests,
decorates, references. File-level ``contains`` is packaging and is not support.
``state`` edges do not record a reader and a writer, so they are not witnesses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np

from .backbone import build_anchor_balls
from .fitness import IssueCoverageFitness
from .schema import CandidatePool

# Directed repository relations whose (source, target) pairs are valid witnesses.
_ROLE_RELATIONS = frozenset({
    "encloses",
    "imports",
    "calls",
    "inherits",
    "tests",
    "decorates",
    "references",
})

_TAU0 = 0.5


@dataclass
class RelGroup:
    members: np.ndarray
    partners: dict[int, list[int]]


@dataclass
class WitnessIndex:
    pair_j: np.ndarray
    pair_k: np.ndarray
    groups: list[RelGroup]
    incident: list[list[int]]


@dataclass
class HyCAROResult:
    mask: np.ndarray
    fitness: float
    iterations: int
    method: str = "HyCARO"
    history: list[dict[str, Any]] = field(default_factory=list)
    cache_hits: int = 0
    cache_misses: int = 0
    n_accepted: int = 0
    n_proposals: int = 0
    search_seconds: float = 0.0
    fitness_init_seconds: float = 0.0
    ball_size: int = 0
    n_witness_pairs: int = 0
    n_groups: int = 0
    stopped: str = "generations"
    trajectory: list[dict[str, Any]] = field(default_factory=list)


def project_preferences(
    u: np.ndarray,
    mask: np.ndarray,
    member: np.ndarray,
    eps_x: float = 1e-3,
) -> np.ndarray:
    """Map a preference vector onto the side of the threshold that reproduces ``mask``."""
    u = np.clip(np.asarray(u, dtype=np.float64), 0.0, 1.0)
    out = np.zeros(u.shape[0], dtype=np.float64)
    selected = np.asarray(mask, dtype=bool) & np.asarray(member, dtype=bool)
    rejected = ~np.asarray(mask, dtype=bool) & np.asarray(member, dtype=bool)
    out[selected] = _TAU0 + eps_x + (1.0 - _TAU0 - eps_x) * u[selected]
    out[rejected] = (_TAU0 - eps_x) * u[rejected]
    return out


def build_witness_index(pool: CandidatePool, member: np.ndarray) -> WitnessIndex:
    """Role-compatible pairs and the hyperedges that nominate joint proposals."""
    m = len(pool.units)
    member = np.asarray(member, dtype=bool)
    uid_to_idx = {u.uid: i for i, u in enumerate(pool.units)}
    pair_j: list[int] = []
    pair_k: list[int] = []
    seen: set[tuple[int, int]] = set()
    groups: list[RelGroup] = []
    incident: list[list[int]] = [[] for _ in range(m)]
    graph = pool.graph
    if graph is None:
        return WitnessIndex(
            np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), groups, incident
        )
    for edge in graph.edges:
        if edge.relation not in _ROLE_RELATIONS or not edge.sources or not edge.targets:
            continue
        src = sorted({
            uid_to_idx[uid]
            for uid in edge.sources
            if uid in uid_to_idx and member[uid_to_idx[uid]]
        })
        tgt = sorted({
            uid_to_idx[uid]
            for uid in edge.targets
            if uid in uid_to_idx and member[uid_to_idx[uid]]
        })
        role_pairs: list[tuple[int, int]] = []
        partners: dict[int, list[int]] = {}
        for a in src:
            for b in tgt:
                if a == b:
                    continue
                role_pairs.append((a, b))
                partners.setdefault(a, []).append(b)
                partners.setdefault(b, []).append(a)
                key = (a, b) if a < b else (b, a)
                if key not in seen:
                    seen.add(key)
                    pair_j.append(key[0])
                    pair_k.append(key[1])
        if not role_pairs:
            continue
        partners = {i: sorted(set(vs)) for i, vs in partners.items()}
        gid = len(groups)
        groups.append(RelGroup(
            members=np.asarray(sorted(partners), dtype=np.int64),
            partners=partners,
        ))
        for i in partners:
            incident[i].append(gid)
    return WitnessIndex(
        np.asarray(pair_j, dtype=np.int64),
        np.asarray(pair_k, dtype=np.int64),
        groups,
        incident,
    )


class MaxWitnessFitness(IssueCoverageFitness):
    """Coverage and token price from the shared surrogate, with max-witness support.

    A selected unit is supported when at least one role-compatible partner is
    also selected. Extra witnesses do not increase the score, and two units that
    only share a role (two callees, two readers) do not support each other.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.pair_j = np.zeros(0, dtype=np.int64)
        self.pair_k = np.zeros(0, dtype=np.int64)
        super().__init__(*args, **kwargs)

    def _build_support(self, pool: CandidatePool, support_relations: Any):
        return np.zeros((self.m, 0), dtype=np.float64), np.zeros(self.m, dtype=np.float64)

    def attach_witnesses(self, index: WitnessIndex) -> None:
        self.pair_j = index.pair_j
        self.pair_k = index.pair_k

    def _support(self, masks: np.ndarray) -> np.ndarray:
        out = np.zeros(masks.shape[0], dtype=np.float64)
        if self.pair_j.size == 0 or self.z_need <= 0.0 or self.need_affinity.shape[0] == 0:
            return out
        for b in range(masks.shape[0]):
            sel = masks[b]
            both = sel[self.pair_j] & sel[self.pair_k]
            if not both.any():
                continue
            supported = np.zeros(self.m, dtype=bool)
            supported[self.pair_j[both]] = True
            supported[self.pair_k[both]] = True
            gated = np.where(sel & supported, self.need_affinity, 0.0)
            out[b] = self._aggregate_row(gated)
        return out

    def _aggregate_row(self, sub: np.ndarray) -> float:
        """Weighted top-p aggregation. ``sub`` is [n_needs, m] and already gated."""
        p = self.need_depth
        if p == 1:
            best = sub.max(axis=1, keepdims=True)
        else:
            if sub.shape[1] > p:
                top = -np.sort(np.partition(-sub, p - 1, axis=1)[:, :p], axis=1)
            else:
                top = -np.sort(-sub, axis=1)[:, :p]
            best = np.zeros((sub.shape[0], p), dtype=np.float64)
            best[:, : top.shape[1]] = top
        answered = (best * self.depth_w[:, : best.shape[1]]).sum(axis=1)
        return float((self.need_w * answered).sum() / self.z_need)


class _MaskCache:
    def __init__(self, fitness: MaxWitnessFitness) -> None:
        self.fitness = fitness
        self.store: dict[bytes, float] = {}
        self.hits = 0
        self.misses = 0

    def evaluate(self, mask: np.ndarray) -> float:
        key = np.ascontiguousarray(mask, dtype=np.uint8).tobytes()
        hit = self.store.get(key)
        if hit is not None:
            self.hits += 1
            return hit
        self.misses += 1
        value = float(self.fitness.evaluate(mask))
        self.store[key] = value
        return value


class HyCARO:
    """Bounded relation-guided context selection.

    Search settings default to the manuscript starting point (N=6, T=3, P=3).
    They are not taken from RepoMARO's population size. The feasible region is
    still the shared k-anchor radius-d ball.
    """

    def __init__(
        self,
        pool: CandidatePool,
        budget: int,
        fitness_cfg: dict[str, Any],
        optimizer_cfg: dict[str, Any],
        seed: int = 42,
    ) -> None:
        self.pool = pool
        self.budget = int(budget)
        self.rng = np.random.default_rng(seed)
        self.m = len(pool.units)
        self.costs = pool.token_costs.astype(np.int64)
        self.N = int(optimizer_cfg.get("hycaro_population", 6))
        self.Tmax = int(optimizer_cfg.get("hycaro_iterations", 3))
        self.P = int(optimizer_cfg.get("hycaro_proposals", 3))
        self.a_max = int(optimizer_cfg.get("hycaro_a_max", 3))
        self.d_max = int(optimizer_cfg.get("hycaro_d_max", 3))
        self.k_stop = int(optimizer_cfg.get("hycaro_stop", 2))
        self.pc = float(optimizer_cfg.get("hycaro_crossover", 0.8))
        self.lam_min = float(optimizer_cfg.get("hycaro_lambda_min", 0.2))
        self.lam_max = float(optimizer_cfg.get("hycaro_lambda_max", 0.8))
        self.eps_x = float(optimizer_cfg.get("hycaro_eps_x", 1e-3))
        self.eps_f = float(optimizer_cfg.get("hycaro_eps_f", 1e-9))
        raw_budget = optimizer_cfg.get("hycaro_time_budget_s")
        self.time_budget = None if raw_budget is None else float(raw_budget)
        self.ball_k = int(optimizer_cfg.get("ball_anchors", 5))
        self.ball_d = int(optimizer_cfg.get("ball_radius", 1))
        self.enable_seeding = bool(optimizer_cfg.get("enable_retrieval_seeding", True))
        # "relation": hyperedge-group bundles. "agnostic": same bundle sizes, but
        # every non-anchor unit is drawn by priority from the whole region.
        # "pairwise": only the witness pair, with partners pooled over all groups.
        self.proposal_mode = str(optimizer_cfg.get("hycaro_proposal_mode", "relation"))
        if self.proposal_mode not in {"relation", "agnostic", "pairwise"}:
            raise ValueError(f"unknown hycaro_proposal_mode {self.proposal_mode!r}")
        self.record_trajectory = bool(optimizer_cfg.get("hycaro_record_trajectory", False))
        self.search_mode = str(optimizer_cfg.get("hycaro_search", "population"))
        if self.search_mode not in {"population", "greedy"}:
            raise ValueError(f"unknown hycaro_search {self.search_mode!r}")
        self.fcfg = {**fitness_cfg, **(fitness_cfg.get("ball") or {})}
        if bool(optimizer_cfg.get("unrestricted_pool", False)) or self.m == 0:
            self.member = np.ones(self.m, dtype=bool)
        else:
            self.member = build_anchor_balls(pool, self.ball_k, self.ball_d)
        self.r = np.clip(pool.relevance.astype(np.float64), 0.0, 1.0) if self.m else np.zeros(0)
        self.lexical = pool.lexical_scores.astype(np.float64) if self.m else np.zeros(0)
        self.dense = pool.dense_scores.astype(np.float64) if self.m else np.zeros(0)

    def _expired(self, t0: float) -> bool:
        return self.time_budget is not None and (perf_counter() - t0) >= self.time_budget

    def _pack(self, scores: np.ndarray) -> np.ndarray:
        mask = np.zeros(self.m, dtype=bool)
        used = 0
        ranked = np.where(self.member, scores, -np.inf)
        for j in np.argsort(-ranked, kind="mergesort").tolist():
            if not self.member[j]:
                continue
            cost = int(self.costs[j])
            if cost <= 0 or used + cost > self.budget:
                continue
            mask[j] = True
            used += cost
        return mask

    def _proximity(self) -> np.ndarray:
        prox = np.full(self.m, -np.inf, dtype=np.float64)
        if self.m == 0 or not self.member.any():
            return prox
        k = max(self.ball_k, 1)
        anchors = np.argsort(-np.where(self.member, self.r, -np.inf), kind="mergesort")[:k]
        graph = self.pool.graph
        uid_to_idx = {u.uid: i for i, u in enumerate(self.pool.units)}
        for a in anchors.tolist():
            if not self.member[a]:
                continue
            prox[a] = max(prox[a], 1.0)
            uid = self.pool.units[a].uid
            if graph is None or self.ball_d <= 0 or uid not in graph:
                continue
            for other, dist in graph.hop_lengths(uid, cutoff=self.ball_d).items():
                j = uid_to_idx.get(other)
                if j is None or not self.member[j]:
                    continue
                prox[j] = max(prox[j], 1.0 / (1.0 + float(dist)))
        prox[~self.member] = -np.inf
        return prox

    def _priority(self, u: np.ndarray) -> np.ndarray:
        return (u + self.r) / 2.0

    def _sample_weighted(self, idxs: np.ndarray, weights: np.ndarray) -> int:
        w = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
        total = float(w.sum())
        if total <= 0.0:
            return int(idxs[int(self.rng.integers(0, len(idxs)))])
        return int(self.rng.choice(np.asarray(idxs, dtype=np.int64), p=w / total))

    def _ratio(self, j: int, priority: np.ndarray) -> float:
        return float(priority[j]) / max(float(self.costs[j]), 1e-9)

    def _fit_additions(
        self,
        mask: np.ndarray,
        additions: list[int],
        protected: set[int],
        priority: np.ndarray,
    ) -> np.ndarray | None:
        additions = [j for j in additions if self.member[j] and not mask[j]]
        if not additions or len(additions) > self.a_max:
            return None
        add_cost = int(self.costs[additions].sum())
        used = int(self.costs[mask].sum())
        trial = mask.copy()
        dropped = 0
        if used + add_cost > self.budget:
            victims = [j for j in np.flatnonzero(mask).tolist() if j not in protected]
            victims.sort(key=lambda j: (self._ratio(j, priority), j))
            for j in victims:
                if dropped >= self.d_max:
                    break
                trial[j] = False
                used -= int(self.costs[j])
                dropped += 1
                if used + add_cost <= self.budget:
                    break
        if used + add_cost > self.budget:
            return None
        trial[np.asarray(additions, dtype=np.int64)] = True
        outside = int((trial & ~self.member).sum())
        if int(self.costs[trial].sum()) > self.budget or outside:
            return None
        return trial

    def _bundle(
        self,
        mask: np.ndarray,
        priority: np.ndarray,
        anchor: int,
        gid: int,
        *,
        missing_partner: bool,
        fill: str = "random",
    ) -> np.ndarray | None:
        group = self.index.groups[gid]
        if self.proposal_mode == "pairwise":
            pooled = {
                j
                for g in self.index.incident[anchor]
                for j in self.index.groups[g].partners.get(anchor, [])
            }
        else:
            pooled = set(group.partners.get(anchor, []))
        partners = [j for j in sorted(pooled) if self.member[j] and (not missing_partner or not mask[j])]
        if not partners:
            return None
        partner = max(partners, key=lambda j: (self._ratio(j, priority), -j))
        witness = [j for j in (anchor, partner) if not mask[j]]
        if len(witness) > self.a_max:
            return None
        if self.proposal_mode == "pairwise":
            protected = {j for j in (anchor, partner) if mask[j]}
            return self._fit_additions(mask, witness, protected, priority)
        witness_set = set(witness)
        extras = [int(j) for j in group.members.tolist() if not mask[j] and int(j) not in witness_set]
        extras.sort(key=lambda j: (-self._ratio(j, priority), j))
        cap = min(self.a_max, len(witness) + len(extras))
        if len(witness) > cap:
            return None
        target = cap if fill == "max" else int(self.rng.integers(len(witness), cap + 1))
        additions = list(witness)
        for j in extras:
            if len(additions) >= target:
                break
            additions.append(j)
        if self.proposal_mode == "agnostic":
            keep = [anchor] if anchor in additions else []
            domain = np.asarray(
                [j for j in np.flatnonzero(~mask & self.member).tolist() if j != anchor],
                dtype=np.int64,
            )
            need = len(additions) - len(keep)
            if need > len(domain):
                return None
            drawn: list[int] = []
            if need:
                w = np.maximum(priority[domain], 0.0) + self.eps_x
                drawn = self.rng.choice(domain, size=need, replace=False, p=w / w.sum()).tolist()
            protected = {anchor} if mask[anchor] else set()
            return self._fit_additions(mask, keep + [int(j) for j in drawn], protected, priority)
        protected = {j for j in (anchor, partner) if mask[j]}
        return self._fit_additions(mask, additions, protected, priority)

    def _propose(self, mode: int, mask: np.ndarray, priority: np.ndarray) -> np.ndarray | None:
        if mode == 2:
            selected = np.flatnonzero(mask & self.member)
            if len(selected) == 0:
                return None
            j = min(selected.tolist(), key=lambda i: (self._ratio(i, priority), i))
            trial = mask.copy()
            trial[j] = False
            return trial
        if mode == 0:
            domain = np.flatnonzero(~mask & self.member)
            if len(domain) == 0:
                return None
            anchor = self._sample_weighted(domain, priority[domain] + self.eps_x)
            groups = [
                gid for gid in self.index.incident[anchor]
                if any(self.member[j] for j in self.index.groups[gid].partners.get(anchor, []))
            ]
            if not groups:
                return self._fit_additions(mask, [anchor], set(), priority)
            gid = int(self.rng.choice(np.asarray(groups, dtype=np.int64)))
            return self._bundle(mask, priority, anchor, gid, missing_partner=False)
        selected = np.flatnonzero(mask & self.member)
        if len(selected) == 0:
            return None
        anchor = self._sample_weighted(selected, priority[selected] + self.eps_x)
        groups = []
        for gid in self.index.incident[anchor]:
            if any(not mask[j] and self.member[j] for j in self.index.groups[gid].partners.get(anchor, [])):
                groups.append(gid)
        if not groups:
            return None
        gid = int(self.rng.choice(np.asarray(groups, dtype=np.int64)))
        return self._bundle(mask, priority, anchor, gid, missing_partner=True)

    def _greedy_actions(self, mask: np.ndarray, priority: np.ndarray) -> list[np.ndarray]:
        """Every relation-guided insertion or deletion from one incumbent."""
        seen: set[bytes] = set()
        actions: list[np.ndarray] = []

        def add(trial: np.ndarray | None) -> None:
            if trial is None or np.array_equal(trial, mask):
                return
            key = trial.tobytes()
            if key in seen:
                return
            seen.add(key)
            actions.append(trial)

        selected = np.flatnonzero(mask & self.member)
        if len(selected):
            victim = min(selected.tolist(), key=lambda i: (self._ratio(i, priority), i))
            deleted = mask.copy()
            deleted[victim] = False
            add(deleted)
        for anchor in np.flatnonzero(self.member).tolist():
            groups = list(self.index.incident[anchor])
            if not groups:
                if not mask[anchor]:
                    add(self._fit_additions(mask, [anchor], set(), priority))
                continue
            for gid in groups:
                add(self._bundle(
                    mask, priority, anchor, gid, missing_partner=False, fill="max",
                ))
                add(self._bundle(
                    mask, priority, anchor, gid, missing_partner=True, fill="max",
                ))
        return actions

    def _greedy_improve(
        self,
        best_mask: np.ndarray,
        best_fit: float,
        cache: _MaskCache,
        t0: float,
        init_s: float,
        trajectory: list[dict[str, Any]],
    ) -> HyCAROResult:
        """Single-incumbent greedy improvement under the same actions and budget."""
        incumbent = best_mask.copy()
        fit = float(best_fit)
        n_accepted = 0
        n_proposals = 0
        rounds = 0
        stopped = "no-improvement"
        history: list[dict[str, Any]] = []
        for _ in range(64):
            if self._expired(t0):
                stopped = "time"
                break
            priority = self._priority(
                project_preferences(self.r, incumbent, self.member, self.eps_x)
            )
            best_trial = None
            best_trial_fit = fit
            for trial in self._greedy_actions(incumbent, priority):
                if self._expired(t0):
                    stopped = "time"
                    break
                n_proposals += 1
                trial_fit = cache.evaluate(trial)
                if trial_fit > best_trial_fit + self.eps_f:
                    best_trial_fit = trial_fit
                    best_trial = trial
            if best_trial is None:
                break
            n_accepted += 1
            rounds += 1
            incumbent = best_trial
            fit = best_trial_fit
            history.append({
                "iteration": rounds,
                "best_fitness": float(fit),
                "accepted": n_accepted,
            })
            if self.record_trajectory:
                trajectory.append({
                    "iteration": rounds,
                    "best_fitness": float(fit),
                    "mean_fitness": float(fit),
                    "selected": np.flatnonzero(incumbent).astype(int).tolist(),
                })
        return HyCAROResult(
            mask=incumbent,
            fitness=float(fit),
            iterations=rounds,
            method="Relation-guided greedy",
            history=history,
            cache_hits=cache.hits,
            cache_misses=cache.misses,
            n_accepted=n_accepted,
            n_proposals=n_proposals,
            search_seconds=perf_counter() - t0,
            fitness_init_seconds=init_s,
            ball_size=int(self.member.sum()),
            n_witness_pairs=int(self.index.pair_j.size),
            n_groups=len(self.index.groups),
            stopped=stopped,
            trajectory=trajectory,
        )

    def _mutate(self, X: np.ndarray, i: int, x_best: np.ndarray, lam: float) -> np.ndarray:
        donors = [j for j in range(X.shape[0]) if j != i]
        r1, r2 = self.rng.choice(donors, size=2, replace=False)
        v = X[i] + lam * (x_best - X[i]) + lam * (X[int(r1)] - X[int(r2)])
        take = self.rng.random(self.m) < self.pc
        active = np.flatnonzero(self.member)
        if active.size and not np.any(take[active]):
            take[int(self.rng.choice(active))] = True
        u = np.where(take, v, X[i])
        np.clip(u, 0.0, 1.0, out=u)
        u[~self.member] = 0.0
        return u

    def _empty(self, stopped: str, init_s: float, search_s: float) -> HyCAROResult:
        mask = np.zeros(self.m, dtype=bool)
        return HyCAROResult(
            mask=mask,
            fitness=0.0,
            iterations=0,
            search_seconds=search_s,
            fitness_init_seconds=init_s,
            ball_size=int(self.member.sum()),
            stopped=stopped,
        )

    def optimize(self) -> HyCAROResult:
        if self.m == 0 or not self.member.any():
            return self._empty("empty-region", 0.0, 0.0)
        t_fit = perf_counter()
        fitness = MaxWitnessFitness(
            self.pool,
            budget=self.budget,
            alpha=float(self.fcfg.get("alpha", 1.0)),
            beta=float(self.fcfg.get("beta", 0.35)),
            gamma=float(self.fcfg.get("gamma", 0.0)),
            cost_lambda=float(self.fcfg.get("cost_lambda", 0.25)),
            semantic_need_share=float(self.fcfg.get("semantic_need_share", 0.5)),
            max_lexical_needs=int(self.fcfg.get("max_lexical_needs", 32)),
            max_need_df_ratio=float(self.fcfg.get("max_need_df_ratio", 0.5)),
            semantic_need_depth=int(self.fcfg.get("semantic_need_depth", 4)),
            lexical_need_depth=int(self.fcfg.get("lexical_need_depth", 1)),
            need_depth_decay=float(self.fcfg.get("need_depth_decay", 0.5)),
            redundancy_delta=float(self.fcfg.get("redundancy_delta", 0.3)),
            support_relations=self.fcfg.get("support_relations"),
        )
        init_s = perf_counter() - t_fit
        t0 = perf_counter()
        self.index = build_witness_index(self.pool, self.member)
        fitness.attach_witnesses(self.index)
        cache = _MaskCache(fitness)
        best_fit = 0.0
        best_mask = np.zeros(self.m, dtype=bool)
        x_best = np.zeros(self.m, dtype=np.float64)
        n_accepted = 0
        n_proposals = 0
        trajectory: list[dict[str, Any]] = []

        def snap(iteration: int, mean_fit: float) -> None:
            if not self.record_trajectory:
                return
            trajectory.append({
                "iteration": int(iteration),
                "best_fitness": float(best_fit),
                "mean_fitness": float(mean_fit),
                "selected": np.flatnonzero(best_mask).astype(int).tolist(),
            })

        specs: list[np.ndarray] = []
        if self.enable_seeding:
            specs.extend((self.lexical, self.dense, self._proximity()))
        while len(specs) < self.N:
            noise = self.rng.uniform(0.0, 0.1, size=self.m)
            specs.append(np.where(self.member, self.r + noise, -np.inf))
        X = np.zeros((self.N, self.m), dtype=np.float64)
        masks = np.zeros((self.N, self.m), dtype=bool)
        fits = np.zeros(self.N, dtype=np.float64)
        n = 0
        for scores in specs[: self.N]:
            if self._expired(t0):
                break
            mask = self._pack(scores)
            fit = cache.evaluate(mask)
            x = project_preferences(self.r, mask, self.member, self.eps_x)
            X[n] = x
            masks[n] = mask
            fits[n] = fit
            if fit > best_fit + self.eps_f:
                best_fit = fit
                best_mask = mask.copy()
                x_best = x.copy()
            n += 1
        snap(0, float(np.mean(fits[:n])) if n else 0.0)
        if self.search_mode == "greedy":
            return self._greedy_improve(
                best_mask, best_fit, cache, t0, init_s, trajectory,
            )
        if n < 3:
            return HyCAROResult(
                mask=best_mask,
                fitness=best_fit,
                iterations=0,
                history=[],
                cache_hits=cache.hits,
                cache_misses=cache.misses,
                n_accepted=0,
                n_proposals=0,
                search_seconds=perf_counter() - t0,
                fitness_init_seconds=init_s,
                ball_size=int(self.member.sum()),
                n_witness_pairs=int(self.index.pair_j.size),
                n_groups=len(self.index.groups),
                stopped="few-rabbits" if n else "time",
                trajectory=trajectory,
            )

        X = X[:n]
        masks = masks[:n]
        fits = fits[:n]
        history: list[dict[str, Any]] = []
        stagnant = 0
        generations = 0
        stopped = "generations"
        denom = max(1, self.Tmax - 1)
        for t in range(self.Tmax):
            if self._expired(t0):
                stopped = "time"
                break
            lam = self.lam_min + (self.lam_max - self.lam_min) * (1.0 - t / denom)
            snap_X = X.copy()
            snap_best = x_best.copy()
            previous = best_fit
            for i in range(n):
                u = self._mutate(snap_X, i, snap_best, lam)
                priority = self._priority(u)
                offset = int(self.rng.integers(0, 3))
                for attempt in range(self.P):
                    if self._expired(t0):
                        stopped = "time"
                        break
                    n_proposals += 1
                    trial = self._propose((attempt + offset) % 3, masks[i], priority)
                    if trial is None:
                        continue
                    fit_new = cache.evaluate(trial)
                    if fit_new <= fits[i] + self.eps_f:
                        continue
                    n_accepted += 1
                    masks[i] = trial
                    fits[i] = fit_new
                    X[i] = project_preferences(u, trial, self.member, self.eps_x)
                    if fit_new > best_fit + self.eps_f:
                        best_fit = fit_new
                        best_mask = trial.copy()
                        x_best = X[i].copy()
                if stopped == "time":
                    break
            generations = t + 1
            if best_fit > previous + self.eps_f:
                stagnant = 0
            else:
                stagnant += 1
            history.append({
                "iteration": generations,
                "best_fitness": float(best_fit),
                "mean_fitness": float(np.mean(fits)),
                "accepted": n_accepted,
            })
            snap(generations, float(np.mean(fits)))
            if stopped == "time" or stagnant >= self.k_stop:
                if stagnant >= self.k_stop and stopped != "time":
                    stopped = "stagnation"
                break

        return HyCAROResult(
            mask=best_mask,
            fitness=float(best_fit),
            iterations=generations,
            history=history,
            cache_hits=cache.hits,
            cache_misses=cache.misses,
            n_accepted=n_accepted,
            n_proposals=n_proposals,
            search_seconds=perf_counter() - t0,
            fitness_init_seconds=init_s,
            ball_size=int(self.member.sum()),
            n_witness_pairs=int(self.index.pair_j.size),
            n_groups=len(self.index.groups),
            stopped=stopped,
            trajectory=trajectory,
        )
