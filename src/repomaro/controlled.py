"""Main-table controls that share HyCARO's admissible ball and, for search, its fitness.

Retrieval controls pack inside the ball. Greedy+1-swap, two-unit greedy, and
DE+full repair optimize the same max-witness surrogate and stop at the selection
deadline, keeping the best fully evaluated feasible set.
"""
from __future__ import annotations

from time import perf_counter
from typing import Any, Callable

import numpy as np

from .backbone import build_anchor_balls
from .fitness import SurrogateFitness
from .hycaro import MaxWitnessFitness, build_witness_index, project_preferences
from .retrieval import reciprocal_rank_fusion
from .schema import CandidatePool

_REPOGRAPH_RELATIONS = frozenset({"calls", "references", "encloses", "inherits"})
_NAMES = {
    "bm25": "BM25",
    "dense": "Dense",
    "rrf": "RRF",
    "mmr": "MMR",
    "repograph": "RepoGraph",
    "greedy_1swap": "Greedy+1swap",
    "greedy_f": "Greedy-F",
    "two_unit": "Two-unit",
    "de_repair": "DE+repair",
}


def admissible_member(pool: CandidatePool, optimizer_cfg: dict[str, Any]) -> np.ndarray:
    if bool(optimizer_cfg.get("unrestricted_pool", False)) or len(pool.units) == 0:
        return np.ones(len(pool.units), dtype=bool)
    return build_anchor_balls(
        pool,
        int(optimizer_cfg.get("ball_anchors", 5)),
        int(optimizer_cfg.get("ball_radius", 1)),
    )


def pack_in_ball(
    scores: np.ndarray,
    costs: np.ndarray,
    budget: int,
    member: np.ndarray,
) -> np.ndarray:
    mask = np.zeros(len(scores), dtype=bool)
    used = 0
    ranked = np.where(np.asarray(member, dtype=bool), scores, -np.inf)
    for j in np.argsort(-ranked, kind="mergesort").tolist():
        if not member[j]:
            continue
        cost = int(costs[j])
        if cost <= 0 or used + cost > budget:
            continue
        mask[j] = True
        used += cost
    return mask


def _merged_fitness_cfg(fitness_cfg: dict[str, Any]) -> dict[str, Any]:
    return {**fitness_cfg, **(fitness_cfg.get("ball") or {})}


def make_witness_fitness(
    pool: CandidatePool,
    budget: int,
    fitness_cfg: dict[str, Any],
    member: np.ndarray,
) -> MaxWitnessFitness:
    cfg = _merged_fitness_cfg(fitness_cfg)
    fitness = MaxWitnessFitness(
        pool,
        budget=budget,
        alpha=float(cfg.get("alpha", 1.0)),
        beta=float(cfg.get("beta", 0.35)),
        gamma=float(cfg.get("gamma", 0.0)),
        cost_lambda=float(cfg.get("cost_lambda", 0.25)),
        semantic_need_share=float(cfg.get("semantic_need_share", 0.5)),
        max_lexical_needs=int(cfg.get("max_lexical_needs", 32)),
        max_need_df_ratio=float(cfg.get("max_need_df_ratio", 0.5)),
        semantic_need_depth=int(cfg.get("semantic_need_depth", 4)),
        lexical_need_depth=int(cfg.get("lexical_need_depth", 1)),
        need_depth_decay=float(cfg.get("need_depth_decay", 0.5)),
        redundancy_delta=float(cfg.get("redundancy_delta", 0.3)),
        support_relations=cfg.get("support_relations"),
    )
    fitness.attach_witnesses(build_witness_index(pool, member))
    return fitness


def _deadline(optimizer_cfg: dict[str, Any]) -> float | None:
    raw = optimizer_cfg.get("selection_deadline_s")
    if raw is None:
        return None
    return float(raw)


def _chunk_scores(fitness: MaxWitnessFitness, base: np.ndarray, idxs: np.ndarray, value: bool) -> np.ndarray:
    if len(idxs) == 0:
        return np.zeros(0, dtype=np.float64)
    trials = np.repeat(base[None, :], len(idxs), axis=0)
    trials[np.arange(len(idxs)), idxs] = value
    return np.asarray(fitness.evaluate_batch(trials), dtype=np.float64)


def _best_positive(scores: np.ndarray, current: float, ratios: np.ndarray, eps: float) -> int | None:
    improve = scores > current + eps
    if not np.any(improve):
        return None
    ratios = np.where(improve, ratios, -np.inf)
    return int(np.argmax(ratios))


def mmr_in_ball(
    pool: CandidatePool,
    budget: int,
    member: np.ndarray,
    sim: np.ndarray,
    lambda_rel: float = 0.7,
) -> np.ndarray:
    selected: list[int] = []
    remaining = [int(j) for j in np.flatnonzero(member).tolist()]
    used = 0
    relevance = pool.relevance
    costs = pool.token_costs
    while remaining:
        best = None
        best_score = -np.inf
        for j in remaining:
            cost = int(costs[j])
            if cost <= 0 or used + cost > budget:
                continue
            redundancy = max((float(sim[j, k]) for k in selected), default=0.0)
            score = lambda_rel * float(relevance[j]) - (1.0 - lambda_rel) * redundancy
            if score > best_score:
                best_score = score
                best = j
        if best is None:
            break
        selected.append(best)
        remaining.remove(best)
        used += int(costs[best])
    mask = np.zeros(len(pool.units), dtype=bool)
    if selected:
        mask[selected] = True
    return mask


def _embedding_sim(pool: CandidatePool) -> np.ndarray:
    from .fitness import _normalize_rows

    embeddings = _normalize_rows(np.stack([u.embedding for u in pool.units], axis=0))
    sim = np.maximum(embeddings @ embeddings.T, 0.0)
    np.fill_diagonal(sim, 0.0)
    return sim


def repograph_mask(pool: CandidatePool, budget: int, member: np.ndarray, k: int) -> np.ndarray:
    m = len(pool.units)
    chosen = np.zeros(m, dtype=bool)
    if m == 0 or not member.any():
        return chosen
    relevance = pool.relevance
    anchors = np.argsort(-np.where(member, relevance, -np.inf), kind="mergesort")[: max(int(k), 1)]
    anchor_set = {int(j) for j in anchors.tolist() if member[j]}
    chosen[list(anchor_set)] = True
    uid_to_idx = {u.uid: i for i, u in enumerate(pool.units)}
    graph = pool.graph
    if graph is not None and anchor_set:
        anchor_uids = {pool.units[j].uid for j in anchor_set}
        for edge in graph.edges:
            if edge.relation not in _REPOGRAPH_RELATIONS:
                continue
            idxs = [uid_to_idx[uid] for uid in edge.members if uid in uid_to_idx]
            if not any(pool.units[j].uid in anchor_uids for j in idxs):
                continue
            for j in idxs:
                if member[j]:
                    chosen[j] = True
    return pack_in_ball(relevance, pool.token_costs, budget, chosen & member)


def _fit_additions(
    mask: np.ndarray,
    additions: list[int],
    protected: set[int],
    priority: np.ndarray,
    costs: np.ndarray,
    budget: int,
    d_max: int,
) -> np.ndarray | None:
    additions = [j for j in additions if not mask[j]]
    if not additions:
        return None
    add_cost = int(costs[additions].sum())
    used = int(costs[mask].sum())
    trial = mask.copy()
    dropped = 0
    if used + add_cost > budget:
        victims = [int(j) for j in np.flatnonzero(mask).tolist() if j not in protected]
        victims.sort(key=lambda j: (float(priority[j]) / max(float(costs[j]), 1e-9), j))
        for j in victims:
            if dropped >= d_max:
                break
            trial[j] = False
            used -= int(costs[j])
            dropped += 1
            if used + add_cost <= budget:
                break
    if used + add_cost > budget:
        return None
    trial[np.asarray(additions, dtype=np.int64)] = True
    if int(costs[trial].sum()) > budget:
        return None
    return trial


def best_shared_seed(
    pool: CandidatePool,
    fitness: MaxWitnessFitness,
    budget: int,
    member: np.ndarray,
    optimizer_cfg: dict[str, Any],
    seed: int,
    expired: Callable[[], bool],
) -> tuple[np.ndarray, float, int]:
    """Rebuild and score HyCARO's initial contexts in HyCARO's order.

    The noise draws use a fresh generator with the run seed, so the masks equal
    the ones HyCARO packs before its first generation.
    """
    m = len(pool.units)
    rng = np.random.default_rng(seed)
    n_pop = int(optimizer_cfg.get("hycaro_population", 6))
    k = int(optimizer_cfg.get("ball_anchors", 5))
    d = int(optimizer_cfg.get("ball_radius", 1))
    relevance = np.clip(pool.relevance.astype(np.float64), 0.0, 1.0)
    specs = [pool.lexical_scores.astype(np.float64), pool.dense_scores.astype(np.float64)]
    specs.append(_proximity(pool, member, k, d))
    while len(specs) < n_pop:
        noise = rng.uniform(0.0, 0.1, size=m)
        specs.append(np.where(member, relevance + noise, -np.inf))
    best_mask = np.zeros(m, dtype=bool)
    best_fit = 0.0
    evals = 0
    for scores in specs[:n_pop]:
        if expired():
            break
        mask = pack_in_ball(scores, pool.token_costs, budget, member)
        fit = float(fitness.evaluate(mask))
        evals += 1
        if fit > best_fit + 1e-9:
            best_fit, best_mask = fit, mask
    return best_mask, best_fit, evals


def greedy_1swap(
    fitness: MaxWitnessFitness,
    costs: np.ndarray,
    budget: int,
    member: np.ndarray,
    deadline_s: float | None,
    eps: float = 1e-9,
    init: Callable[[Callable[[], bool]], tuple[np.ndarray, float, int]] | None = None,
    additions_only: bool = False,
) -> tuple[np.ndarray, float, str, int]:
    """``additions_only`` stops after the insertion phase, which is Greedy-F."""
    t0 = perf_counter()

    def expired() -> bool:
        return deadline_s is not None and (perf_counter() - t0) >= deadline_s

    if init is None:
        mask = np.zeros(len(costs), dtype=bool)
        current = float(fitness.evaluate(mask))
        evals = 1
    else:
        mask, current, evals = init(expired)

    def consider(idxs: np.ndarray, value: bool, ratio_of) -> tuple[np.ndarray | None, float]:
        nonlocal evals
        best_mask = None
        best_fit = current
        best_ratio = 0.0
        for start in range(0, len(idxs), 128):
            if expired():
                break
            chunk = idxs[start:start + 128]
            scores = _chunk_scores(fitness, mask, chunk, value)
            evals += len(chunk)
            ratios = ratio_of(chunk, scores)
            local = _best_positive(scores, current, ratios, eps)
            if local is None:
                continue
            if float(ratios[local]) > best_ratio:
                best_ratio = float(ratios[local])
                best_fit = float(scores[local])
                trial = mask.copy()
                trial[int(chunk[local])] = value
                best_mask = trial
        return best_mask, best_fit

    stopped = "optimum"
    while not expired():
        cand = np.flatnonzero(member & ~mask & (costs <= budget - int(costs[mask].sum())))
        if len(cand) == 0:
            break
        trial, fit = consider(
            cand,
            True,
            lambda chunk, scores: (scores - current) / np.maximum(costs[chunk], 1),
        )
        if trial is None:
            if expired():
                stopped = "time"
            break
        mask = trial
        current = fit
    else:
        stopped = "time"
    if additions_only:
        return mask, current, stopped, evals

    while not expired():
        improved = False
        sel = np.flatnonzero(mask)
        if len(sel):
            trial, fit = consider(sel, False, lambda chunk, scores: scores - current)
            if trial is not None and fit > current + eps:
                mask, current, improved = trial, fit, True
        if expired():
            stopped = "time"
            break
        sel = np.flatnonzero(mask)
        used = int(costs[mask].sum())
        cand = np.flatnonzero(member & ~mask & (costs <= budget - used))
        if len(cand):
            trial, fit = consider(
                cand,
                True,
                lambda chunk, scores: (scores - current) / np.maximum(costs[chunk], 1),
            )
            if trial is not None and fit > current + eps:
                mask, current, improved = trial, fit, True
        if expired():
            stopped = "time"
            break
        # Single exchanges. Each completed trial is eligible; an unfinished sweep
        # still keeps the best fully evaluated one.
        sel = np.flatnonzero(mask)
        used = int(costs[mask].sum())
        best_mask = None
        best_fit = current
        affordable = np.flatnonzero(member & ~mask)
        for out_j in sel.tolist():
            if expired():
                stopped = "time"
                break
            freed = used - int(costs[out_j])
            partners = affordable[costs[affordable] <= budget - freed]
            if len(partners) == 0:
                continue
            base = mask.copy()
            base[out_j] = False
            for start in range(0, len(partners), 128):
                if expired():
                    stopped = "time"
                    break
                chunk = partners[start:start + 128]
                scores = _chunk_scores(fitness, base, chunk, True)
                evals += len(chunk)
                better = np.flatnonzero(scores > best_fit + eps)
                if len(better) == 0:
                    continue
                local = better[int(np.argmax(scores[better]))]
                best_fit = float(scores[local])
                trial = base.copy()
                trial[int(chunk[local])] = True
                best_mask = trial
        if best_mask is not None:
            mask, current, improved = best_mask, best_fit, True
        if not improved:
            break
    return mask, current, stopped, evals


def two_unit_greedy(
    fitness: MaxWitnessFitness,
    costs: np.ndarray,
    budget: int,
    member: np.ndarray,
    relevance: np.ndarray,
    deadline_s: float | None,
    d_max: int = 3,
    eps: float = 1e-9,
    init: Callable[[Callable[[], bool]], tuple[np.ndarray, float, int]] | None = None,
) -> tuple[np.ndarray, float, str, int]:
    t0 = perf_counter()

    def expired() -> bool:
        return deadline_s is not None and (perf_counter() - t0) >= deadline_s

    if init is None:
        mask = np.zeros(len(costs), dtype=bool)
        current = float(fitness.evaluate(mask))
        evals = 1
    else:
        mask, current, evals = init(expired)
    idxs = np.flatnonzero(member)
    pairs: list[tuple[float, int, int]] = []
    for a_i, i in enumerate(idxs.tolist()):
        for j in idxs[a_i + 1:].tolist():
            denom = max(int(costs[i]) + int(costs[j]), 1)
            pairs.append(((float(relevance[i]) + float(relevance[j])) / denom, int(i), int(j)))
    pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
    stopped = "optimum"

    while not expired():
        best_mask = None
        best_fit = current
        for _, i, j in pairs:
            if expired():
                stopped = "time"
                break
            missing = [u for u in (i, j) if not mask[u]]
            if not missing:
                continue
            protected = {u for u in (i, j) if mask[u]}
            trial = _fit_additions(mask, missing, protected, relevance, costs, budget, d_max)
            if trial is None:
                continue
            fit = float(fitness.evaluate(trial))
            evals += 1
            if fit > best_fit + eps:
                best_fit = fit
                best_mask = trial
        if best_mask is None:
            break
        mask = best_mask
        current = best_fit
    return mask, current, stopped, evals


def _proximity(pool: CandidatePool, member: np.ndarray, k: int, d: int) -> np.ndarray:
    m = len(pool.units)
    prox = np.full(m, -np.inf, dtype=np.float64)
    if m == 0 or not member.any():
        return prox
    relevance = np.clip(pool.relevance, 0.0, 1.0)
    anchors = np.argsort(-np.where(member, relevance, -np.inf), kind="mergesort")[: max(k, 1)]
    graph = pool.graph
    uid_to_idx = {u.uid: i for i, u in enumerate(pool.units)}
    for a in anchors.tolist():
        if not member[a]:
            continue
        prox[a] = max(prox[a], 1.0)
        uid = pool.units[a].uid
        if graph is None or d <= 0 or uid not in graph:
            continue
        for other, dist in graph.hop_lengths(uid, cutoff=d).items():
            j = uid_to_idx.get(other)
            if j is None or not member[j]:
                continue
            prox[j] = max(prox[j], 1.0 / (1.0 + float(dist)))
    prox[~member] = -np.inf
    return prox


def _full_repair(
    fitness: MaxWitnessFitness,
    costs: np.ndarray,
    budget: int,
    member: np.ndarray,
    selected: np.ndarray,
    expired: Callable[[], bool],
    eps: float,
    counter: list[int],
) -> np.ndarray | None:
    selected = np.asarray(selected, dtype=bool) & member
    while int(costs[selected].sum()) > budget:
        if expired():
            return None
        heads = np.flatnonzero(selected)
        if len(heads) == 0:
            break
        current = float(fitness.evaluate(selected))
        scores = _chunk_scores(fitness, selected, heads, False)
        counter[0] += 1 + len(heads)
        losses = current - scores
        j = int(heads[int(np.argmin(losses / np.maximum(costs[heads], 1)))])
        selected = selected.copy()
        selected[j] = False
    while True:
        if expired():
            return None
        heads = np.flatnonzero(selected)
        if len(heads) == 0:
            break
        current = float(fitness.evaluate(selected))
        scores = _chunk_scores(fitness, selected, heads, False)
        counter[0] += 1 + len(heads)
        gains = scores - current
        if float(gains.max()) <= eps:
            break
        selected = selected.copy()
        selected[int(heads[int(np.argmax(gains))])] = False
    while True:
        if expired():
            return None
        used = int(costs[selected].sum())
        cand = np.flatnonzero(member & ~selected & (costs <= budget - used))
        if len(cand) == 0:
            break
        current = float(fitness.evaluate(selected))
        scores = _chunk_scores(fitness, selected, cand, True)
        counter[0] += 1 + len(cand)
        ratios = (scores - current) / np.maximum(costs[cand], 1)
        pick = _best_positive(scores, current, ratios, eps)
        if pick is None:
            break
        selected = selected.copy()
        selected[int(cand[pick])] = True
    if int(costs[selected].sum()) > budget or np.any(selected & ~member):
        return None
    return selected


def de_full_repair(
    pool: CandidatePool,
    fitness: MaxWitnessFitness,
    budget: int,
    member: np.ndarray,
    optimizer_cfg: dict[str, Any],
    seed: int,
    deadline_s: float | None,
) -> tuple[np.ndarray, float, str, int, int]:
    t0 = perf_counter()
    m = len(pool.units)
    costs = pool.token_costs
    rng = np.random.default_rng(seed)
    n_pop = int(optimizer_cfg.get("de_population", optimizer_cfg.get("hycaro_population", 6)))
    t_max = int(optimizer_cfg.get("de_iterations", 40))
    pc = float(optimizer_cfg.get("hycaro_crossover", 0.8))
    lam_min = float(optimizer_cfg.get("hycaro_lambda_min", 0.2))
    lam_max = float(optimizer_cfg.get("hycaro_lambda_max", 0.8))
    eps = float(optimizer_cfg.get("hycaro_eps_f", 1e-9))
    eps_x = float(optimizer_cfg.get("hycaro_eps_x", 1e-3))
    k = int(optimizer_cfg.get("ball_anchors", 5))
    d = int(optimizer_cfg.get("ball_radius", 1))
    relevance = np.clip(pool.relevance, 0.0, 1.0)
    counter = [0]

    def expired() -> bool:
        return deadline_s is not None and (perf_counter() - t0) >= deadline_s

    specs = [pool.lexical_scores, pool.dense_scores, _proximity(pool, member, k, d)]
    while len(specs) < n_pop:
        noise = rng.uniform(0.0, 0.1, size=m)
        specs.append(np.where(member, relevance + noise, -np.inf))
    X = np.zeros((n_pop, m), dtype=np.float64)
    masks = np.zeros((n_pop, m), dtype=bool)
    fits = np.full(n_pop, -np.inf, dtype=np.float64)
    best_mask = np.zeros(m, dtype=bool)
    best_fit = 0.0
    n = 0
    for scores in specs[:n_pop]:
        if expired():
            break
        mask = pack_in_ball(scores, costs, budget, member)
        fit = float(fitness.evaluate(mask))
        counter[0] += 1
        x = project_preferences(relevance, mask, member, eps_x)
        X[n] = x
        masks[n] = mask
        fits[n] = fit
        if fit > best_fit + eps:
            best_fit = fit
            best_mask = mask.copy()
        n += 1
    if n < 3:
        return best_mask, best_fit, "time" if n == 0 else "few-rabbits", counter[0], 0

    X, masks, fits = X[:n], masks[:n], fits[:n]
    generations = 0
    stopped = "generations"
    denom = max(1, t_max - 1)
    for t in range(t_max):
        if expired():
            stopped = "time"
            break
        lam = lam_min + (lam_max - lam_min) * (1.0 - t / denom)
        snap_X = X.copy()
        snap_best = project_preferences(relevance, best_mask, member, eps_x)
        for i in range(n):
            if expired():
                stopped = "time"
                break
            donors = [j for j in range(n) if j != i]
            r1, r2 = rng.choice(donors, size=2, replace=False)
            v = snap_X[i] + lam * (snap_best - snap_X[i]) + lam * (snap_X[int(r1)] - snap_X[int(r2)])
            take = rng.random(m) < pc
            active = np.flatnonzero(member)
            if active.size and not np.any(take[active]):
                take[int(rng.choice(active))] = True
            u = np.where(take, v, snap_X[i])
            np.clip(u, 0.0, 1.0, out=u)
            u[~member] = 0.0
            proposed = (u >= 0.5) & member
            repaired = _full_repair(fitness, costs, budget, member, proposed, expired, eps, counter)
            if repaired is None:
                stopped = "time"
                break
            fit_new = float(fitness.evaluate(repaired))
            counter[0] += 1
            if fit_new <= fits[i] + eps:
                continue
            masks[i] = repaired
            fits[i] = fit_new
            X[i] = project_preferences(u, repaired, member, eps_x)
            if fit_new > best_fit + eps:
                best_fit = fit_new
                best_mask = repaired.copy()
        generations = t + 1
        if stopped == "time":
            break
    return best_mask, best_fit, stopped, counter[0], generations


def run_controlled(
    method: str,
    pool: CandidatePool,
    budget: int,
    fitness_cfg: dict[str, Any],
    optimizer_cfg: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    member = admissible_member(pool, optimizer_cfg)
    costs = pool.token_costs
    name = _NAMES[method]
    deadline = _deadline(optimizer_cfg)
    meta: dict[str, Any] = {
        "ball_size": int(member.sum()),
        "deadline_s": deadline,
        "restrict_to_ball": True,
    }
    if method in {"bm25", "dense", "rrf", "mmr", "repograph"}:
        if method == "bm25":
            mask = pack_in_ball(pool.lexical_scores, costs, budget, member)
        elif method == "dense":
            mask = pack_in_ball(pool.dense_scores, costs, budget, member)
        elif method == "rrf":
            fused = reciprocal_rank_fusion([pool.lexical_scores, pool.dense_scores])
            mask = pack_in_ball(fused, costs, budget, member)
        elif method == "mmr":
            sim = _embedding_sim(pool) if len(pool.units) else np.zeros((0, 0))
            mask = mmr_in_ball(pool, budget, member, sim)
        else:
            mask = repograph_mask(pool, budget, member, int(optimizer_cfg.get("ball_anchors", 5)))
        fitness = SurrogateFitness(
            pool,
            alpha=float(fitness_cfg.get("alpha", 0.45)),
            beta=float(fitness_cfg.get("beta", 0.35)),
            gamma=float(fitness_cfg.get("gamma", 0.20)),
            sigma=float(fitness_cfg.get("sigma", 2.0)),
        ) if len(pool.units) else None
        score = float(fitness.evaluate(mask)) if fitness is not None else 0.0
        return {"method": name, "mask": mask, "fitness": score, "metadata": meta}

    witness = make_witness_fitness(pool, budget, fitness_cfg, member)
    init = None
    if method == "greedy_f" or bool(optimizer_cfg.get("shared_seed_init", False)):
        def init(expired):
            return best_shared_seed(pool, witness, budget, member, optimizer_cfg, seed, expired)
        meta["init"] = "best_shared_seed"
    if method in {"greedy_1swap", "greedy_f"}:
        mask, score, stopped, evals = greedy_1swap(
            witness, costs, budget, member, deadline,
            init=init, additions_only=method == "greedy_f",
        )
        meta.update({"stopped": stopped, "n_evals": evals})
    elif method == "two_unit":
        mask, score, stopped, evals = two_unit_greedy(
            witness,
            costs,
            budget,
            member,
            pool.relevance,
            deadline,
            d_max=int(optimizer_cfg.get("hycaro_d_max", 3)),
            init=init,
        )
        meta.update({"stopped": stopped, "n_evals": evals})
    elif method == "de_repair":
        mask, score, stopped, evals, generations = de_full_repair(
            pool, witness, budget, member, optimizer_cfg, seed, deadline
        )
        meta.update({"stopped": stopped, "n_evals": evals, "iterations": generations})
    else:
        raise ValueError(method)
    meta["n_selected"] = int(np.asarray(mask).sum())
    return {"method": name, "mask": mask, "fitness": float(score), "metadata": meta}
