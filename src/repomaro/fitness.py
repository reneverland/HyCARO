from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .schema import CandidatePool


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("embeddings must be a 2D matrix")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return x / norms


@dataclass
class FitnessComponents:
    total: float
    relevance: float
    structural: float
    redundancy: float


class SurrogateFitness:
    """Node-wise predecessor surrogate F_node(S) used by retrieval/optimizer controls.

    F_node = alpha * (1 - prod(1-r_j)) + beta * U_anchor(S) - gamma * R_sem(S)
    """

    def __init__(self, pool: CandidatePool, alpha: float, beta: float, gamma: float, sigma: float = 2.0):
        self.pool = pool
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.sigma = float(sigma)
        self.r = np.clip(pool.relevance.astype(np.float64), 0.0, 1.0 - 1e-12)
        embeddings = np.stack([u.embedding for u in pool.units], axis=0)
        embeddings = _normalize_rows(embeddings)
        sim = embeddings @ embeddings.T
        sim = np.maximum(sim, 0.0)
        np.fill_diagonal(sim, 0.0)
        self.sim = sim
        self.struct = self._build_structural_affinity(pool)
        self.node_to_index = {u.uid: i for i, u in enumerate(pool.units)}
        self.adjacency = self._candidate_adjacency(pool)

    def _build_structural_affinity(self, pool: CandidatePool) -> np.ndarray:
        A = len(pool.anchor_node_ids)
        m = len(pool.units)
        if A == 0:
            return np.zeros((0, m), dtype=np.float64)
        uid_to_idx = {u.uid: i for i, u in enumerate(pool.units)}
        out = np.zeros((A, m), dtype=np.float64)
        for ai, anchor in enumerate(pool.anchor_node_ids):
            if anchor not in pool.graph:
                continue
            lengths = pool.graph.hop_lengths(anchor)
            for uid, d in lengths.items():
                j = uid_to_idx.get(uid)
                if j is not None:
                    out[ai, j] = np.exp(-float(d) / max(self.sigma, 1e-12))
        return out

    @staticmethod
    def _candidate_adjacency(pool: CandidatePool) -> list[np.ndarray]:
        uid_to_idx = {u.uid: i for i, u in enumerate(pool.units)}
        out: list[np.ndarray] = []
        for u in pool.units:
            neigh = []
            if u.uid in pool.graph:
                for v in pool.graph.neighbors(u.uid):
                    if v in uid_to_idx:
                        neigh.append(uid_to_idx[v])
            out.append(np.asarray(sorted(set(neigh)), dtype=np.int64))
        return out

    def evaluate(self, mask: np.ndarray, return_components: bool = False) -> float | FitnessComponents:
        vals = self.evaluate_batch(np.asarray(mask, dtype=bool)[None, :], return_components=return_components)
        if return_components:
            return vals[0]
        return float(vals[0])

    def evaluate_batch(self, masks: np.ndarray, return_components: bool = False):
        masks = np.asarray(masks, dtype=bool)
        if masks.ndim != 2 or masks.shape[1] != len(self.pool.units):
            raise ValueError("masks must have shape [batch, m]")
        b, m = masks.shape
        if b == 0:
            return [] if return_components else np.zeros(0, dtype=np.float64)

        rel_terms = np.where(masks, 1.0 - self.r[None, :], 1.0)
        u_rel = 1.0 - np.prod(rel_terms, axis=1)

        if self.struct.shape[0] == 0:
            u_str = np.zeros(b, dtype=np.float64)
        else:
            # [b, A, m] is manageable because candidate pools are capped.
            masked = np.where(masks[:, None, :], self.struct[None, :, :], 0.0)
            u_str = masked.max(axis=2).mean(axis=1)

        k = masks.sum(axis=1).astype(np.float64)
        mf = masks.astype(np.float64)
        pair_double_sum = np.einsum("bi,ij,bj->b", mf, self.sim, mf, optimize=True)
        denom = k * (k - 1.0)
        redundancy = np.zeros(b, dtype=np.float64)
        valid = denom > 0
        redundancy[valid] = pair_double_sum[valid] / denom[valid]

        total = self.alpha * u_rel + self.beta * u_str - self.gamma * redundancy
        if not return_components:
            return total
        return [
            FitnessComponents(float(total[i]), float(u_rel[i]), float(u_str[i]), float(redundancy[i]))
            for i in range(b)
        ]

    def build_local_state(self, mask: np.ndarray) -> "LocalSubsetState":
        return LocalSubsetState(self, np.asarray(mask, dtype=bool))


class FitnessCache:
    """Hash cache keyed by decoded binary mask with batch deduplication."""

    def __init__(self, evaluator: SurrogateFitness):
        self.evaluator = evaluator
        self.cache: dict[bytes, float] = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(mask: np.ndarray) -> bytes:
        mask = np.asarray(mask, dtype=np.uint8)
        return len(mask).to_bytes(4, "little") + np.packbits(mask, bitorder="little").tobytes()

    def evaluate_batch(self, masks: np.ndarray) -> np.ndarray:
        masks = np.asarray(masks, dtype=bool)
        out = np.zeros(len(masks), dtype=np.float64)
        missing_by_key: dict[bytes, tuple[int, np.ndarray]] = {}
        positions_by_key: dict[bytes, list[int]] = {}
        for i, mask in enumerate(masks):
            k = self.key(mask)
            positions_by_key.setdefault(k, []).append(i)
            if k in self.cache:
                out[i] = self.cache[k]
                self.hits += 1
            elif k not in missing_by_key:
                missing_by_key[k] = (i, mask.copy())
            else:
                self.hits += 1  # within-batch duplicate
        if missing_by_key:
            keys = list(missing_by_key)
            miss_masks = np.stack([missing_by_key[k][1] for k in keys], axis=0)
            values = self.evaluator.evaluate_batch(miss_masks)
            self.misses += len(keys)
            for k, value in zip(keys, values.tolist()):
                self.cache[k] = float(value)
                for pos in positions_by_key[k]:
                    out[pos] = float(value)
        return out

    def evaluate(self, mask: np.ndarray) -> float:
        return float(self.evaluate_batch(np.asarray(mask, dtype=bool)[None, :])[0])


class LocalSubsetState:
    """Exact O(s + A) scoring for one-unit add/drop/swap proposals from a fixed base subset."""

    def __init__(self, evaluator: SurrogateFitness, mask: np.ndarray):
        self.ev = evaluator
        self.mask = np.asarray(mask, dtype=bool).copy()
        self.selected = np.flatnonzero(self.mask)
        self.selected_set = set(self.selected.tolist())
        self.k = len(self.selected)
        self.log_rel_sum = float(np.log1p(-self.ev.r[self.selected]).sum()) if self.k else 0.0
        if self.k >= 2:
            sub = self.ev.sim[np.ix_(self.selected, self.selected)]
            self.pair_sum = float(np.triu(sub, k=1).sum())
        else:
            self.pair_sum = 0.0
        A = self.ev.struct.shape[0]
        if A and self.k:
            vals = self.ev.struct[:, self.selected]
            order = np.argsort(-vals, axis=1, kind="mergesort")
            top1_pos = order[:, 0]
            self.top1_idx = self.selected[top1_pos]
            self.top1_val = vals[np.arange(A), top1_pos]
            if self.k >= 2:
                top2_pos = order[:, 1]
                self.top2_val = vals[np.arange(A), top2_pos]
            else:
                self.top2_val = np.zeros(A, dtype=np.float64)
        else:
            self.top1_idx = np.full(A, -1, dtype=np.int64)
            self.top1_val = np.zeros(A, dtype=np.float64)
            self.top2_val = np.zeros(A, dtype=np.float64)
        self.base_score = self._compose(self.log_rel_sum, self.top1_val, self.pair_sum, self.k)

    def _compose(self, log_rel_sum: float, struct_max: np.ndarray, pair_sum: float, k: int) -> float:
        u_rel = 1.0 - float(np.exp(log_rel_sum))
        u_str = float(struct_max.mean()) if len(struct_max) else 0.0
        red = (2.0 * pair_sum / (k * (k - 1))) if k >= 2 else 0.0
        return self.ev.alpha * u_rel + self.ev.beta * u_str - self.ev.gamma * red

    def score_add(self, j: int) -> float:
        if j in self.selected_set:
            return -np.inf
        log_rel = self.log_rel_sum + float(np.log1p(-self.ev.r[j]))
        struct = np.maximum(self.top1_val, self.ev.struct[:, j]) if self.ev.struct.shape[0] else self.top1_val
        pair_sum = self.pair_sum + float(self.ev.sim[j, self.selected].sum())
        return self._compose(log_rel, struct, pair_sum, self.k + 1)

    def score_drop(self, j: int) -> float:
        if j not in self.selected_set:
            return -np.inf
        log_rel = self.log_rel_sum - float(np.log1p(-self.ev.r[j]))
        if self.ev.struct.shape[0]:
            struct = np.where(self.top1_idx == j, self.top2_val, self.top1_val)
        else:
            struct = self.top1_val
        others = self.selected[self.selected != j]
        pair_sum = self.pair_sum - float(self.ev.sim[j, others].sum())
        return self._compose(log_rel, struct, pair_sum, self.k - 1)

    def score_swap(self, out_j: int, in_j: int) -> float:
        if out_j not in self.selected_set or in_j in self.selected_set or out_j == in_j:
            return -np.inf
        log_rel = (
            self.log_rel_sum
            - float(np.log1p(-self.ev.r[out_j]))
            + float(np.log1p(-self.ev.r[in_j]))
        )
        if self.ev.struct.shape[0]:
            base_without = np.where(self.top1_idx == out_j, self.top2_val, self.top1_val)
            struct = np.maximum(base_without, self.ev.struct[:, in_j])
        else:
            struct = self.top1_val
        others = self.selected[self.selected != out_j]
        pair_sum = (
            self.pair_sum
            - float(self.ev.sim[out_j, others].sum())
            + float(self.ev.sim[in_j, others].sum())
        )
        return self._compose(log_rel, struct, pair_sum, self.k)


class IssueCoverageFitness:
    """F(S) = alpha*Cov(q,S) + beta*Supp(S|q) - gamma*Red(S) - lambda*ell(S)/B.

    The three terms answer three separate questions and nothing else:

    * Cov(q, S) -- does S contain the information the issue asks about? The needs
      are read off the query itself (the identifiers it names, plus the issue as
      a whole semantically), not off a retrieval ranking, so covering the same
      need twice is worth nothing.
    * Supp(S | q) -- can a patch actually be written against S? A selected unit
      is supported when the repository relations it participates in (its class,
      its callees, its tests, ...) are realised inside S. Support is weighted by
      how much the unit matters to the query, so pulling in a neighbour only pays
      when it props up something the issue is about.
    * cost -- tokens are priced explicitly on top of the hard budget, so the
      objective can stop early instead of always packing B.

    Both utility terms live in [0, 1], which makes alpha and beta directly
    comparable.
    """

    #: Packaging-only relation; two units sharing a file is not structural support.
    NON_SUPPORT_RELATIONS = frozenset({"contains"})

    def __init__(
        self,
        pool: CandidatePool,
        budget: int,
        alpha: float = 1.0,
        beta: float = 0.35,
        gamma: float = 0.0,
        cost_lambda: float = 0.25,
        semantic_need_share: float = 0.5,
        max_lexical_needs: int = 32,
        max_need_df_ratio: float = 0.5,
        semantic_need_depth: int = 4,
        lexical_need_depth: int = 1,
        need_depth_decay: float = 0.5,
        redundancy_delta: float = 0.3,
        support_relations: Iterable[str] | None = None,
    ):
        self.pool = pool
        self.budget = float(max(int(budget), 1))
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.cost_lambda = float(cost_lambda)
        self.redundancy_delta = float(redundancy_delta)
        self.m = len(pool.units)
        self.ell = pool.token_costs.astype(np.float64)
        self.r = np.clip(pool.relevance.astype(np.float64), 0.0, 1.0)

        embeddings = _normalize_rows(np.stack([u.embedding for u in pool.units], axis=0))
        sim = np.maximum(embeddings @ embeddings.T, 0.0)
        np.fill_diagonal(sim, 0.0)
        self.sim = sim
        excess = np.maximum(sim - self.redundancy_delta, 0.0)
        self.red_pair = excess
        self.pair_norm = float(self.m * (self.m - 1))

        self.need_affinity, self.need_w, depths = self._build_needs(
            pool,
            semantic_need_share=float(semantic_need_share),
            max_lexical_needs=int(max_lexical_needs),
            max_need_df_ratio=float(max_need_df_ratio),
            semantic_need_depth=int(semantic_need_depth),
            lexical_need_depth=int(lexical_need_depth),
        )
        self.z_need = float(self.need_w.sum())
        self.depth_w = self._depth_weights(depths, float(need_depth_decay))
        self.need_depth = int(self.depth_w.shape[1])

        self.support_mass, self.unit_support_total = self._build_support(pool, support_relations)
        # Membership indicator of each relation; static, so it is built once
        # rather than on every evaluation.
        self.support_present = (self.support_mass > 0.0).astype(np.float64)
        self.prize = self.r.copy()

    # ---- construction -----------------------------------------------------

    def _build_needs(
        self,
        pool: CandidatePool,
        semantic_need_share: float,
        max_lexical_needs: int,
        max_need_df_ratio: float,
        semantic_need_depth: int,
        lexical_need_depth: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Rows of an [n_needs, m] affinity matrix, one row per thing the issue asks for.

        Lexical needs are the identifiers the issue names, weighted by inverse
        document frequency inside the pool so that a rare symbol (the thing the
        traceback actually points at) outweighs a ubiquitous one. The semantic
        need is the issue as a whole, and its affinity is the dense retrieval
        score, which is already cos(q, h_j).
        """
        from .retrieval import tokenize

        m = self.m
        rows: list[np.ndarray] = []
        weights: list[float] = []

        query_terms = [t for t in dict.fromkeys(tokenize(pool.query or "")) if len(t) >= 3]
        if query_terms and m > 0:
            unit_terms = [set(tokenize(u.text)) for u in pool.units]
            cands: list[tuple[float, np.ndarray]] = []
            for term in query_terms:
                hit = np.fromiter((term in terms for terms in unit_terms), dtype=bool, count=m)
                df = int(hit.sum())
                if df == 0 or df > max(1, int(max_need_df_ratio * m)):
                    continue
                cands.append((float(np.log(1.0 + m / df)), hit.astype(np.float64)))
            cands.sort(key=lambda c: -c[0])
            for w, hit in cands[:max_lexical_needs]:
                weights.append(w)
                rows.append(hit)
        depths = [max(1, lexical_need_depth)] * len(rows)

        dense = np.clip(
            np.asarray([getattr(u, "dense_score", 0.0) or 0.0 for u in pool.units], dtype=np.float64),
            0.0,
            1.0,
        )
        lexical_mass = float(sum(weights))
        share = min(max(semantic_need_share, 0.0), 1.0 - 1e-9)
        w_sem = 1.0 if lexical_mass <= 0.0 else lexical_mass * share / (1.0 - share)
        rows.append(dense)
        weights.append(w_sem)
        depths.append(max(1, semantic_need_depth))

        return (
            np.stack(rows, axis=0),
            np.asarray(weights, dtype=np.float64),
            np.asarray(depths, dtype=np.int64),
        )

    @staticmethod
    def _depth_weights(depths: np.ndarray, decay: float) -> np.ndarray:
        """[n_needs, max_depth] geometric weights that regularise the max operator.

        A need scored by max is answered in full by a single unit, so a query that
        yields only one need (an issue naming no identifier at all) leaves the
        coverage term saturated and alpha inert. Reading a need off its best p
        units under decaying weights removes that cliff while keeping the term
        monotone submodular -- an ordered weighted average with decreasing
        weights is -- and reduces exactly to max at p = 1.
        """
        depths = np.asarray(depths, dtype=np.int64)
        p_max = int(depths.max()) if depths.size else 1
        out = np.zeros((depths.size, p_max), dtype=np.float64)
        for i, p in enumerate(depths.tolist()):
            w = np.power(float(decay), np.arange(p, dtype=np.float64))
            out[i, :p] = w / w.sum()
        return out

    def _build_support(
        self, pool: CandidatePool, support_relations: Iterable[str] | None
    ) -> tuple[np.ndarray, np.ndarray]:
        """[m, n_edges] matrix of relation weights plus each unit's total support mass."""
        from .backbone import DEFAULT_EDGE_WEIGHT, RELATION_WEIGHT

        graph = pool.graph
        edges = list(getattr(graph, "edges", None) or []) if graph is not None else []
        allowed = set(support_relations) if support_relations is not None else None
        uid_to_idx = {u.uid: i for i, u in enumerate(pool.units)}
        cols: list[np.ndarray] = []
        for edge in edges:
            if allowed is None:
                if edge.relation in self.NON_SUPPORT_RELATIONS:
                    continue
            elif edge.relation not in allowed:
                continue
            idxs = [uid_to_idx[u] for u in edge.members if u in uid_to_idx]
            if len(idxs) < 2:
                continue
            col = np.zeros(self.m, dtype=np.float64)
            col[idxs] = float(RELATION_WEIGHT.get(edge.relation, DEFAULT_EDGE_WEIGHT))
            cols.append(col)
        if not cols:
            return np.zeros((self.m, 0), dtype=np.float64), np.zeros(self.m, dtype=np.float64)
        mass = np.stack(cols, axis=1)
        return mass, mass.sum(axis=1)

    # ---- terms ------------------------------------------------------------

    def _coverage(self, masks: np.ndarray) -> np.ndarray:
        if self.z_need <= 0.0 or self.need_affinity.shape[0] == 0:
            return np.zeros(masks.shape[0], dtype=np.float64)
        out = np.zeros(masks.shape[0], dtype=np.float64)
        p = self.need_depth
        for b in range(masks.shape[0]):
            sel = masks[b]
            if not sel.any():
                continue
            sub = self.need_affinity[:, sel]
            if p == 1:
                best = sub.max(axis=1, keepdims=True)
            else:
                if sub.shape[1] > p:
                    # Only the p largest per need are read, so select them
                    # instead of ordering the whole row.
                    top = -np.sort(np.partition(-sub, p - 1, axis=1)[:, :p], axis=1)
                else:
                    top = -np.sort(-sub, axis=1)[:, :p]
                best = np.zeros((sub.shape[0], p), dtype=np.float64)
                best[:, : top.shape[1]] = top
            answered = (best * self.depth_w[:, : best.shape[1]]).sum(axis=1)
            out[b] = float((self.need_w * answered).sum() / self.z_need)
        return out

    def _support(self, masks: np.ndarray) -> np.ndarray:
        """Prize-weighted share of each selected unit's relations that S realises."""
        if self.support_mass.shape[1] == 0:
            return np.zeros(masks.shape[0], dtype=np.float64)
        mf = masks.astype(np.float64)
        counts = mf @ self.support_present  # [b, n_edges] members of each relation inside S
        realized = (counts >= 2.0).astype(np.float64)
        realized_mass = realized @ self.support_mass.T  # [b, m]
        total = self.unit_support_total
        frac = np.zeros_like(realized_mass)
        has = total > 0.0
        frac[:, has] = realized_mass[:, has] / total[None, has]
        weighted = (mf * self.prize[None, :] * frac).sum(axis=1)
        denom = (mf * self.prize[None, :]).sum(axis=1)
        return np.divide(weighted, denom, out=np.zeros_like(weighted), where=denom > 1e-12)

    def _redundancy(self, mf: np.ndarray) -> np.ndarray:
        if self.gamma <= 0.0 or self.pair_norm <= 0.0:
            return np.zeros(mf.shape[0], dtype=np.float64)
        return np.einsum("bi,ij,bj->b", mf, self.red_pair, mf, optimize=True) / self.pair_norm

    # ---- api --------------------------------------------------------------

    def cost(self, mask: np.ndarray) -> float:
        return float(self.ell[np.asarray(mask, dtype=bool)].sum())

    def evaluate(self, mask: np.ndarray, return_components: bool = False) -> float | FitnessComponents:
        vals = self.evaluate_batch(np.asarray(mask, dtype=bool)[None, :], return_components=return_components)
        return vals[0] if return_components else float(vals[0])

    def evaluate_batch(self, masks: np.ndarray, return_components: bool = False):
        masks = np.asarray(masks, dtype=bool)
        if masks.ndim != 2 or masks.shape[1] != self.m:
            raise ValueError("masks must have shape [batch, m]")
        if masks.shape[0] == 0:
            return [] if return_components else np.zeros(0, dtype=np.float64)

        mf = masks.astype(np.float64)
        cov = self._coverage(masks)
        supp = self._support(masks)
        red = self._redundancy(mf)
        total = self.alpha * cov + self.beta * supp - self.gamma * red
        if self.cost_lambda > 0.0:
            total = total - self.cost_lambda * (mf * self.ell[None, :]).sum(axis=1) / self.budget
        if not return_components:
            return total
        return [
            FitnessComponents(float(total[i]), float(cov[i]), float(supp[i]), float(red[i]))
            for i in range(masks.shape[0])
        ]

    def build_local_state(self, mask: np.ndarray) -> "FullEvalState":
        return FullEvalState(self, mask)


class FullEvalState:
    """Local-search state for objectives whose terms are not additive over units.

    Coverage saturates and support is a ratio, so neither admits an incremental
    mass; re-evaluating the candidate mask is the only consistent baseline.
    """

    def __init__(self, evaluator: IssueCoverageFitness, mask: np.ndarray):
        self.ev = evaluator
        self.mask = np.asarray(mask, dtype=bool).copy()
        self.selected = np.flatnonzero(self.mask)
        self.selected_set = set(self.selected.tolist())
        self.k = len(self.selected)
        self.base_score = float(self.ev.evaluate(self.mask))

    def score_remove(self, nodes: np.ndarray) -> float:
        remove = [j for j in np.asarray(nodes, dtype=int).tolist() if j in self.selected_set]
        if not remove:
            return self.base_score
        keep = self.mask.copy()
        keep[remove] = False
        return float(self.ev.evaluate(keep))

    def score_add(self, nodes: np.ndarray) -> float:
        add = [j for j in np.asarray(nodes, dtype=int).tolist() if j not in self.selected_set]
        if not add:
            return self.base_score
        keep = self.mask.copy()
        keep[add] = True
        return float(self.ev.evaluate(keep))
