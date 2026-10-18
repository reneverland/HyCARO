from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from .graph import Hypergraph, RepositoryScanner
from .retrieval import BM25Retriever, DenseRetriever, greedy_pack_by_ranking, minmax_normalize
from .schema import CandidatePool, ContextUnit
from .tokenization import TokenCounter

_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


class CandidatePoolBuilder:
    def __init__(self, config: dict[str, Any], downstream_model: str | None = None):
        self.config = config
        self.token_counter = TokenCounter(downstream_model)
        dense_cfg = config.get("dense", {}) or {}
        self.dense = DenseRetriever(
            enabled=bool(dense_cfg.get("enabled", False)),
            model_name=str(dense_cfg.get("model", "Qwen/Qwen3-Embedding-4B")),
            device=str(dense_cfg.get("device", "cuda")),
            batch_size=int(dense_cfg.get("batch_size", 8)),
            max_seq_length=int(dense_cfg.get("max_seq_length", 1024)),
        )
        self.bm25 = BM25Retriever()

    def build(
        self,
        query: str,
        repo_root: str | Path,
        budget: int,
        fitness_cfg: dict[str, Any],
    ) -> CandidatePool:
        graph_cfg = self.config.get("graph") or {}
        scanner = RepositoryScanner(
            token_counter=self.token_counter,
            granularity=str(self.config.get("granularity", "symbol")),
            include_extensions=self.config.get("include_extensions"),
            exclude_dirs=self.config.get("exclude_dirs"),
            max_file_bytes=int(self.config.get("max_file_bytes", 500_000)),
            generic_chunk_lines=int(self.config.get("generic_chunk_lines", 120)),
            relations=graph_cfg.get("relations"),
            traversal_relations=graph_cfg.get("traversal_relations"),
            max_symbol_fanout=int(graph_cfg.get("max_symbol_fanout", 4)),
        )
        scan = scanner.scan(repo_root)
        all_units = scan.units
        graph = scan.graph
        if not all_units:
            raise ValueError(f"No eligible repository units found in {repo_root}")

        lexical_raw = self.bm25.score(query, all_units)
        lexical = minmax_normalize(lexical_raw)

        dense_top_k = int(self.config.get("dense_top_k", 0))
        dense_enabled = bool((self.config.get("dense") or {}).get("enabled", False)) and dense_top_k > 0
        if dense_enabled:
            dense_raw, embeddings = self.dense.score(query, all_units)
            dense = minmax_normalize(dense_raw)
        else:
            # Hashing embeddings remain useful for redundancy even when the paper-scale dense retriever is disabled.
            embeddings = self.dense.encode_units(all_units)
            dense_raw, _ = self.dense.score(query, all_units, embeddings=embeddings)
            dense = minmax_normalize(dense_raw)

        for i, unit in enumerate(all_units):
            unit.lexical_score = float(lexical[i])
            unit.dense_score = float(dense[i])
            unit.embedding = np.asarray(embeddings[i], dtype=np.float64)

        lex_top = self._top_indices(lexical, int(self.config.get("lexical_top_k", 60)))
        dense_top = self._top_indices(dense, dense_top_k)
        seed_top_k = int(self.config.get("structural_seed_top_k", 12))
        combined = 0.5 * lexical + 0.5 * dense
        structural_seeds = self._top_indices(combined, seed_top_k)
        expanded = self._expand_indices(all_units, graph, structural_seeds, int(self.config.get("structural_expand_hops", 1)))

        candidate_ids = set(lex_top) | set(dense_top) | set(expanded)
        if not candidate_ids:
            candidate_ids = set(self._top_indices(combined, min(20, len(all_units))))

        max_units = int(self.config.get("max_units", 120))
        if len(candidate_ids) > max_units:
            ranked = sorted(candidate_ids, key=lambda i: (-combined[i], all_units[i].uid))[:max_units]
            candidate_ids = set(ranked)

        selected_idx = sorted(candidate_ids)
        units = [all_units[i] for i in selected_idx]
        subgraph = graph.induced(u.uid for u in units)

        lex_w = float(fitness_cfg.get("relevance_lexical_weight", 0.5))
        dense_w = float(fitness_cfg.get("relevance_dense_weight", 0.5))
        denom = max(lex_w + dense_w, 1e-12)
        for u in units:
            u.relevance = float(np.clip((lex_w * u.lexical_score + dense_w * u.dense_score) / denom, 0.0, 1.0))

        anchor_node_ids = self._anchors(query, units, int(fitness_cfg.get("anchor_count", 5)))
        self._assign_structural_scores(units, subgraph, anchor_node_ids, sigma=float(fitness_cfg.get("sigma", 2.0)))
        pool = CandidatePool(
            query=query,
            repo_root=Path(repo_root).resolve(),
            units=units,
            graph=subgraph,
            anchor_node_ids=anchor_node_ids,
            metadata={
                "all_unit_count": len(all_units),
                "candidate_count": len(units),
                "budget": budget,
                "dense_backend": "sentence-transformers" if dense_enabled else "hashing-fallback",
            },
        )
        return pool

    @staticmethod
    def _top_indices(scores: np.ndarray, k: int) -> list[int]:
        if k <= 0 or len(scores) == 0:
            return []
        k = min(k, len(scores))
        return np.argsort(-scores, kind="mergesort")[:k].tolist()

    @staticmethod
    def _expand_indices(units: list[ContextUnit], graph: Hypergraph, seeds: list[int], hops: int) -> list[int]:
        uid_to_idx = {u.uid: i for i, u in enumerate(units)}
        chosen: set[int] = set(seeds)
        frontier = {units[i].uid for i in seeds}
        for _ in range(max(0, hops)):
            nxt: set[str] = set()
            for uid in frontier:
                if uid not in graph:
                    continue
                nxt.update(graph.neighbors(uid))
            frontier = nxt
            chosen.update(uid_to_idx[u] for u in frontier if u in uid_to_idx)
        return sorted(chosen)

    @staticmethod
    def _anchors(query: str, units: list[ContextUnit], k: int) -> list[str]:
        tokens = {x.lower() for x in _IDENTIFIER_RE.findall(query)}
        explicit: list[ContextUnit] = []
        for u in units:
            hay = f"{u.path} {u.symbol or ''}".lower()
            if any(t in hay for t in tokens):
                explicit.append(u)
        explicit.sort(key=lambda u: (-u.relevance, u.uid))
        anchors = [u.uid for u in explicit[:k]]
        if len(anchors) < k:
            for u in sorted(units, key=lambda x: (-x.relevance, x.uid)):
                if u.uid not in anchors:
                    anchors.append(u.uid)
                if len(anchors) >= k:
                    break
        return anchors

    @staticmethod
    def _assign_structural_scores(units: list[ContextUnit], graph: Hypergraph, anchors: list[str], sigma: float) -> None:
        if not anchors:
            for u in units:
                u.structural_score = 0.0
            return
        max_scores = {u.uid: 0.0 for u in units}
        for a in anchors:
            if a not in graph:
                continue
            lengths = graph.hop_lengths(a, cutoff=6)
            for uid, d in lengths.items():
                if uid in max_scores:
                    max_scores[uid] = max(max_scores[uid], float(np.exp(-float(d) / max(sigma, 1e-6))))
        for u in units:
            u.structural_score = max_scores.get(u.uid, 0.0)


def seed_masks(pool: CandidatePool, budget: int) -> dict[str, np.ndarray]:
    costs = pool.token_costs
    return {
        "bm25": greedy_pack_by_ranking(pool.lexical_scores, costs, budget),
        "dense": greedy_pack_by_ranking(pool.dense_scores, costs, budget),
        "struct": greedy_pack_by_ranking(pool.structural_scores, costs, budget),
    }
