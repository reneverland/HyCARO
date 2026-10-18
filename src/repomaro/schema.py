from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .graph import Hypergraph


@dataclass(slots=True)
class ContextUnit:
    uid: str
    path: str
    text: str
    kind: str = "chunk"
    symbol: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    token_cost: int = 0
    node_id: str | None = None
    lexical_score: float = 0.0
    dense_score: float = 0.0
    structural_score: float = 0.0
    relevance: float = 0.0
    embedding: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def short_name(self) -> str:
        if self.symbol:
            return f"{self.path}::{self.symbol}"
        return self.path


@dataclass
class CandidatePool:
    query: str
    repo_root: Path
    units: list[ContextUnit]
    graph: "Hypergraph"
    anchor_node_ids: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.units)

    @property
    def token_costs(self) -> np.ndarray:
        return np.asarray([u.token_cost for u in self.units], dtype=np.int64)

    @property
    def relevance(self) -> np.ndarray:
        return np.asarray([u.relevance for u in self.units], dtype=np.float64)

    @property
    def lexical_scores(self) -> np.ndarray:
        return np.asarray([u.lexical_score for u in self.units], dtype=np.float64)

    @property
    def dense_scores(self) -> np.ndarray:
        return np.asarray([u.dense_score for u in self.units], dtype=np.float64)

    @property
    def structural_scores(self) -> np.ndarray:
        return np.asarray([u.structural_score for u in self.units], dtype=np.float64)

    def selected_units(self, mask: np.ndarray) -> list[ContextUnit]:
        return [u for u, keep in zip(self.units, mask.tolist()) if keep]


@dataclass
class SelectionInstance:
    instance_id: str
    query: str
    repo_root: Path
    gold_paths: list[str] = field(default_factory=list)
    gold_spans: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
