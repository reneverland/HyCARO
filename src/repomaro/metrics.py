from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .schema import CandidatePool


def _norm_path(path: str) -> str:
    return Path(path).as_posix().lstrip("./")


@dataclass
class PRF:
    precision: float
    recall: float
    f1: float


def file_prf(selected_paths: Iterable[str], gold_paths: Iterable[str]) -> PRF:
    pred = {_norm_path(x) for x in selected_paths}
    gold = {_norm_path(x) for x in gold_paths}
    if not pred and not gold:
        return PRF(1.0, 1.0, 1.0)
    if not pred:
        return PRF(0.0, 0.0 if gold else 1.0, 0.0)
    inter = len(pred & gold)
    p = inter / len(pred)
    r = inter / len(gold) if gold else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return PRF(p, r, f1)


def gold_coverage(selected_paths: Iterable[str], gold_paths: Iterable[str]) -> float:
    gold = {_norm_path(x) for x in gold_paths}
    if not gold:
        return 1.0
    pred = {_norm_path(x) for x in selected_paths}
    return len(pred & gold) / len(gold)


def selected_paths(pool: CandidatePool, mask: np.ndarray) -> list[str]:
    # Unique file paths preserve the file-level interpretation used by ARB.
    out: list[str] = []
    seen: set[str] = set()
    for unit, keep in zip(pool.units, np.asarray(mask, dtype=bool).tolist()):
        if keep and unit.path not in seen:
            out.append(unit.path)
            seen.add(unit.path)
    return out


def selected_token_count(pool: CandidatePool, mask: np.ndarray) -> int:
    return int(pool.token_costs[np.asarray(mask, dtype=bool)].sum())


def line_level_prf(
    selected_spans: dict[str, list[tuple[int, int]]],
    gold_spans: dict[str, list[tuple[int, int]]],
) -> PRF:
    def lines(spans: dict[str, list[tuple[int, int]]]) -> set[tuple[str, int]]:
        out = set()
        for path, ranges in spans.items():
            p = _norm_path(path)
            for start, end in ranges:
                for line in range(int(start), int(end) + 1):
                    out.add((p, line))
        return out
    pred = lines(selected_spans)
    gold = lines(gold_spans)
    if not pred and not gold:
        return PRF(1.0, 1.0, 1.0)
    inter = len(pred & gold)
    p = inter / len(pred) if pred else 0.0
    r = inter / len(gold) if gold else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return PRF(p, r, f1)
