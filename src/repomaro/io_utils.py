from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import selected_paths, selected_token_count
from .schema import CandidatePool


def selector_record(instance_id: str, pool: CandidatePool, result: dict[str, Any]) -> dict[str, Any]:
    mask = np.asarray(result["mask"], dtype=bool)
    units = pool.selected_units(mask)
    return {
        "instance_id": instance_id,
        "method": result["method"],
        "fitness": float(result["fitness"]),
        "selected_tokens": selected_token_count(pool, mask),
        "selected_paths": selected_paths(pool, mask),
        "selected_units": [
            {
                "uid": u.uid,
                "path": u.path,
                "kind": u.kind,
                "symbol": u.symbol,
                "start_line": u.start_line,
                "end_line": u.end_line,
                "token_cost": u.token_cost,
                "relevance": u.relevance,
            }
            for u in units
        ],
        "metadata": result.get("metadata", {}),
    }


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
