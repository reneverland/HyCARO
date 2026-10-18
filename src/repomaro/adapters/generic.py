from __future__ import annotations

import json
from pathlib import Path

from ..schema import SelectionInstance


def load_generic_jsonl(path: str | Path) -> list[SelectionInstance]:
    out: list[SelectionInstance] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            out.append(
                SelectionInstance(
                    instance_id=str(row["instance_id"]),
                    query=str(row["query"]),
                    repo_root=Path(row["repo_root"]),
                    gold_paths=list(row.get("gold_paths", [])),
                    gold_spans={k: [tuple(x) for x in v] for k, v in row.get("gold_spans", {}).items()},
                    metadata=dict(row.get("metadata", {})),
                )
            )
    return out
