from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..checkout import ensure_checkout
from ..schema import SelectionInstance

_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? ")


def parse_gold_spans(patch: str) -> dict[str, list[tuple[int, int]]]:
    """Old-side line ranges the reference patch touches, keyed by file.

    Diagnostic use only: this reads the answer, so anything derived from it must
    never reach a selector or a prompt on a reported run.
    """
    out: dict[str, list[tuple[int, int]]] = {}
    current = None
    for line in patch.splitlines():
        if line.startswith("--- a/"):
            current = line[6:].strip()
        elif line.startswith("+++ b/"):
            path = line[6:].strip()
            current = None if path == "/dev/null" else path
        elif line.startswith("@@") and current:
            m = _HUNK.match(line)
            if m:
                start = int(m.group(1))
                length = int(m.group(2) or 1)
                out.setdefault(current, []).append((start, start + max(length, 1) - 1))
    return out


def load_swebench_lite(
    checkout_cache: str | Path,
    limit: int | None = None,
    split: str = "test",
    offset: int = 0,
    instance_ids: list[str] | None = None,
) -> list[SelectionInstance]:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as e:
        raise RuntimeError("Install benchmark extras: `pip install -e '.[benchmarks]'`") from e
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split=split)
    wanted = set(instance_ids) if instance_ids else None
    out: list[SelectionInstance] = []
    skipped = 0
    for row_obj in ds:
        row: dict[str, Any] = dict(row_obj)
        if wanted is not None and str(row["instance_id"]) not in wanted:
            continue
        if wanted is None and skipped < int(offset or 0):
            skipped += 1
            continue
        if not (len(out) % 25):
            print(f"  checkout {offset + len(out) + 1} {row['repo']}@{str(row['base_commit'])[:10]}", flush=True)
        root = ensure_checkout(str(row["repo"]), str(row["base_commit"]), checkout_cache)
        spans = parse_gold_spans(str(row.get("patch") or ""))
        out.append(
            SelectionInstance(
                instance_id=str(row["instance_id"]),
                query=str(row["problem_statement"]),
                repo_root=root,
                gold_paths=sorted(spans),
                gold_spans=spans,
                metadata={"repo": row["repo"], "base_commit": row["base_commit"]},
            )
        )
        if limit and len(out) >= limit:
            break
    return out


def write_swebench_predictions(predictions: list[dict[str, str]], path: str | Path) -> Path:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    # Official harness accepts a JSON list in the documented format.
    path.write_text(json.dumps(predictions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
