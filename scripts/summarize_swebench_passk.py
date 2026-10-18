#!/usr/bin/env python
"""Aggregate official SWE-bench reports into pass@1 and pass@k."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _load_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="+", help="Official harness report JSON files")
    ap.add_argument("--out", default="results/swebench_passk.json")
    args = ap.parse_args()

    # method -> sample_idx -> resolved_ids
    by: dict[str, dict[int, set[str]]] = defaultdict(dict)
    submitted: dict[str, set[str]] = defaultdict(set)
    for raw in args.reports:
        path = Path(raw)
        report = _load_report(path)
        # model_name_or_path is in the filename: gpt-5.6-luna+RepoMARO+s0.<run_id>.json
        stem = path.name
        method = "unknown"
        sample = 0
        if "+" in stem:
            mid = stem.split("+", 1)[1]
            mid = mid.split(".", 1)[0]
            if "+s" in mid:
                method, _, s = mid.rpartition("+s")
                sample = int(s)
            else:
                method = mid
        resolved = set(report.get("resolved_ids") or [])
        submitted[method].update(report.get("submitted_ids") or [])
        by[method].setdefault(sample, set()).update(resolved)

    rows = []
    print(f"{'method':16} {'n':>4} {'pass@1':>8} {'pass@k':>8} {'k':>3} resolved@1  resolved@k")
    for method in sorted(by):
        samples = by[method]
        k = max(samples) + 1 if samples else 0
        union: set[str] = set()
        for ids in samples.values():
            union |= ids
        first = samples.get(0, set())
        n = max(len(submitted[method]), 1)
        p1 = len(first) / n
        pk = len(union) / n
        rows.append({
            "method": method,
            "n_instances": n,
            "k": k,
            "pass@1": p1,
            "pass@k": pk,
            "resolved_s0": sorted(first),
            "resolved_any": sorted(union),
        })
        print(
            f"{method:16} {n:4d} {p1:8.3f} {pk:8.3f} {k:3d} "
            f"{len(first):10d} {len(union):10d}"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
