#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from repomaro.metrics import file_prf, gold_coverage


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate externally produced context selections on ARB positive samples")
    ap.add_argument("--samples", nargs="+", required=True)
    ap.add_argument("--pred", required=True, help="JSONL with instance_id, method, selected_paths")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    try:
        from agent_retrieval_bench.baseline import iter_samples, target_gold_files  # type: ignore
    except ImportError as e:
        raise RuntimeError("Install Agent Retrieval Bench first") from e
    gold = {}
    for s in iter_samples([Path(x) for x in args.samples]):
        g = target_gold_files(s)
        if g:
            gold[str(s.get("id"))] = list(g)
    rows = []
    with Path(args.pred).open() as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            iid = str(r["instance_id"])
            if iid not in gold: continue
            p = file_prf(r.get("selected_paths", []), gold[iid])
            r.update({"gold_paths": gold[iid], "BCY": gold_coverage(r.get("selected_paths", []), gold[iid]), "file_precision":p.precision,"file_recall":p.recall,"file_f1":p.f1})
            rows.append(r)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out).open("w") as f:
        for r in rows: f.write(json.dumps(r)+"\n")

if __name__ == "__main__": main()
