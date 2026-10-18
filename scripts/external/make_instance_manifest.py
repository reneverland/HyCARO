#!/usr/bin/env python
"""Dump the benchmark instances external baselines need, in the main venv.

External baselines (aider's RepoMap, LongCodeZip, LocAgent, CodeScout) pull in
tree-sitter and torch pins that conflict with ours, so they run in their own
virtual environments and cannot import `repomaro`. This writes the small amount
of state they need -- where each repository is checked out and what the issue
says -- so every baseline reads exactly the same instances we score ourselves.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from repomaro.adapters.swebench import load_swebench_lite


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkout-cache", default="cache/checkouts")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--out", default="results/external/instances_lite40.jsonl")
    args = ap.parse_args()

    instances = load_swebench_lite(args.checkout_cache, args.limit, offset=args.offset)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for inst in instances:
            fh.write(
                json.dumps(
                    {
                        "instance_id": inst.instance_id,
                        "repo": inst.metadata.get("repo"),
                        "base_commit": inst.metadata.get("base_commit"),
                        "root": str(inst.repo_root),
                        "issue": inst.query,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"wrote {len(instances)} instances -> {out}")


if __name__ == "__main__":
    main()
