#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    args = ap.parse_args()
    groups = defaultdict(list)
    with Path(args.path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line); groups[row["method"]].append(row)
    for method, rows in groups.items():
        keys = ["fitness", "selected_tokens", "BCY", "file_f1", "selector_time_seconds"]
        print(f"\n{method} (n={len(rows)})")
        for key in keys:
            vals = [r[key] for r in rows if key in r and isinstance(r[key], (int, float))]
            if vals:
                print(f"  {key}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")


if __name__ == "__main__":
    main()
