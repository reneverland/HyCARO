#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

from repomaro.candidate_pool import CandidatePoolBuilder
from repomaro.config import load_config
from repomaro.io_utils import selector_record
from repomaro.runner import run_selector

ABLATIONS = {
    "full": {},
    "no_budget_repair": {"optimizer": {"enable_budget_repair": False}},
    "no_repo_local_search": {"optimizer": {"enable_local_search": False}},
    "no_retrieval_seeding": {"optimizer": {"enable_retrieval_seeding": False}},
    "no_memory": {"optimizer": {"enable_memory": False}},
    "no_population_difference": {"optimizer": {"enable_population_difference": False}},
    "no_incremental_local": {"optimizer": {"enable_incremental_local": False}},
    "no_early_stopping": {"optimizer": {"enable_early_stopping": False}},
    "no_cost_term": {"fitness": {"ball": {"cost_lambda": 0.0}}},
    "no_support_term": {"fitness": {"ball": {"beta": 0.0}}},
    "radius_zero": {"optimizer": {"ball_radius": 0}},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--issue", required=True)
    ap.add_argument("--instance-id", default="local-instance")
    ap.add_argument("--config", default="configs/hycaro.yaml")
    ap.add_argument("--budget", type=int, default=8192)
    ap.add_argument("--out", default="results/ablation.json")
    args = ap.parse_args()

    base = load_config(args.config)
    pool = CandidatePoolBuilder(base["candidate_pool"], (base.get("downstream") or {}).get("model")).build(
        args.issue, args.repo, args.budget, base["fitness"]
    )
    rows = []
    for name, changes in ABLATIONS.items():
        cfg = copy.deepcopy(base)
        cfg["optimizer"].update(changes.get("optimizer", {}))
        fit_changes = dict(changes.get("fitness", {}))
        ball_changes = fit_changes.pop("ball", None)
        cfg["fitness"].update(fit_changes)
        if ball_changes:
            cfg["fitness"].setdefault("ball", {}).update(ball_changes)
        t0 = time.perf_counter()
        result = run_selector("repomaro", pool, args.budget, cfg)
        row = selector_record(args.instance_id, pool, result)
        row["ablation"] = name
        row["selector_time_seconds"] = time.perf_counter() - t0
        rows.append(row)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
