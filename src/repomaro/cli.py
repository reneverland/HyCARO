from __future__ import annotations

import argparse
import json
from pathlib import Path

from .candidate_pool import CandidatePoolBuilder
from .config import load_config
from .io_utils import selector_record
from .runner import BUILTIN_METHODS, run_selector


def _select(args) -> None:
    cfg = load_config(args.config)
    budget = int(args.budget or cfg.get("budget_tokens", 8192))
    model = (cfg.get("downstream") or {}).get("model")
    pool = CandidatePoolBuilder(cfg["candidate_pool"], downstream_model=model).build(
        args.issue, args.repo, budget, cfg["fitness"]
    )
    result = run_selector(args.method, pool, budget, cfg)
    rec = selector_record(args.instance_id, pool, result)
    text = json.dumps(rec, ensure_ascii=False, indent=2)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="HyCARO experiment CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("select", help="Run one selector on one local repository")
    p.add_argument("--repo", required=True)
    p.add_argument("--issue", required=True)
    p.add_argument("--instance-id", default="local-instance")
    p.add_argument("--config", default="configs/hycaro.yaml")
    p.add_argument("--method", choices=BUILTIN_METHODS, default="hycaro")
    p.add_argument("--budget", type=int)
    p.add_argument("--out")
    p.set_defaults(func=_select)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
