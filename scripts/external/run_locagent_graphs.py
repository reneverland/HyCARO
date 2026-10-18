#!/usr/bin/env python
"""Build official LocAgent dependency graphs from our SWE-bench checkouts."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "external" / "LocAgent"))

os.environ.setdefault("MPLBACKEND", "Agg")
if "matplotlib" not in sys.modules:
    import types

    _mpl = types.ModuleType("matplotlib")
    _plt = types.ModuleType("matplotlib.pyplot")
    _lines = types.ModuleType("matplotlib.lines")
    _lines.Line2D = object
    sys.modules["matplotlib"] = _mpl
    sys.modules["matplotlib.pyplot"] = _plt
    sys.modules["matplotlib.lines"] = _lines


def _ensure_mpl_stub() -> None:
    if "matplotlib" in sys.modules:
        return
    import types

    mpl = types.ModuleType("matplotlib")
    plt = types.ModuleType("matplotlib.pyplot")
    lines = types.ModuleType("matplotlib.lines")
    lines.Line2D = object
    sys.modules["matplotlib"] = mpl
    sys.modules["matplotlib.pyplot"] = plt
    sys.modules["matplotlib.lines"] = lines


def _build_one(iid: str, repo_dir: str, out_file: str) -> tuple[str, str]:
    if Path(out_file).exists():
        return iid, "skip"
    _ensure_mpl_stub()
    from dependency_graph.build_graph import build_graph

    G = build_graph(repo_dir, global_import=True)
    tmp = Path(out_file + ".tmp")
    tmp.write_bytes(pickle.dumps(G))
    tmp.replace(out_file)
    return iid, "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", default="cache/locagent_graphs")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    rows = [
        json.loads(l)
        for l in Path(args.manifest).read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for row in rows:
        iid = row["instance_id"]
        dest = out_dir / f"{iid}.pkl"
        jobs.append((iid, row["root"], str(dest)))
    print(f"LocAgent graphs {len(jobs)} -> {out_dir}", flush=True)
    ok = skip = err = 0
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [ex.submit(_build_one, *job) for job in jobs]
        for n, fut in enumerate(as_completed(futs), 1):
            try:
                iid, status = fut.result()
            except Exception as exc:
                err += 1
                print(f"[{n}/{len(jobs)}] ERROR {exc}", flush=True)
                continue
            if status == "skip":
                skip += 1
            else:
                ok += 1
            print(f"[{n}/{len(jobs)}] {iid} {status}", flush=True)
    print(f"done ok={ok} skip={skip} err={err}", flush=True)


if __name__ == "__main__":
    main()
