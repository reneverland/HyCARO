#!/usr/bin/env python
"""Official LongCodeZip coarse ranking over our cached candidate pools.

Runs in the main venv with PYTHONPATH=external/LongCodeZip. Only the ranking
crosses the boundary; packing and generation stay in run_swebench.py.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "external" / "LongCodeZip"))
sys.path.insert(0, str(ROOT / "src"))


class RankOnlyLongCodeZip:
    """Official compressor without the unused entropy-chunking 0.5B load."""

    def __init__(self, model_name: str, device_map: str = "cuda"):
        from longcodezip import LongCodeZip

        self._impl = LongCodeZip.__new__(LongCodeZip)
        self._impl.model_name = model_name
        self._impl.device = device_map
        self._impl.model_config = {}
        self._impl.load_model(model_name, device_map, {})
        self._impl.entropy_chunking = None
        self._impl.max_position_embeddings = getattr(
            self._impl.model.config, "max_position_embeddings", 4096
        )

    def rank_units(self, texts: list[str], query: str, target_token: int):
        out = self._impl.control_context_budget(
            texts,
            target_token=target_token,
            question=query,
            reorder_context="original",
            condition_in_question="prefix",
            context_budget="+100",
            dynamic_context_compression_ratio=0.2,
        )
        if len(out) == 4:
            _selected, _used, _ratios, demonstrations_sort = out
        else:
            demonstrations_sort = [(i, 0.0) for i in range(len(texts))]
        return demonstrations_sort


def _load_done(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    for line in path.open(encoding="utf-8"):
        if line.strip():
            done.add(json.loads(line).get("instance_id") or "")
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--pool-cache", default="cache/pools_v3")
    ap.add_argument("--budget", type=int, default=8192)
    ap.add_argument("--out", default="cache/longcodezip_rankings_lite300.jsonl")
    ap.add_argument(
        "--model",
        default=os.environ.get("LCZ_MODEL", "Qwen/Qwen2.5-Coder-0.5B-Instruct"),
    )
    args = ap.parse_args()

    rows = [
        json.loads(l)
        for l in Path(args.manifest).read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = _load_done(out)
    print(
        f"LongCodeZip ranking {len(rows)} rows, {len(done)} already written, "
        f"model={args.model} -> {out}",
        flush=True,
    )
    compressor = RankOnlyLongCodeZip(args.model)
    cache = Path(args.pool_cache)
    with out.open("a", encoding="utf-8") as fh:
        for i, row in enumerate(rows, 1):
            iid = row["instance_id"]
            if iid in done:
                continue
            pkl = cache / f"{iid}__{args.budget}.pkl"
            if not pkl.exists():
                print(f"[{i}/{len(rows)}] {iid} missing pool {pkl}", flush=True)
                continue
            pool = pickle.loads(pkl.read_bytes())
            texts = [u.text or "" for u in pool.units]
            print(f"[{i}/{len(rows)}] {iid} units={len(texts)}", flush=True)
            scored = compressor.rank_units(texts, row.get("query") or row.get("issue") or "", args.budget)
            ranking = []
            for rank, (idx, score) in enumerate(scored):
                if idx < 0 or idx >= len(pool.units):
                    continue
                u = pool.units[idx]
                ranking.append({
                    "path": u.path,
                    "line": int(u.start_line or 0),
                    "rank": rank,
                    "name": u.symbol or "",
                    "score": float(score),
                })
            fh.write(json.dumps({
                "instance_id": iid,
                "root": str(Path(row["root"]).resolve()),
                "query": row.get("query") or row.get("issue") or "",
                "model": args.model,
                "rank_only": True,
                "ranking": ranking,
            }, ensure_ascii=False) + "\n")
            fh.flush()
            done.add(iid)
    print(f"done {len(done)}", flush=True)


if __name__ == "__main__":
    main()
