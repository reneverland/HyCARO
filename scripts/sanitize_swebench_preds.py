#!/usr/bin/env python
"""Rewrite existing SWE-bench prediction JSON so patches pass `git apply --check`."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset

from repomaro.checkout import ensure_checkout
from repomaro.patch_format import make_applyable


def _index_lite(split: str, needed: set[str]) -> dict[str, dict]:
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split=split)
    index: dict[str, dict] = {}
    for row in ds:
        iid = row["instance_id"]
        if iid not in needed:
            continue
        index[iid] = {"repo": row["repo"], "base_commit": row["base_commit"]}
        if len(index) >= len(needed):
            break
    return index


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pred", nargs="+", help="Official harness prediction JSON files")
    ap.add_argument("--checkout-cache", default="cache/checkouts")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-dir", help="Write rewritten files here (default: overwrite)")
    ap.add_argument(
        "--status-out",
        default="",
        help="Optional JSONL of per-row apply status (instance_id, method, apply_ok, rebuilt, error)",
    )
    args = ap.parse_args()

    pred_files = [Path(p) for p in args.pred]
    needed: set[str] = set()
    loaded: dict[Path, list] = {}
    for src in pred_files:
        rows = json.loads(src.read_text(encoding="utf-8"))
        loaded[src] = rows
        needed.update(row["instance_id"] for row in rows)
    index = _index_lite(args.split, needed)
    cache = Path(args.checkout_cache)
    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    status_path = Path(args.status_out) if args.status_out else (out_dir / "apply_status.jsonl" if out_dir else None)
    if status_path:
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text("", encoding="utf-8")

    for src, rows in loaded.items():
        ok = rebuilt = failed = 0
        method = src.stem
        for row in rows:
            iid = row["instance_id"]
            if iid not in index:
                failed += 1
                print(f"  FAIL {src.name} {iid}: not in SWE-bench Lite {args.split}", flush=True)
                _append_status(status_path, iid, method, False, False, "missing instance")
                continue
            meta = index[iid]
            root = ensure_checkout(str(meta["repo"]), str(meta["base_commit"]), cache)
            patch, info = make_applyable(row.get("model_patch") or "", root)
            row["model_patch"] = patch
            apply_ok = bool(info.get("apply_ok"))
            was_rebuilt = bool(info.get("rebuilt"))
            if was_rebuilt:
                rebuilt += 1
            if apply_ok:
                ok += 1
            else:
                failed += 1
                print(f"  FAIL {src.name} {iid}: {info.get('apply_error', '')[:180]}", flush=True)
            _append_status(
                status_path,
                iid,
                method,
                apply_ok,
                was_rebuilt,
                "" if apply_ok else (info.get("apply_error") or ""),
            )
        dest = (out_dir / src.name) if out_dir else src
        dest.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"{src} -> {dest}  apply_ok={ok}/{len(rows)} rebuilt={rebuilt} fail={failed}", flush=True)


def _append_status(path: Path | None, iid: str, method: str, apply_ok: bool, rebuilt: bool, error: str) -> None:
    if path is None:
        return
    rec = {
        "instance_id": iid,
        "method": method,
        "apply_ok": apply_ok,
        "rebuilt": rebuilt,
        "error": (error or "")[:500],
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
