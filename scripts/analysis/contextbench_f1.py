#!/usr/bin/env python
"""Instance-mean File / Block / Line F1 from a ContextBench selection JSONL."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _f1(prec: float, rec: float) -> float:
    if prec + rec <= 0:
        return 0.0
    return 2.0 * prec * rec / (prec + rec)


def _prf(inter: int, n_pred: int, n_gold: int) -> tuple[float, float, float]:
    prec = (inter / n_pred) if n_pred else 1.0
    rec = (inter / n_gold) if n_gold else 1.0
    return prec, rec, _f1(prec, rec)


def _overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


def _line_set(spans: list[tuple[int, int]]) -> set[int]:
    out: set[int] = set()
    for lo, hi in spans:
        if hi < lo:
            lo, hi = hi, lo
        out.update(range(int(lo), int(hi) + 1))
    return out


def _gold_map(parquet: Path) -> dict[str, dict]:
    import pyarrow.dataset as ds

    table = ds.dataset(parquet, format="parquet").to_table(
        columns=["instance_id", "original_inst_id", "gold_context"]
    )
    out = {}
    for row in table.to_pylist():
        raw = row.get("gold_context")
        try:
            items = json.loads(raw) if isinstance(raw, str) else (raw or [])
        except json.JSONDecodeError:
            items = []
        files: set[str] = set()
        blocks: list[tuple[str, int, int]] = []
        lines: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for item in items or []:
            if not isinstance(item, dict):
                continue
            path = str(item.get("file") or item.get("path") or "")
            if not path:
                continue
            files.add(path)
            lo = item.get("start_line", item.get("start"))
            hi = item.get("end_line", item.get("end"))
            if lo is None or hi is None:
                continue
            lo, hi = int(lo), int(hi)
            blocks.append((path, lo, hi))
            lines[path].append((lo, hi))
        rec = {"files": files, "blocks": blocks, "lines": dict(lines)}
        for key in (row.get("instance_id"), row.get("original_inst_id")):
            if key:
                out[str(key)] = rec
    return out


def score_file(path: Path, gold: dict[str, dict], method: str | None = None) -> dict:
    by_inst: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if method and rec.get("method") != method:
                continue
            by_inst[rec["instance_id"]] = rec
    file_f1s, block_f1s, line_f1s, toks, times = [], [], [], [], []
    missing = 0
    for iid, rec in by_inst.items():
        g = gold.get(iid) or gold.get(str(rec.get("original_inst_id") or ""))
        if not g:
            missing += 1
            continue
        units = rec.get("selected_units") or []
        pred_files = {u["path"] for u in units if u.get("path")}
        pred_blocks = [
            (u["path"], int(u["start_line"]), int(u["end_line"]))
            for u in units
            if u.get("path") is not None and u.get("start_line") is not None and u.get("end_line") is not None
        ]
        pred_lines: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for path_, lo, hi in pred_blocks:
            pred_lines[path_].append((lo, hi))

        inter_f = len(pred_files & g["files"])
        file_f1s.append(_prf(inter_f, len(pred_files), len(g["files"]))[2])

        gold_hit = sum(
            1
            for gp, glo, ghi in g["blocks"]
            if any(pp == gp and _overlap((plo, phi), (glo, ghi)) for pp, plo, phi in pred_blocks)
        )
        pred_hit = sum(
            1
            for pp, plo, phi in pred_blocks
            if any(gp == pp and _overlap((plo, phi), (glo, ghi)) for gp, glo, ghi in g["blocks"])
        )
        block_f1s.append(_f1(
            (pred_hit / len(pred_blocks)) if pred_blocks else 1.0,
            (gold_hit / len(g["blocks"])) if g["blocks"] else 1.0,
        ))

        gold_line = set()
        pred_line = set()
        for pth, spans in g["lines"].items():
            gold_line |= {(pth, ln) for ln in _line_set(spans)}
        for pth, spans in pred_lines.items():
            pred_line |= {(pth, ln) for ln in _line_set(spans)}
        inter_l = len(gold_line & pred_line)
        line_f1s.append(_prf(inter_l, len(pred_line), len(gold_line))[2])
        toks.append(int(rec.get("selected_tokens") or 0))
        if rec.get("selector_time_seconds") is not None:
            times.append(float(rec["selector_time_seconds"]))

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "n": len(file_f1s),
        "missing_gold": missing,
        "file_f1": mean(file_f1s),
        "block_f1": mean(block_f1s),
        "line_f1": mean(line_f1s),
        "ctx_tok": mean(toks),
        "time_s": mean(times),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--gold", default="cache/contextbench_verified.parquet")
    ap.add_argument("--method")
    args = ap.parse_args()
    gold = _gold_map(Path(args.gold))
    stats = score_file(Path(args.jsonl), gold, args.method)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
