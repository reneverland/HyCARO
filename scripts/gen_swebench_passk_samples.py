#!/usr/bin/env python
"""Generate extra SWE-bench samples for pass@k without re-running selectors.

Reuses selected_units from the existing v3 jsonl (same context as pass@1).
Writes one prediction JSON per method×sample, resumable via --out jsonl.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

from repomaro.adapters.swebench import write_swebench_predictions
from repomaro.checkout import ensure_checkout
from repomaro.config import load_config
from repomaro.downstream import OpenAIPatchGenerator


def _index_lite(needed: set[str], split: str) -> dict[str, dict]:
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split=split)
    index: dict[str, dict] = {}
    for row in ds:
        iid = row["instance_id"]
        if iid not in needed:
            continue
        index[iid] = {
            "repo": row["repo"],
            "base_commit": row["base_commit"],
            "problem_statement": row["problem_statement"],
        }
        if len(index) >= len(needed):
            break
    return index


def _payload_context(repo_root: Path, rec: dict) -> str:
    parts: list[str] = []
    for unit in rec.get("selected_units") or []:
        rel = unit["path"]
        header = f"### {rel}"
        if unit.get("symbol"):
            header += f" :: {unit['symbol']}"
        start = unit.get("start_line")
        end = unit.get("end_line")
        if start is not None:
            header += f" (lines {start}-{end})"
        parts.append(header)
        path = repo_root / rel
        if not path.is_file():
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        if start is None:
            chunk = lines
            start = 1
        else:
            lo = max(int(start) - 1, 0)
            hi = int(end) if end is not None else lo + 1
            chunk = lines[lo:hi]
        numbered = "\n".join(f"{int(start) + i:6d}: {line}" for i, line in enumerate(chunk))
        parts.append(numbered)
    return "\n\n".join(parts)


def _load_jsonl(paths: list[Path]) -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    for path in paths:
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                out[(rec["instance_id"], rec["method"])] = rec
    return out


def _done_keys(path: Path) -> set[tuple[str, str, int]]:
    done: set[tuple[str, str, int]] = set()
    if not path.is_file():
        return done
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            done.add((rec["instance_id"], rec["method"], int(rec["sample_idx"])))
    return done


def _write_pred_files(sample_jsonl: Path, out_dir: Path, model: str) -> None:
    by: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    if not sample_jsonl.is_file():
        return
    with sample_jsonl.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            si = int(rec["sample_idx"])
            by[(rec["method"], si)].append({
                "instance_id": rec["instance_id"],
                "model_patch": rec.get("patch") or "",
                "model_name_or_path": f"{model}+{rec['method']}+s{si}",
            })
    for (method, si), rows in by.items():
        dest = out_dir / f"{method}_s{si}.json"
        write_swebench_predictions(rows, dest)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--jsonl",
        nargs="+",
        default=["results/swebench_lite_8k_ball40v3.jsonl"],
    )
    ap.add_argument("--out-dir", default="results/swebench_lite_8k_ball40v3_passk")
    ap.add_argument("--checkout-cache", default="cache/checkouts")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--sample-start", type=int, default=1)
    ap.add_argument("--sample-end", type=int, default=2, help="inclusive")
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.4,
        help="Sampling temperature for extra pass@k samples (s0 stays greedy)",
    )
    ap.add_argument(
        "--methods",
        default="",
        help="Comma-separated record method names to keep (empty = all)",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_jsonl = out_dir / "extra_samples.jsonl"
    selection = _load_jsonl([Path(p) for p in args.jsonl])
    keep = {x.strip() for x in args.methods.split(",") if x.strip()}
    if keep:
        selection = {k: v for k, v in selection.items() if k[1] in keep}
    needed = {iid for iid, _ in selection}
    index = _index_lite(needed, args.split)
    cfg = load_config(args.config)
    downstream_cfg = cfg.get("downstream") or {}
    model = str(downstream_cfg.get("model", "gpt-5.6-luna"))
    generator = OpenAIPatchGenerator(
        model=model,
        reasoning_effort=str(downstream_cfg.get("reasoning_effort", "medium")),
        max_output_tokens=int(downstream_cfg.get("max_output_tokens", 12000)),
        apply_retries=int(downstream_cfg.get("apply_retries", 1)),
        temperature=float(args.temperature),
        request_timeout=float(downstream_cfg.get("request_timeout", 900)),
        timeout_retries=int(downstream_cfg.get("timeout_retries", 2)),
    )
    cache = Path(args.checkout_cache)
    done = _done_keys(sample_jsonl)
    jobs: list[tuple[str, str, int, dict]] = []
    for (iid, method), rec in sorted(selection.items()):
        for si in range(int(args.sample_start), int(args.sample_end) + 1):
            if (iid, method, si) not in done:
                jobs.append((iid, method, si, rec))
    print(
        f"pass@k extra samples: {len(selection)} contexts, "
        f"todo={len(jobs)} skip={len(done)} s{args.sample_start}-{args.sample_end} "
        f"temperature={args.temperature}",
        flush=True,
    )

    for n, (iid, method, si, rec) in enumerate(jobs, 1):
        try:
            meta = index[iid]
            root = ensure_checkout(str(meta["repo"]), str(meta["base_commit"]), cache)
            context = _payload_context(root, rec)
            print(f"  [{n}/{len(jobs)}] {method} {iid} s{si} ctx={len(context)}", flush=True)
            patch, gen_meta = generator.generate(
                str(meta["problem_statement"]),
                context,
                repo_root=root,
            )
        except Exception as exc:
            print(f"  [{n}/{len(jobs)}] {method} {iid} s{si} generate failed: {exc}", flush=True)
            patch, gen_meta = "", {"error": str(exc), "apply": {"apply_ok": False}}
        apply_info = gen_meta.get("apply") or {}
        row = {
            "instance_id": iid,
            "method": method,
            "sample_idx": si,
            "patch": patch,
            "apply_ok": bool(apply_info.get("apply_ok")),
            "repaired": bool(gen_meta.get("repaired")),
            "error": gen_meta.get("error"),
            "usage": gen_meta.get("usage") or {},
            "temperature": gen_meta.get("temperature"),
            "selected_tokens": rec.get("selected_tokens"),
        }
        with sample_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        _write_pred_files(sample_jsonl, out_dir, model)
        print(
            f"    apply_ok={row['apply_ok']} repaired={row['repaired']}",
            flush=True,
        )

    _write_pred_files(sample_jsonl, out_dir, model)
    print(f"predictions under {out_dir}/<method>_s<k>.json", flush=True)


if __name__ == "__main__":
    main()
