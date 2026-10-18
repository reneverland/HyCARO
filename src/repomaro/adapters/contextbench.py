from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..checkout import ensure_checkout
from ..schema import SelectionInstance


def _first(row: dict[str, Any], *keys: str):
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return row[k]
    return None


def _github_repo(repo: Any, repo_url: Any) -> str:
    raw = str(repo or "").strip()
    if raw and "://" not in raw and raw.count("/") == 1:
        return raw.removesuffix(".git")
    url = str(repo_url or raw)
    for prefix in ("https://github.com/", "http://github.com/", "git@github.com:"):
        if url.startswith(prefix):
            return url[len(prefix):].removesuffix(".git").strip("/")
    raise KeyError(f"Cannot parse GitHub repo from repo={repo!r} repo_url={repo_url!r}")


def _parse_gold_context(raw: Any) -> list[dict[str, Any]]:
    if raw is None or raw == "":
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []
    if isinstance(raw, dict):
        raw = raw.get("spans") or raw.get("blocks") or raw.get("items") or []
    if not isinstance(raw, (list, tuple)):
        return []
    return [item for item in raw if isinstance(item, dict)]


def ensure_contextbench_gold_parquet(path: str | Path) -> Path:
    path = Path(path)
    if path.exists() and path.stat().st_size > 0:
        return path
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as e:
        raise RuntimeError("Install benchmark extras: `pip install -e '.[benchmarks]'`") from e
    ds = load_dataset("Contextbench/ContextBench", "contextbench_verified", split="train")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    ds.to_parquet(tmp)
    tmp.replace(path)
    return path


def load_contextbench_verified(
    checkout_cache: str | Path,
    limit: int | None = None,
    gold_parquet: str | Path | None = None,
    skip_log: str | Path | None = None,
    defer_checkout: bool = False,
) -> list[SelectionInstance]:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as e:
        raise RuntimeError("Install benchmark extras: `pip install -e '.[benchmarks]'`") from e
    ds = load_dataset("Contextbench/ContextBench", "contextbench_verified", split="train")
    if gold_parquet:
        parquet = Path(gold_parquet)
        if not parquet.exists() or parquet.stat().st_size == 0:
            parquet.parent.mkdir(parents=True, exist_ok=True)
            tmp = parquet.with_suffix(parquet.suffix + ".tmp")
            ds.to_parquet(tmp)
            tmp.replace(parquet)
            print(f"  wrote gold parquet {parquet}", flush=True)
    out: list[SelectionInstance] = []
    skipped = 0
    for row_obj in ds:
        row = dict(row_obj)
        instance_id = str(_first(row, "instance_id", "id"))
        query = str(_first(row, "problem_statement", "issue", "query", "prompt") or "")
        commit = _first(row, "base_commit", "commit")
        try:
            repo = _github_repo(_first(row, "repo", "repository"), _first(row, "repo_url"))
        except KeyError as exc:
            _log_skip(skip_log, instance_id, str(exc))
            skipped += 1
            continue
        if not commit:
            _log_skip(skip_log, instance_id, "missing base_commit")
            skipped += 1
            continue
        root = Path(checkout_cache)
        if not defer_checkout:
            print(f"  checkout {len(out) + skipped + 1}/500 {repo}@{str(commit)[:10]} {instance_id}", flush=True)
            try:
                root = ensure_checkout(repo, str(commit), checkout_cache)
            except Exception as exc:
                print(f"  checkout failed {instance_id} {repo}@{str(commit)[:10]}: {exc}", flush=True)
                _log_skip(skip_log, instance_id, f"checkout: {exc}")
                skipped += 1
                continue
        gold_paths, gold_spans = _gold_from_row(row)
        out.append(SelectionInstance(
            instance_id=instance_id,
            query=query,
            repo_root=root,
            gold_paths=gold_paths,
            gold_spans=gold_spans,
            metadata={
                "repo": repo,
                "base_commit": commit,
                "original_inst_id": row.get("original_inst_id"),
                "language": row.get("language"),
                "source": row.get("source"),
            },
        ))
        if limit and len(out) >= limit:
            break
    if skipped:
        print(f"  skipped {skipped} ContextBench rows", flush=True)
    return out


def _log_skip(path: str | Path | None, instance_id: str, reason: str) -> None:
    if not path:
        return
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"instance_id": instance_id, "reason": reason}, ensure_ascii=False) + "\n")


def _gold_from_row(row: dict[str, Any]) -> tuple[list[str], dict[str, list[tuple[int, int]]]]:
    items = _parse_gold_context(_first(row, "gold_context", "gold_ctx", "gold_spans", "gt_spans"))
    paths: list[str] = []
    spans: dict[str, list[tuple[int, int]]] = {}
    if items:
        for item in items:
            path = item.get("file") or item.get("path")
            if not path:
                continue
            path = str(path).replace("\\", "/").lstrip("./")
            if path.startswith("/testbed/"):
                path = path[len("/testbed/"):]
            paths.append(path)
            lo = item.get("start_line", item.get("start"))
            hi = item.get("end_line", item.get("end"))
            if lo is not None and hi is not None:
                spans.setdefault(path, []).append((int(lo), int(hi)))
        return sorted(set(paths)), spans

    raw_files = _first(row, "gold_files", "gt_files", "files", "oracle_files", "context_files")
    if isinstance(raw_files, str):
        paths = [raw_files]
    elif isinstance(raw_files, (list, tuple)):
        paths = [str(p) for p in raw_files if p]
    raw_spans = _first(row, "gold_spans", "gt_spans", "spans", "oracle_spans", "line_spans")
    if isinstance(raw_spans, dict):
        for path, span_items in raw_spans.items():
            out_spans: list[tuple[int, int]] = []
            for item in span_items or []:
                if isinstance(item, dict):
                    lo = item.get("start") or item.get("start_line")
                    hi = item.get("end") or item.get("end_line")
                    if lo is not None and hi is not None:
                        out_spans.append((int(lo), int(hi)))
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    out_spans.append((int(item[0]), int(item[1])))
            if out_spans:
                spans[str(path)] = out_spans
    return sorted(set(paths)), spans


def selection_to_contextbench_record(instance_id: str, selected_units, model_patch: str = "") -> dict[str, Any]:
    pred_files: list[str] = []
    spans: dict[str, list[dict[str, Any]]] = {}
    seen = set()
    for unit in selected_units:
        if unit.path not in seen:
            pred_files.append(unit.path); seen.add(unit.path)
        if unit.start_line is not None and unit.end_line is not None:
            spans.setdefault(unit.path, []).append({"type": "line", "start": int(unit.start_line), "end": int(unit.end_line)})
    step = {"files": pred_files, "spans": spans, "symbols": {}}
    return {
        "instance_id": instance_id,
        "traj_data": {"pred_steps": [step], "pred_files": pred_files, "pred_spans": spans},
        "model_patch": model_patch,
    }


def write_contextbench_predictions(records: list[dict[str, Any]], path: str | Path) -> Path:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path
