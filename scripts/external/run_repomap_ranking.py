#!/usr/bin/env python
"""Precompute aider RepoMap tag rankings.

Pass a manifest JSONL ({instance_id, root, query}) produced by this repo's
main venv. Ranking itself must run under external/venvs/repomap.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    "node_modules",
    ".aider.tags.cache.v3",
    ".aider.tags.cache.v4",
}


def _list_files(root: Path) -> list[str]:
    out: list[str] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if any(part in _SKIP_DIRS for part in p.parts):
            continue
        if p.suffix.lower() not in {
            ".py", ".pyi", ".c", ".h", ".cc", ".cpp", ".js", ".ts", ".java", ".go", ".rs",
        }:
            continue
        out.append(str(p))
    return sorted(out)


class _IO:
    def tool_output(self, *args, **kwargs):
        return None

    def tool_warning(self, *args, **kwargs):
        return None

    def read_text(self, fname):
        try:
            return Path(fname).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""


def _rank_one(root: Path, query: str) -> list[dict]:
    from aider.repomap import RepoMap

    class _Tok:
        def token_count(self, text):
            return max(1, len(text) // 4)

    rm = RepoMap(map_tokens=8192, root=str(root), main_model=_Tok(), io=_IO(), verbose=False)
    files = _list_files(root)
    idents = set(_IDENT.findall(query or ""))
    tags = rm.get_ranked_tags([], files, set(), idents) or []
    ranking = []
    for i, tag in enumerate(tags):
        if hasattr(tag, "rel_fname"):
            ranking.append({
                "path": tag.rel_fname,
                "line": int(tag.line) + 1,
                "rank": i,
                "name": getattr(tag, "name", ""),
            })
        elif tag:
            ranking.append({"path": str(tag[0]), "line": 1, "rank": i})
    return ranking


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="JSONL with instance_id, root, query")
    ap.add_argument("--out", default="cache/repomap_rankings_lite300.jsonl")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.manifest).read_text(encoding="utf-8").splitlines() if l.strip()]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.open(encoding="utf-8"):
            if line.strip():
                done.add(json.loads(line).get("instance_id") or "")
    print(f"RepoMap ranking {len(rows)} rows, {len(done)} already written -> {out}", flush=True)
    with out.open("a", encoding="utf-8") as fh:
        for i, row in enumerate(rows, 1):
            iid = row["instance_id"]
            if iid in done:
                continue
            root = Path(row["root"]).resolve()
            print(f"[{i}/{len(rows)}] {iid}", flush=True)
            ranking = _rank_one(root, row.get("query") or "")
            fh.write(json.dumps({
                "instance_id": iid,
                "root": str(root),
                "query": row.get("query") or "",
                "ranking": ranking,
            }, ensure_ascii=False) + "\n")
            fh.flush()
            done.add(iid)
    print(f"done {len(done)}", flush=True)


if __name__ == "__main__":
    main()
