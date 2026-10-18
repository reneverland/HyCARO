"""Method-agnostic patch triage: classify why a patch fails and standardize it.

Every method (Dense, BM25, Node-RepoMARO, RepoMARO, ...) goes through the exact
same ladder, so `git apply` success rate stays a property of the patch text and
not of who produced it. Anything the ladder cannot fix is still reported as a
failure and still counts as unresolved downstream.

The ladder only ever emits a patch that plain `git apply` accepts: permissive
flags (`--recount`, `--ignore-whitespace`, reduced context) are used to *land*
the edit in a scratch checkout, and the canonical patch is then re-derived with
`git diff`.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .patch_format import (
    FilePatch,
    check_git_apply,
    extract_patch,
    parse_unified_diff,
    rebuild_patch,
    render_unified_diff,
)

# Ordered: the first rung that lands the edit wins.
_PERMISSIVE_FLAGS: list[tuple[str, list[str]]] = [
    ("plain", []),
    ("recount", ["--recount"]),
    ("whitespace", ["--ignore-whitespace"]),
    ("recount+whitespace", ["--recount", "--ignore-whitespace"]),
    ("fuzz", ["--recount", "--ignore-whitespace", "-C1"]),
    ("unidiff-zero", ["--recount", "--ignore-whitespace", "--unidiff-zero"]),
]

CATEGORIES = (
    "ok",
    "empty",
    "no_diff",
    "bad_path",
    "truncated",
    "corrupt_hunk",
    "context_mismatch",
    "other",
)


@dataclass
class RepairResult:
    patch: str
    ok: bool
    stage: str = ""
    category: str = "empty"
    fixes: list[str] = field(default_factory=list)
    error: str = ""


def repair_patch(text: str, repo_root: str | Path) -> RepairResult:
    """Normalize `text` into a patch that plain `git apply` accepts, if possible."""
    root = Path(repo_root)
    raw = extract_patch(text)
    if not raw.strip():
        return RepairResult(patch="", ok=False, category="empty", error="empty patch")

    files = parse_unified_diff(raw)
    if not files:
        return RepairResult(patch=raw, ok=False, category="no_diff", error="no unified diff found")

    fixes: list[str] = []

    # Variant A: fences stripped and hunk counts recomputed.
    variants: list[tuple[str, str]] = [("sanitized", render_unified_diff(files))]

    # Variant B: repository-relative paths repaired (extra prefix, wrong root, ...).
    path_fixed, path_changed = _fix_paths(files, root)
    if path_changed:
        fixes.append("path_prefix")
        variants.append(("path_fixed", render_unified_diff(path_fixed)))

    # Variant C: a truncated trailing hunk dropped rather than failing the file.
    trimmed, trunc = _drop_truncated_tail(path_fixed if path_changed else files)
    if trunc:
        fixes.append("truncation")
        variants.append(("truncated_trimmed", render_unified_diff(trimmed)))

    # Variant D: the model text exactly as emitted, in case our rewrite hurt.
    variants.append(("raw", raw))

    last_error = ""
    for name, candidate in variants:
        if not candidate.strip():
            continue
        for flag_name, flags in _PERMISSIVE_FLAGS:
            ok, err = _check(root, candidate, flags)
            if not ok:
                last_error = last_error or err
                continue
            if flag_name == "plain" and name != "raw":
                return RepairResult(
                    patch=_eol(candidate), ok=True, stage=name, category="ok", fixes=fixes
                )
            canonical = _canonicalize(root, candidate, flags)
            if canonical:
                stage = name if flag_name == "plain" else f"{name}+{flag_name}"
                if flag_name != "plain":
                    fixes.append(flag_name)
                return RepairResult(
                    patch=_eol(canonical), ok=True, stage=stage, category="ok", fixes=fixes
                )
            if flag_name == "plain":
                return RepairResult(
                    patch=_eol(candidate), ok=True, stage=name, category="ok", fixes=fixes
                )

    # Variant E: replay the hunks by content match and re-diff from scratch.
    rebuilt = rebuild_patch(raw, root)
    if rebuilt:
        ok, err = _check(root, rebuilt, [])
        if ok:
            fixes.append("rebuilt")
            return RepairResult(patch=_eol(rebuilt), ok=True, stage="rebuilt", category="ok", fixes=fixes)
        last_error = last_error or err

    fallback = render_unified_diff(files) or raw
    return RepairResult(
        patch=_eol(fallback),
        ok=False,
        stage="",
        category=classify_error(last_error, files, root),
        fixes=fixes,
        error=last_error[:500],
    )


def classify_error(error: str, files: list[FilePatch], root: Path) -> str:
    text = (error or "").lower()
    if not text:
        return "other"
    if "no valid patches in input" in text:
        return "no_diff"
    if "no such file or directory" in text or "does not exist in index" in text:
        return "bad_path"
    if any(f.path and not (root / f.path).exists() and not f.is_new for f in files):
        return "bad_path"
    if "unexpected end of file" in text or "truncated" in text:
        return "truncated"
    if "corrupt patch" in text or "expected" in text:
        return "corrupt_hunk"
    if "does not apply" in text or "while searching for" in text or "already exists" in text:
        return "context_mismatch"
    return "other"


def _fix_paths(files: list[FilePatch], root: Path) -> tuple[list[FilePatch], bool]:
    """Resolve paths that carry an extra prefix or are missing repository depth."""
    changed = False
    out: list[FilePatch] = []
    tracked: dict[str, list[str]] | None = None
    for file in files:
        path = file.path
        if not path or file.is_new or (root / path).is_file():
            out.append(file)
            continue
        resolved = None
        parts = path.split("/")
        for start in range(1, len(parts)):
            cand = "/".join(parts[start:])
            if (root / cand).is_file():
                resolved = cand
                break
        if resolved is None:
            if tracked is None:
                tracked = _tracked_by_basename(root)
            hits = tracked.get(parts[-1], [])
            suffix_hits = [h for h in hits if h.endswith("/".join(parts[-2:]))] or hits
            if len(suffix_hits) == 1:
                resolved = suffix_hits[0]
        if resolved is None or resolved == path:
            out.append(file)
            continue
        changed = True
        out.append(FilePatch(path=resolved, hunks=file.hunks, is_new=file.is_new, is_delete=file.is_delete))
    return out, changed


def _tracked_by_basename(root: Path) -> dict[str, list[str]]:
    proc = subprocess.run(
        ["git", "ls-files"], cwd=str(root), text=True, capture_output=True
    )
    index: dict[str, list[str]] = {}
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if line:
            index.setdefault(line.rsplit("/", 1)[-1], []).append(line)
    return index


def _drop_truncated_tail(files: list[FilePatch]) -> tuple[list[FilePatch], bool]:
    """Drop a trailing hunk that the model cut off mid-edit."""
    if not files:
        return files, False
    last = files[-1]
    if len(last.hunks) < 2 and len(files) < 2:
        return files, False
    if not last.hunks:
        return files, False
    tail = last.hunks[-1]
    if not _looks_truncated(tail.lines):
        return files, False
    kept_hunks = last.hunks[:-1]
    out = list(files[:-1])
    if kept_hunks:
        out.append(FilePatch(path=last.path, hunks=kept_hunks, is_new=last.is_new, is_delete=last.is_delete))
    return (out, True) if out else (files, False)


def _looks_truncated(lines: list[str]) -> bool:
    """A hunk whose body is pure additions with no closing context is suspect."""
    if not lines:
        return True
    if not any(line.startswith((" ", "-")) for line in lines):
        return True
    return not lines[-1].startswith((" ", "-"))


def _check(root: Path, patch: str, flags: list[str]) -> tuple[bool, str]:
    if not flags:
        return check_git_apply(root, patch)
    proc = subprocess.run(
        ["git", "apply", "--check", "--verbose", *flags, "-"],
        cwd=str(root),
        input=_eol(patch),
        text=True,
        capture_output=True,
    )
    if proc.returncode == 0:
        return True, ""
    return False, (proc.stderr or proc.stdout or "").strip()


def _canonicalize(root: Path, patch: str, flags: list[str]) -> str:
    """Land the edit with permissive flags, then re-derive a strict `git diff`."""
    _reset(root)
    proc = subprocess.run(
        ["git", "apply", *flags, "-"],
        cwd=str(root),
        input=_eol(patch),
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        _reset(root)
        return ""
    subprocess.run(["git", "add", "-A"], cwd=str(root), capture_output=True, text=True)
    diff = subprocess.run(
        ["git", "diff", "--cached", "--binary", "--no-color"],
        cwd=str(root),
        capture_output=True,
        text=True,
    )
    out = diff.stdout or ""
    _reset(root)
    return out if out.strip() else ""


def _reset(root: Path) -> None:
    subprocess.run(["git", "reset", "-q"], cwd=str(root), capture_output=True, text=True)
    subprocess.run(["git", "checkout", "-q", "--", "."], cwd=str(root), capture_output=True, text=True)
    subprocess.run(["git", "clean", "-qfd"], cwd=str(root), capture_output=True, text=True)


def _eol(text: str) -> str:
    return text if text.endswith("\n") else text + "\n"
