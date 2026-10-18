"""Make downstream patches acceptable to `git apply` / the SWE-bench harness.

Luna often emits almost-correct unified diffs (wrong hunk counts, fake index
lines, apply_patch fences). Those fail official evaluation before any test runs.
This module is deterministic cleanup plus an optional rebuild against a checkout.
"""

from __future__ import annotations

import difflib
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

_DIFF_FENCE = re.compile(r"```(?:diff|patch)?\s*(.*?)```", re.S | re.I)
_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$")
_INDEX_RE = re.compile(r"^index [0-9a-fA-F]+\.\.[0-9a-fA-F]+(?: \d+)?\s*$")
_JUNK_LINE_RE = re.compile(
    r"^\*\*\* (?:Begin Patch|End Patch|Update File:|Add File:|Delete File:)"
)
_NO_EOL = "\\ No newline at end of file"


@dataclass
class Hunk:
    old_start: int
    new_start: int
    lines: list[str]  # raw hunk body, including prefixes
    header_suffix: str = ""

    @property
    def old_lines(self) -> list[str]:
        out: list[str] = []
        for line in self.lines:
            if line.startswith("\\"):
                continue
            if line.startswith("+"):
                continue
            if line.startswith(("-", " ")):
                out.append(line[1:])
            else:
                out.append(line)
        return out

    @property
    def new_lines(self) -> list[str]:
        out: list[str] = []
        for line in self.lines:
            if line.startswith("\\"):
                continue
            if line.startswith("-"):
                continue
            if line.startswith(("+", " ")):
                out.append(line[1:])
            else:
                out.append(line)
        return out

    @property
    def removed_lines(self) -> list[str]:
        return [line[1:] for line in self.lines if line.startswith("-")]

    @property
    def added_lines(self) -> list[str]:
        return [line[1:] for line in self.lines if line.startswith("+")]


@dataclass
class FilePatch:
    path: str
    hunks: list[Hunk] = field(default_factory=list)
    is_new: bool = False
    is_delete: bool = False


def extract_patch(text: str) -> str:
    """Pull a patch out of model prose / markdown fences."""
    text = (text or "").strip()
    if not text:
        return ""
    fences = list(_DIFF_FENCE.finditer(text))
    if fences:
        chosen = None
        for match in fences:
            body = match.group(1)
            if _looks_like_patch(body):
                chosen = body
                break
        text = (chosen or fences[0].group(1)).strip()
    starts = []
    for marker in ("diff --git ", "*** Begin Patch", "*** Update File:", "--- a/"):
        idx = text.find(marker)
        if idx >= 0:
            starts.append(idx)
    if starts:
        text = text[min(starts) :]
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.strip("\n") + ("\n" if text.strip() else "")


def sanitize_patch(text: str) -> str:
    """Deterministic cleanup that does not need the repository."""
    text = extract_patch(text)
    if not text:
        return ""
    if _looks_like_apply_patch(text) and "diff --git " not in text:
        text = _apply_patch_to_unified(text)
    files = parse_unified_diff(text)
    if not files:
        return text if text.endswith("\n") else text + "\n"
    rendered = render_unified_diff(files)
    return rendered or (text if text.endswith("\n") else text + "\n")


def make_applyable(text: str, repo_root: str | Path | None = None) -> tuple[str, dict]:
    """Return the first variant of `text` that `git apply --check` accepts.

    Candidates are tried in order: rebuilt against the checkout, deterministically
    sanitized, then the model output as-is. A patch that already applies is never
    replaced by a rewrite.
    """
    info = {"sanitized": False, "rebuilt": False, "apply_ok": False, "apply_error": ""}
    raw = extract_patch(text)
    sanitized = sanitize_patch(raw)
    fallback = sanitized or raw
    if not fallback.strip():
        info["apply_error"] = "empty patch"
        return "", info
    if repo_root is None:
        info["sanitized"] = bool(sanitized) and sanitized != raw
        return _ensure_eol(fallback), info

    root = Path(repo_root)
    # Prefer a working sanitized patch over a rebuild. Rebuild can drop files
    # when only some hunks match, which would still pass `git apply --check`.
    candidates: list[tuple[str, str]] = []
    if sanitized.strip():
        candidates.append((sanitized, "sanitized"))
    rebuilt = rebuild_patch(sanitized or raw, root)
    if rebuilt:
        candidates.append((rebuilt, "rebuilt"))
    if raw.strip() and raw.strip() != sanitized.strip():
        candidates.append((raw, "raw"))

    last_error = ""
    for candidate, kind in candidates:
        ok, err = check_git_apply(root, candidate)
        if ok:
            info["apply_ok"] = True
            info["rebuilt"] = kind == "rebuilt"
            info["sanitized"] = kind != "raw"
            return _ensure_eol(candidate), info
        last_error = last_error or err
    info["apply_error"] = last_error
    info["sanitized"] = bool(sanitized) and sanitized != raw
    return _ensure_eol(fallback), info


def check_git_apply(repo_root: str | Path, patch: str) -> tuple[bool, str]:
    if not patch.strip():
        return False, "empty patch"
    proc = subprocess.run(
        ["git", "apply", "--check", "--verbose", "-"],
        cwd=str(repo_root),
        input=_ensure_eol(patch),
        text=True,
        capture_output=True,
    )
    if proc.returncode == 0:
        return True, ""
    return False, (proc.stderr or proc.stdout or "").strip()


def parse_unified_diff(text: str) -> list[FilePatch]:
    files: list[FilePatch] = []
    current: FilePatch | None = None
    hunk: Hunk | None = None
    pending_path: str | None = None
    is_new = False
    is_delete = False

    def flush_hunk() -> None:
        nonlocal hunk
        if current is not None and hunk is not None and hunk.lines:
            current.hunks.append(hunk)
        hunk = None

    def close_file() -> None:
        nonlocal current
        flush_hunk()
        if current is not None and (current.hunks or current.is_new or current.is_delete):
            files.append(current)
        current = None

    for raw in text.split("\n"):
        line = raw.rstrip("\r")
        if line.startswith("diff --git "):
            close_file()
            parts = line.split()
            pending_path = _clean_path(parts[-1]) if len(parts) >= 4 else None
            is_new = False
            is_delete = False
            continue
        if line.startswith("new file mode"):
            is_new = True
            continue
        if line.startswith("deleted file mode"):
            is_delete = True
            continue
        if (
            _INDEX_RE.match(line)
            or line.startswith("old mode")
            or line.startswith("new mode")
            or line.startswith("similarity index")
            or line.startswith("rename from")
            or line.startswith("rename to")
        ):
            continue
        if _JUNK_LINE_RE.match(line):
            continue

        old_header = _match_old_header(line, in_hunk=hunk is not None)
        if old_header is not None:
            close_file()
            if old_header == "/dev/null":
                is_new = True
            else:
                pending_path = _clean_path(old_header)
            continue
        new_header = _match_new_header(line, in_hunk=hunk is not None)
        if new_header is not None:
            flush_hunk()
            if new_header == "/dev/null":
                is_delete = True
                path = pending_path or ""
            else:
                path = _clean_path(new_header)
            if current is not None and (current.hunks or current.is_new or current.is_delete):
                files.append(current)
            current = FilePatch(
                path=path or pending_path or "",
                is_new=is_new,
                is_delete=is_delete,
            )
            continue

        match = _HUNK_RE.match(line)
        if match or line.startswith("@@"):
            flush_hunk()
            if current is None:
                current = FilePatch(path=pending_path or "", is_new=is_new, is_delete=is_delete)
            if match:
                hunk = Hunk(
                    old_start=int(match.group(1)),
                    new_start=int(match.group(3)),
                    lines=[],
                    header_suffix=match.group(5) or "",
                )
            else:
                # apply_patch / Luna often emits a bare `@@` after a dummy header.
                hunk = Hunk(old_start=1, new_start=1, lines=[], header_suffix="")
            continue
        if line.startswith("*** "):
            continue
        if hunk is not None:
            if line.startswith(("+", "-", " ", "\\")):
                hunk.lines.append(line)
            elif line == "":
                # A blank line inside a hunk is a context line for an empty source line.
                hunk.lines.append(" ")
            else:
                # The model dropped the leading space on a context line.
                hunk.lines.append(" " + line)
    close_file()
    # Trailing blank context lines are usually the model's paragraph break, not code.
    for file in files:
        for h in file.hunks:
            while h.lines and h.lines[-1] == " ":
                h.lines.pop()
        file.hunks = [h for h in file.hunks if h.lines]
    return [f for f in files if f.hunks or f.is_new or f.is_delete]


def render_unified_diff(files: list[FilePatch]) -> str:
    """Render with recomputed hunk counts. Never strips trailing hunk lines."""
    out: list[str] = []
    for file in files:
        if not file.hunks and not file.is_new and not file.is_delete:
            continue
        path = file.path
        if not path:
            continue
        out.append(f"diff --git a/{path} b/{path}\n")
        if file.is_new:
            out.append("new file mode 100644\n")
            out.append("--- /dev/null\n")
            out.append(f"+++ b/{path}\n")
        elif file.is_delete:
            out.append("deleted file mode 100644\n")
            out.append(f"--- a/{path}\n")
            out.append("+++ /dev/null\n")
        else:
            out.append(f"--- a/{path}\n")
            out.append(f"+++ b/{path}\n")
        for hunk in file.hunks:
            old_n, new_n = _hunk_counts(hunk.lines)
            if old_n == 0 and new_n == 0:
                continue
            suffix = hunk.header_suffix.strip()
            header = f"@@ -{hunk.old_start},{old_n} +{hunk.new_start},{new_n} @@"
            if suffix:
                header += " " + suffix
            out.append(header + "\n")
            for line in hunk.lines:
                out.append(line + "\n")
    return "".join(out)


def rebuild_patch(text: str, repo_root: str | Path) -> str | None:
    """Replay hunks onto the checkout and re-diff, so hunk headers are exact."""
    root = Path(repo_root)
    files = parse_unified_diff(text)
    if not files and _looks_like_apply_patch(text):
        files = parse_unified_diff(_apply_patch_to_unified(text))
    if not files:
        return None
    chunks: list[str] = []
    for file in files:
        rendered = _rebuild_file(root, file)
        if not rendered:
            return None
        chunks.append(rendered)
    if not chunks:
        return None
    return "".join(chunks)


def _rebuild_file(root: Path, file: FilePatch) -> str:
    if not file.path:
        return ""
    path = root / file.path
    if file.is_new or (not path.is_file() and _hunks_are_additions(file.hunks)):
        new_lines: list[str] = []
        for hunk in file.hunks:
            new_lines.extend(hunk.new_lines)
        if not new_lines:
            return ""
        return _diff_text(file.path, [], new_lines, is_new=True)
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    original = text.split("\n")
    trailing_newline = bool(original) and original[-1] == ""
    if trailing_newline:
        original.pop()

    current = list(original)
    applied_any = False
    for hunk in file.hunks:
        applied = _apply_hunk(current, hunk)
        if applied is None:
            continue
        current = applied
        applied_any = True
    if not applied_any or current == original:
        return ""
    if file.is_delete and not current:
        return _diff_text(file.path, original, [], is_delete=True, old_eol=trailing_newline)
    return _diff_text(
        file.path,
        original,
        current,
        old_eol=trailing_newline,
        new_eol=trailing_newline,
    )


def _diff_text(
    path: str,
    old_lines: list[str],
    new_lines: list[str],
    is_new: bool = False,
    is_delete: bool = False,
    old_eol: bool = True,
    new_eol: bool = True,
) -> str:
    """Emit one `git diff`-style file patch. difflib guarantees correct counts."""
    old = _with_eol(old_lines, old_eol)
    new = _with_eol(new_lines, new_eol)
    body = list(difflib.unified_diff(old, new, n=3))
    if not body:
        return ""
    body = body[2:]  # drop difflib's own "--- " / "+++ " header lines
    if not body:
        return ""
    head = [f"diff --git a/{path} b/{path}\n"]
    if is_new:
        head += ["new file mode 100644\n", "--- /dev/null\n", f"+++ b/{path}\n"]
    elif is_delete:
        head += ["deleted file mode 100644\n", f"--- a/{path}\n", "+++ /dev/null\n"]
    else:
        head += [f"--- a/{path}\n", f"+++ b/{path}\n"]
    fixed: list[str] = []
    for line in body:
        if line.endswith("\n"):
            fixed.append(line)
        else:
            fixed.append(line + "\n")
            fixed.append(_NO_EOL + "\n")
    return "".join(head + fixed)


def _with_eol(lines: list[str], trailing_newline: bool) -> list[str]:
    out = [line + "\n" for line in lines]
    if out and not trailing_newline:
        out[-1] = out[-1][:-1]
    return out


def _apply_hunk(lines: list[str], hunk: Hunk) -> list[str] | None:
    old = hunk.old_lines
    new = hunk.new_lines
    hint = max(hunk.old_start - 1, 0)
    if not old:
        idx = min(hint, len(lines))
        return lines[:idx] + new + lines[idx:]
    idx = _find_block(lines, old, hint=hint)
    if idx is not None:
        return lines[:idx] + new + lines[idx + len(old) :]
    # Context lines are often invented; fall back to the removed-line block.
    removed = hunk.removed_lines
    added = hunk.added_lines
    if not removed:
        return None
    idx = _find_block(lines, removed, hint=hint)
    if idx is not None:
        return lines[:idx] + added + lines[idx + len(removed) :]
    # Last resort: leading/trailing context may be wrong, so trim it away.
    trimmed = _trim_context(hunk)
    if trimmed is not None and trimmed != old:
        idx = _find_block(lines, trimmed, hint=hint)
        if idx is not None:
            new_trimmed = _trim_new(hunk)
            return lines[:idx] + new_trimmed + lines[idx + len(trimmed) :]
    return None


def _trim_context(hunk: Hunk) -> list[str] | None:
    """The old block with leading/trailing pure-context lines removed."""
    body = hunk.lines
    first = next((i for i, line in enumerate(body) if line.startswith(("+", "-"))), None)
    if first is None:
        return None
    last = max(i for i, line in enumerate(body) if line.startswith(("+", "-")))
    core = body[first : last + 1]
    return [line[1:] for line in core if not line.startswith("+")]


def _trim_new(hunk: Hunk) -> list[str]:
    body = hunk.lines
    first = next((i for i, line in enumerate(body) if line.startswith(("+", "-"))), 0)
    last = max(
        (i for i, line in enumerate(body) if line.startswith(("+", "-"))),
        default=len(body) - 1,
    )
    core = body[first : last + 1]
    return [line[1:] for line in core if not line.startswith("-")]


def _find_block(lines: list[str], block: list[str], hint: int | None = None) -> int | None:
    if not block:
        return None
    if hint is not None and _block_at(lines, block, hint):
        return hint
    exact = _scan_block(lines, block, strip=False, hint=hint)
    if exact is not None:
        return exact
    return _scan_block(lines, block, strip=True, hint=hint)


def _scan_block(
    lines: list[str], block: list[str], strip: bool, hint: int | None = None
) -> int | None:
    n = len(block)
    if n == 0:
        return None
    target = [x.rstrip() for x in block] if strip else block
    matches: list[int] = []
    for i in range(0, len(lines) - n + 1):
        window = lines[i : i + n]
        cmp_window = [x.rstrip() for x in window] if strip else window
        if cmp_window == target:
            matches.append(i)
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    if hint is None:
        return None
    return min(matches, key=lambda i: abs(i - hint))


def _block_at(lines: list[str], block: list[str], idx: int) -> bool:
    if idx < 0 or idx + len(block) > len(lines):
        return False
    return lines[idx : idx + len(block)] == block


def _hunk_counts(body: list[str]) -> tuple[int, int]:
    old = new = 0
    for line in body:
        if line.startswith("\\"):
            continue
        if line.startswith("+"):
            new += 1
        elif line.startswith("-"):
            old += 1
        else:
            old += 1
            new += 1
    return old, new


def _hunks_are_additions(hunks: list[Hunk]) -> bool:
    return bool(hunks) and all(not hunk.old_lines and hunk.new_lines for hunk in hunks)


def _ensure_eol(text: str) -> str:
    return text if text.endswith("\n") else text + "\n"


def _match_old_header(line: str, in_hunk: bool) -> str | None:
    """Return the `---` header target, or None if this is not a file header.

    Inside a malformed hunk body only the unambiguous `a/...` / `/dev/null`
    forms count, so a removed source line starting with `---` is kept as code.
    """
    if not line.startswith("--- "):
        return None
    target = line[4:].strip()
    if not target:
        return None
    if target == "/dev/null" or target.startswith("a/"):
        return target
    return None if in_hunk else target


def _match_new_header(line: str, in_hunk: bool) -> str | None:
    if not line.startswith("+++ "):
        return None
    target = line[4:].strip()
    if not target:
        return None
    if target == "/dev/null" or target.startswith("b/"):
        return target
    return None if in_hunk else target


def _clean_path(path: str) -> str:
    path = path.strip()
    if "\t" in path:
        path = path.split("\t", 1)[0]
    path = path.strip().strip('"')
    if path.startswith("a/") or path.startswith("b/"):
        path = path[2:]
    return path.strip()


def _looks_like_patch(text: str) -> bool:
    return any(
        marker in text
        for marker in ("diff --git ", "*** Begin Patch", "*** Update File:", "\n--- a/", "\n@@ ")
    )


def _looks_like_apply_patch(text: str) -> bool:
    return "*** Begin Patch" in text or "*** Update File:" in text or "*** Add File:" in text


def _apply_patch_to_unified(text: str) -> str:
    """Translate OpenAI apply_patch / Begin Patch blocks into a rough unified diff."""
    files: list[FilePatch] = []
    current: FilePatch | None = None
    hunk: Hunk | None = None
    old_start = 1

    def flush_hunk() -> None:
        nonlocal hunk
        if current is not None and hunk is not None and hunk.lines:
            current.hunks.append(hunk)
        hunk = None

    def close_file() -> None:
        nonlocal current
        flush_hunk()
        if current is not None:
            files.append(current)
        current = None

    for raw in text.split("\n"):
        line = raw.rstrip("\r")
        if line.startswith("*** Begin Patch") or line.startswith("*** End Patch"):
            continue
        if line.startswith("*** Update File:"):
            close_file()
            current = FilePatch(path=_clean_path(line.split(":", 1)[1]))
            continue
        if line.startswith("*** Add File:"):
            close_file()
            current = FilePatch(path=_clean_path(line.split(":", 1)[1]), is_new=True)
            continue
        if line.startswith("*** Delete File:"):
            close_file()
            current = FilePatch(path=_clean_path(line.split(":", 1)[1]), is_delete=True)
            continue
        match = _HUNK_RE.match(line)
        if match or line.startswith("@@"):
            flush_hunk()
            if current is None:
                current = FilePatch(path="")
            if match:
                old_start = int(match.group(1))
                suffix = match.group(5) or ""
            else:
                old_start = 1
                suffix = ""
            hunk = Hunk(old_start=old_start, new_start=old_start, lines=[], header_suffix=suffix)
            continue
        if current is None:
            continue
        if hunk is None:
            hunk = Hunk(old_start=old_start, new_start=old_start, lines=[])
        if line.startswith(("+", "-", " ", "\\")):
            hunk.lines.append(line)
        elif line == "":
            hunk.lines.append(" ")
        else:
            hunk.lines.append(" " + line)
    close_file()
    for file in files:
        for h in file.hunks:
            while h.lines and h.lines[-1] == " ":
                h.lines.pop()
        file.hunks = [h for h in file.hunks if h.lines]
    return render_unified_diff([f for f in files if f.hunks or f.is_new or f.is_delete])
