from pathlib import Path
import subprocess

from repomaro.downstream import extract_patch
from repomaro.patch_format import check_git_apply, make_applyable, sanitize_patch


def _git_repo(tmp_path: Path, files: dict[str, str]) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@t.t"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True, capture_output=True)
    for rel, text in files.items():
        dest = repo / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True)
    return repo


def test_extract_patch_from_fence():
    raw = "Sure.\n```diff\ndiff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1,1 +1,1 @@\n-old\n+new\n```\n"
    assert extract_patch(raw).startswith("diff --git a/a.py")


def test_sanitize_recounts_hunk():
    bad = """diff --git a/mod.py b/mod.py
index 0000000..1111111 100644
--- a/mod.py
+++ b/mod.py
@@ -3,7 +3,7 @@ def foo():
     else:
         value = 1
-        value = 1
+        value = 2

     return value
"""
    clean = sanitize_patch(bad)
    assert "index " not in clean
    assert "@@ -3,5 +3,5 @@" in clean or "@@ -3,6 +3,6 @@" in clean


def test_sanitize_strips_end_patch_and_empty_hunk():
    bad = """diff --git a/f.py b/f.py
--- a/f.py
+++ b/f.py
@@ -1,3 +1,3 @@
@@ -10,3 +10,3 @@
     a = 1
-    b = 2
+    b = 3
     c = 4
*** End Patch
"""
    clean = sanitize_patch(bad)
    assert "*** End Patch" not in clean
    assert clean.count("@@ ") == 1


def test_begin_patch_rebuilds_and_applies(tmp_path):
    repo = _git_repo(
        tmp_path,
        {
            "astropy/io/fits/fitsrec.py": (
                "def write():\n"
                "        # Replace exponent separator in floating point numbers\n"
                "        if 'D' in format:\n"
                "            output_field.replace(encode_ascii('E'), encode_ascii('D'))\n"
                "        return output_field\n"
            )
        },
    )
    raw = """*** Begin Patch
*** Update File: astropy/io/fits/fitsrec.py
@@
         # Replace exponent separator in floating point numbers
         if 'D' in format:
-            output_field.replace(encode_ascii('E'), encode_ascii('D'))
+            output_field = output_field.replace(encode_ascii('E'),
+                                                 encode_ascii('D'))
*** End Patch
"""
    patch, info = make_applyable(raw, repo)
    assert info["apply_ok"], info
    assert patch.startswith("diff --git a/astropy/io/fits/fitsrec.py")
    assert "output_field = output_field.replace" in patch


def test_wrong_counts_rebuild_against_repo(tmp_path):
    src = (
        "def _cstack(left, right):\n"
        "    else:\n"
        "        cright = np.zeros((noutp, right.shape[1]))\n"
        "        cright[-right.shape[0]:, -right.shape[1]:] = 1\n"
        "\n"
        "    return np.hstack([cleft, cright])\n"
    )
    repo = _git_repo(tmp_path, {"astropy/modeling/separable.py": src})
    bad = """diff --git a/astropy/modeling/separable.py b/astropy/modeling/separable.py
index 0000000..1111111 100644
--- a/astropy/modeling/separable.py
+++ b/astropy/modeling/separable.py
@@ -1,7 +1,7 @@ def _cstack(left, right):
     else:
         cright = np.zeros((noutp, right.shape[1]))
-        cright[-right.shape[0]:, -right.shape[1]:] = 1
+        cright[-right.shape[0]:, -right.shape[1]:] = right

     return np.hstack([cleft, cright])
"""
    assert not check_git_apply(repo, bad)[0]
    patch, info = make_applyable(bad, repo)
    assert info["apply_ok"], (info, patch)
    ok, err = check_git_apply(repo, patch)
    assert ok, err


def test_offset_hunk_still_applies(tmp_path):
    repo = _git_repo(
        tmp_path,
        {
            "pkg/mod.py": (
                "class C:\n"
                "    def make_bytes(self, value):\n"
                "        if isinstance(value, bytes):\n"
                "            return value\n"
                "        if isinstance(value, str):\n"
                "            return value.encode()\n"
            )
        },
    )
    patch, info = make_applyable(
        """diff --git a/pkg/mod.py b/pkg/mod.py
--- a/pkg/mod.py
+++ b/pkg/mod.py
@@ -108,4 +108,4 @@ class C:
         if isinstance(value, bytes):
-            return value
+            return bytes(value)
         if isinstance(value, str):
             return value.encode()
""",
        repo,
    )
    assert info["apply_ok"], info
    assert "return bytes(value)" in patch


def test_rebuild_uses_removed_lines_when_context_is_wrong(tmp_path):
    repo = _git_repo(
        tmp_path,
        {"pkg/mod.py": "alpha = 1\nbeta = 2\ngamma = 3\n"},
    )
    patch, info = make_applyable(
        """diff --git a/pkg/mod.py b/pkg/mod.py
--- a/pkg/mod.py
+++ b/pkg/mod.py
@@ -10,3 +10,3 @@
     not really here
-beta = 2
+beta = 99
     also wrong
""",
        repo,
    )
    assert info["apply_ok"], info
    assert "beta = 99" in patch


def test_trailing_blank_context_is_not_stripped_into_corrupt_patch(tmp_path):
    repo = _git_repo(tmp_path, {"mod.py": "a = 1\nb = 2\n\nc = 3\n"})
    raw = """diff --git a/mod.py b/mod.py
--- a/mod.py
+++ b/mod.py
@@ -1,3 +1,3 @@
 a = 1
-b = 2
+b = 9
 
"""
    patch, info = make_applyable(raw, repo)
    assert info["apply_ok"], (info, patch)
    ok, err = check_git_apply(repo, patch)
    assert ok, err
    assert "b = 9" in patch


def test_second_file_header_is_not_eaten(tmp_path):
    repo = _git_repo(tmp_path, {"a.py": "x = 1\n", "b.py": "y = 1\n"})
    raw = """diff --git a/a.py b/a.py
--- a/a.py
+++ b/a.py
@@ -1,1 +1,1 @@
-x = 1
+x = 2
diff --git a/b.py b/b.py
--- a/b.py
+++ b/b.py
@@ -1,1 +1,1 @@
-y = 1
+y = 2
"""
    patch, info = make_applyable(raw, repo)
    assert info["apply_ok"], (info, patch)
    assert "a.py" in patch and "b.py" in patch
    assert "x = 2" in patch and "y = 2" in patch


def test_already_applyable_patch_is_kept(tmp_path):
    repo = _git_repo(tmp_path, {"mod.py": "alpha = 1\nbeta = 2\n"})
    raw = """diff --git a/mod.py b/mod.py
--- a/mod.py
+++ b/mod.py
@@ -1,2 +1,2 @@
 alpha = 1
-beta = 2
+beta = 3
"""
    assert check_git_apply(repo, raw)[0]
    patch, info = make_applyable(raw, repo)
    assert info["apply_ok"], info
    assert info["rebuilt"] is False
    assert "beta = 3" in patch


def test_bare_at_at_is_not_kept_as_context(tmp_path):
    repo = _git_repo(
        tmp_path,
        {"pkg/mod.py": "class C:\n    def foo(self):\n        if self.path is None:\n            return []\n"},
    )
    raw = """diff --git a/pkg/mod.py b/pkg/mod.py
--- a/pkg/mod.py
+++ b/pkg/mod.py
@@ -1,3 +1,4 @@
@@
     def foo(self):
+        if callable(self.path):
+            return []
         if self.path is None:
             return []
"""
    patch, info = make_applyable(raw, repo)
    assert info["apply_ok"], (info, patch)
    assert "callable(self.path)" in patch
    body_lines = [line for line in patch.splitlines() if not line.startswith("@@ -") and line.strip() == "@@"]
    assert not body_lines
