from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from .patch_format import extract_patch as extract_raw_patch
from .patch_format import make_applyable
from .schema import CandidatePool


def _env_proxy() -> str | None:
    for key in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        value = os.environ.get(key)
        if value:
            return value
    return None


def _openai_http_client(timeout: float = 900.0):
    import httpx

    kwargs: dict[str, Any] = {
        "timeout": httpx.Timeout(float(timeout), connect=30.0),
        "trust_env": True,
    }
    proxy = _env_proxy()
    if proxy:
        kwargs["proxy"] = proxy
    return httpx.Client(**kwargs)


def _load_dotenv() -> None:
    """Load a local .env without printing values. Existing env vars win."""
    for candidate in (Path(".env"), Path(__file__).resolve().parents[2] / ".env"):
        if not candidate.is_file():
            continue
        for raw in candidate.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            if key and key not in os.environ:
                os.environ[key] = value
        break

def serialize_context(
    pool: CandidatePool,
    mask,
    include_line_numbers: bool = True,
) -> str:
    parts: list[str] = []
    for unit, keep in zip(pool.units, mask.tolist()):
        if not keep:
            continue
        header = f"### {unit.path}"
        if unit.symbol:
            header += f" :: {unit.symbol}"
        if unit.start_line is not None:
            header += f" (lines {unit.start_line}-{unit.end_line})"
        parts.append(header)
        if include_line_numbers and unit.start_line is not None:
            lines = unit.text.splitlines()
            numbered = "\n".join(f"{unit.start_line + i:6d}: {line}" for i, line in enumerate(lines))
            parts.append(numbered)
        else:
            parts.append(unit.text)
    return "\n\n".join(parts)


_PATCH_RULES = """Return ONLY a unified git diff that `git apply` can apply at the repository root.

Required format:
- Start with `diff --git a/<path> b/<path>`
- Include `--- a/<path>` and `+++ b/<path>` (`/dev/null` for new or deleted files)
- Every hunk header `@@ -old_start,old_count +new_start,new_count @@` MUST equal the number of old/new lines in that hunk
- Copy context lines EXACTLY from the supplied repository context, including whitespace
- Do not invent `index` hashes, do not use apply_patch / `*** Begin Patch` / `*** End Patch`
- Do not wrap the diff in markdown fences and do not add explanations
- If the supplied context is insufficient, emit the smallest defensible patch rather than inventing unrelated files
"""


def build_patch_prompt(issue: str, context: str) -> str:
    return f"""You are fixing a software repository issue. Use only the supplied issue description and repository context to infer the required code change.

ISSUE
-----
{issue}

SELECTED REPOSITORY CONTEXT
---------------------------
{context}

{_PATCH_RULES}
"""


def build_repair_prompt(issue: str, context: str, bad_patch: str, apply_error: str) -> str:
    return f"""The previous patch could not be applied with `git apply`. Emit a corrected unified git diff only.

ISSUE
-----
{issue}

SELECTED REPOSITORY CONTEXT
---------------------------
{context}

PREVIOUS PATCH
--------------
{bad_patch}

GIT APPLY ERROR
---------------
{apply_error}

{_PATCH_RULES}
"""


def extract_patch(text: str) -> str:
    return extract_raw_patch(text)


def _usage_dict(response) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    return usage.model_dump() if usage is not None and hasattr(usage, "model_dump") else {}


class OpenAIPatchGenerator:
    """Downstream generator. RepoMARO itself never calls this class."""

    def __init__(
        self,
        model: str = "gpt-5.6-luna",
        reasoning_effort: str = "medium",
        max_output_tokens: int = 12000,
        apply_retries: int = 1,
        temperature: float | None = None,
        request_timeout: float = 900.0,
        timeout_retries: int = 2,
    ):
        _load_dotenv()
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set")
        from openai import OpenAI  # type: ignore

        self.request_timeout = float(request_timeout)
        self.timeout_retries = max(int(timeout_retries), 0)
        self.client = OpenAI(
            http_client=_openai_http_client(self.request_timeout),
            timeout=self.request_timeout,
            max_retries=0,
        )
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = int(max_output_tokens)
        self.apply_retries = int(apply_retries)
        self.temperature = None if temperature is None else float(temperature)

    def generate(
        self,
        issue: str,
        context: str,
        repo_root: str | Path | None = None,
    ) -> tuple[str, dict[str, Any]]:
        calls: list[dict[str, Any]] = []
        raw, response = self._complete(build_patch_prompt(issue, context), calls)
        patch, apply_info = make_applyable(raw, repo_root)
        repaired = False
        if repo_root is not None and not apply_info.get("apply_ok") and self.apply_retries > 0:
            repair = build_repair_prompt(issue, context, patch or raw, apply_info.get("apply_error") or "git apply failed")
            raw2, response2 = self._complete(repair, calls)
            patch2, apply_info2 = make_applyable(raw2, repo_root)
            if apply_info2.get("apply_ok") or not patch:
                patch, apply_info = patch2, apply_info2
                raw = raw2
                response = response2
                repaired = True
        return patch, {
            "model": self.model,
            "response_id": response.id,
            "usage": _usage_dict(response),
            # Every call made for this task, including a discarded repair attempt.
            # Requests that raised before returning have no usage and appear with
            # ``failed: true``.
            "calls": calls,
            "raw_output": raw,
            "apply": apply_info,
            "repaired": repaired,
            "temperature": self.temperature,
        }

    def _complete(self, prompt: str, calls: list[dict[str, Any]] | None = None):
        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": prompt,
            "reasoning": {"effort": self.reasoning_effort},
            "max_output_tokens": self.max_output_tokens,
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        last_exc: Exception | None = None
        attempts = self.timeout_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                response = self.client.responses.create(**kwargs)
                if calls is not None:
                    calls.append({"response_id": response.id, "usage": _usage_dict(response)})
                return response.output_text, response
            except Exception as exc:
                last_exc = exc
                if calls is not None:
                    calls.append({"failed": True, "error": str(exc)[:300]})
                msg = str(exc).lower()
                if self.temperature is not None and "temperature" in msg:
                    kwargs.pop("temperature", None)
                    continue
                transient = any(
                    token in msg
                    for token in ("timed out", "timeout", "temporar", "connect", "reset", "unavailable")
                )
                if transient and attempt < attempts:
                    wait = min(30 * attempt, 90)
                    print(
                        f"downstream timeout/retry {attempt}/{attempts} after {self.request_timeout:.0f}s: {exc}; sleep {wait}s",
                        flush=True,
                    )
                    time.sleep(wait)
                    continue
                raise
        assert last_exc is not None
        raise last_exc


def write_batch_requests(
    records: list[dict[str, str]],
    output_path: str | Path,
    model: str = "gpt-5.6-luna",
    reasoning_effort: str = "medium",
    max_output_tokens: int = 12000,
) -> Path:
    """Write OpenAI Batch API JSONL for /v1/responses.

    Each record needs: custom_id, issue, context.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for rec in records:
            body = {
                "model": model,
                "input": build_patch_prompt(rec["issue"], rec["context"]),
                "reasoning": {"effort": reasoning_effort},
                "max_output_tokens": max_output_tokens,
            }
            row = {"custom_id": rec["custom_id"], "method": "POST", "url": "/v1/responses", "body": body}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return output_path


def submit_batch(request_jsonl: str | Path, completion_window: str = "24h") -> str:
    _load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")
    from openai import OpenAI  # type: ignore

    client = OpenAI(http_client=_openai_http_client())
    with Path(request_jsonl).open("rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    batch = client.batches.create(input_file_id=uploaded.id, endpoint="/v1/responses", completion_window=completion_window)
    return batch.id
