from __future__ import annotations

import math


class TokenCounter:
    """Token counter with a deterministic no-download fallback."""

    def __init__(self, model: str | None = None):
        self.model = model
        self._encoding = None
        try:
            import tiktoken  # type: ignore

            if model:
                try:
                    self._encoding = tiktoken.encoding_for_model(model)
                except Exception:
                    self._encoding = tiktoken.get_encoding("o200k_base")
            else:
                self._encoding = tiktoken.get_encoding("o200k_base")
        except Exception:
            self._encoding = None

    def count(self, text: str) -> int:
        if self._encoding is not None:
            return max(1, len(self._encoding.encode(text, disallowed_special=())))
        # Conservative deterministic approximation used only when tiktoken is absent.
        # Final paper runs should install the `tokens` extra.
        return max(1, math.ceil(len(text.encode("utf-8")) / 4.0))
