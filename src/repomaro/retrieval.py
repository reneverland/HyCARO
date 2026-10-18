from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize as sk_normalize

from .schema import ContextUnit

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def minmax_normalize(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        return scores
    lo = float(scores.min())
    hi = float(scores.max())
    if hi - lo < 1e-12:
        return np.zeros_like(scores) if hi <= 0 else np.ones_like(scores)
    return (scores - lo) / (hi - lo)


@dataclass
class BM25Retriever:
    k1: float = 1.5
    b: float = 0.75

    def score(self, query: str, units: list[ContextUnit]) -> np.ndarray:
        docs = [tokenize(f"{u.path} {u.symbol or ''} {u.text}") for u in units]
        query_terms = tokenize(query)
        n = len(docs)
        if n == 0:
            return np.zeros(0, dtype=np.float64)
        lengths = np.asarray([len(d) for d in docs], dtype=np.float64)
        avgdl = float(lengths.mean()) if lengths.size else 1.0
        df: Counter[str] = Counter()
        tf_docs: list[Counter[str]] = []
        for doc in docs:
            tf = Counter(doc)
            tf_docs.append(tf)
            for term in tf:
                df[term] += 1
        scores = np.zeros(n, dtype=np.float64)
        for term in query_terms:
            freq_docs = df.get(term, 0)
            if freq_docs == 0:
                continue
            idf = math.log(1.0 + (n - freq_docs + 0.5) / (freq_docs + 0.5))
            for i, tf in enumerate(tf_docs):
                f = tf.get(term, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * lengths[i] / max(avgdl, 1e-9))
                scores[i] += idf * (f * (self.k1 + 1)) / denom
        return scores


class DenseRetriever:
    """Dense retriever with SentenceTransformers backend and a deterministic hashing fallback."""

    _MODEL_CACHE: dict[tuple[str, str], object] = {}

    def __init__(
        self,
        enabled: bool = False,
        model_name: str = "Qwen/Qwen3-Embedding-4B",
        device: str = "cuda",
        batch_size: int = 8,
        max_seq_length: int = 1024,
    ):
        self.enabled = enabled
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_seq_length = int(max_seq_length)
        self._model = None
        self._hash = HashingVectorizer(
            n_features=2048,
            alternate_sign=False,
            norm=None,
            token_pattern=r"(?u)\b\w+\b",
        )

    def _load_model(self):
        if self._model is not None:
            return self._model
        if not self.enabled:
            return None
        key = (self.model_name, self.device, self.max_seq_length)
        cached = self._MODEL_CACHE.get(key)
        if cached is not None:
            self._model = cached
            return self._model
        try:
            import torch
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "Dense retrieval is enabled but sentence-transformers is not installed. "
                "Run `pip install -e '.[dense]'` or disable candidate_pool.dense.enabled."
            ) from e
        dtype = torch.bfloat16 if str(self.device).startswith("cuda") else torch.float32
        self._model = SentenceTransformer(
            self.model_name,
            device=self.device,
            trust_remote_code=True,
            model_kwargs={"torch_dtype": dtype},
        )
        self._model.max_seq_length = self.max_seq_length
        self._MODEL_CACHE[key] = self._model
        return self._model

    def encode_units(self, units: list[ContextUnit]) -> np.ndarray:
        texts = [f"{u.path}\n{u.symbol or ''}\n{u.text}" for u in units]
        if not texts:
            return np.zeros((0, 1), dtype=np.float64)
        model = self._load_model()
        if model is None:
            mat = self._hash.transform(texts)
            mat = sk_normalize(mat, norm="l2", copy=False)
            return mat.toarray().astype(np.float64)
        emb = model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(emb, dtype=np.float64)

    def score(self, query: str, units: list[ContextUnit], embeddings: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        if embeddings is None:
            embeddings = self.encode_units(units)
        if not len(units):
            return np.zeros(0, dtype=np.float64), embeddings
        model = self._load_model()
        if model is None:
            q = self._hash.transform([query])
            q = sk_normalize(q, norm="l2", copy=False).toarray().astype(np.float64)[0]
        else:
            q = np.asarray(model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0], dtype=np.float64)
        scores = embeddings @ q
        return scores.astype(np.float64), embeddings


def greedy_pack_by_ranking(scores: np.ndarray, costs: np.ndarray, budget: int) -> np.ndarray:
    mask = np.zeros(len(scores), dtype=bool)
    used = 0
    # Stable ordering matters for exact reproducibility.
    order = np.argsort(-np.asarray(scores), kind="mergesort")
    for j in order.tolist():
        c = int(costs[j])
        if c <= 0:
            continue
        if used + c <= budget:
            mask[j] = True
            used += c
    return mask


def reciprocal_rank_fusion(rankings: Iterable[np.ndarray], k: int = 60) -> np.ndarray:
    rankings = list(rankings)
    if not rankings:
        return np.zeros(0, dtype=np.float64)
    m = len(rankings[0])
    out = np.zeros(m, dtype=np.float64)
    for scores in rankings:
        order = np.argsort(-scores, kind="mergesort")
        ranks = np.empty(m, dtype=np.int64)
        ranks[order] = np.arange(1, m + 1)
        out += 1.0 / (k + ranks)
    return out
