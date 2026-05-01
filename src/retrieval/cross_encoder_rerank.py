"""
Cross-encoder reranking: (query, passage) scoring with a small local model.

Uses ``sentence_transformers.CrossEncoder`` once per query over a candidate pool
fused from dense + BM25 retrieval.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import config

_model: Any = None


def get_cross_encoder():
    """
    Lazily load and cache the cross-encoder (downloads weights on first use).

    Sets ``HF_HOME`` to :data:`config.HF_CACHE_DIR` on first load so models live under
    the repo (avoids permission issues with a read-only home cache in some environments).

    Returns:
        A ``CrossEncoder`` instance.

    Raises:
        ImportError: If ``sentence_transformers`` is not installed.
    """
    global _model
    if _model is None:
        import os

        cache = config.HF_CACHE_DIR
        cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(cache))

        from sentence_transformers import CrossEncoder

        _model = CrossEncoder(config.CROSS_ENCODER_MODEL)
    return _model


def rerank_by_cross_encoder(
    query: str,
    chunks: list[dict],
    top_k: int,
    batch_size: int = 32,
) -> list[dict]:
    """
    Score each chunk's ``text`` against ``query`` and return the top ``top_k`` copies.

    Sets ``rerank_score`` (higher is better) and ``distance`` to ``0.0`` so downstream
    grounded prompts do not drop reranked passages solely on vector distance.

    Args:
        query: User question.
        chunks: Dicts with at least ``text``; other keys copied through.
        top_k: Number of chunks to return after reranking.
        batch_size: Forward batch size for the cross-encoder.

    Returns:
        A new list of dicts, length at most ``min(top_k, len(chunks))``, best first.
    """
    if not chunks:
        return []
    model = get_cross_encoder()
    texts = [c.get("text") or "" for c in chunks]
    pairs = [(query, t) for t in texts]
    scores = model.predict(pairs, show_progress_bar=False, batch_size=batch_size)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores)
    take = min(top_k, len(chunks))
    out: list[dict] = []
    for idx in order[:take]:
        i = int(idx)
        row = dict(chunks[i])
        row["rerank_score"] = float(scores[i])
        row["distance"] = 0.0
        out.append(row)
    return out
