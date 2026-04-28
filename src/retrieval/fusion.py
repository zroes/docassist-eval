"""Reciprocal Rank Fusion (RRF) for merging ordered ranked document id lists."""

from __future__ import annotations


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = 60,
    max_docs: int | None = None,
) -> list[str]:
    """
    Merge several rankings of chunk ids into one list by RRF score.

    Args:
        ranked_lists: Each inner list is chunk ids from best to worst for one retriever.
        k: RRF constant (higher flattens differences between ranks). Typical values 30–80.
        max_docs: If set, return at most this many unique ids (best first).

    Returns:
        Unique chunk ids sorted by descending fused score.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    merged = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)
    if max_docs is not None:
        merged = merged[:max_docs]
    return merged
