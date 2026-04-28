"""
Persisted BM25 index over chunk texts (same chunk_ids as Chroma).

Built during :func:`index_chroma.index_chunks_to_chroma` and loaded at query time
for lexical recall complementary to dense embeddings.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

import config
from retrieval.tokenize import tokenize, tokenize_query_for_bm25


def build_corpus_tokens_and_ids(chunks: list[dict]) -> tuple[list[list[str]], list[str]]:
    """
    Tokenize each chunk body in lockstep with Chroma ids.

    Args:
        chunks: Records with ``chunk_id`` and ``text`` (same order as indexing).

    Returns:
        ``(corpus_tokens, chunk_ids)`` where ``corpus_tokens[i]`` corresponds to ``chunk_ids[i]``.
    """
    chunk_ids = [c["chunk_id"] for c in chunks]
    corpus_tokens = [tokenize(c.get("text", "")) for c in chunks]
    return corpus_tokens, chunk_ids


def top_chunk_ids_bm25(
    bm25: BM25Okapi,
    chunk_ids: list[str],
    query: str,
    top_n: int,
) -> list[str]:
    """
    Return the top ``top_n`` chunk ids by BM25 score for ``query``.

    Args:
        bm25: Fitted index whose row order matches ``chunk_ids``.
        chunk_ids: Id for corpus row ``i``.
        query: Raw query string.
        top_n: How many ids to return (fewer if corpus is smaller).

    Returns:
        Chunk ids sorted best BM25 score first.
    """
    q_tokens = tokenize_query_for_bm25(query)
    if not q_tokens or not chunk_ids:
        return []
    scores = bm25.get_scores(q_tokens)
    n = len(scores)
    if n == 0:
        return []
    take = min(top_n, n)
    if take == n:
        order = np.argsort(-scores)
    else:
        partial = np.argpartition(-scores, take - 1)[:take]
        order = partial[np.argsort(-scores[partial])]
    return [chunk_ids[int(i)] for i in order[:take]]


def save_bm25_index(corpus_tokens: list[list[str]], chunk_ids: list[str], path: Path | None = None) -> None:
    """
    Persist tokenized corpus + ids for refitting BM25 on load.

    Args:
        corpus_tokens: Token lists, same length as ``chunk_ids``.
        chunk_ids: Chroma point ids.
        path: Destination file; defaults to :data:`config.BM25_INDEX_PATH`.

    Returns:
        None.
    """
    path = path or config.BM25_INDEX_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"version": 1, "chunk_ids": chunk_ids, "corpus_tokens": corpus_tokens}
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_bm25_index(path: Path | None = None) -> tuple[BM25Okapi, list[str]] | None:
    """
    Load BM25 index from disk.

    Args:
        path: Pickle path; defaults to :data:`config.BM25_INDEX_PATH`.

    Returns:
        ``(bm25, chunk_ids)`` if the file exists and is valid; otherwise ``None``.
    """
    path = path or config.BM25_INDEX_PATH
    if not path.is_file():
        return None
    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        chunk_ids = payload["chunk_ids"]
        corpus_tokens = payload["corpus_tokens"]
        bm25 = BM25Okapi(corpus_tokens)
        return bm25, chunk_ids
    except Exception:
        return None
