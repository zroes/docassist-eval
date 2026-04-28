"""Shared lightweight tokenization for BM25 (language-agnostic, no stemming)."""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)

# Strip from BM25 *queries* only so rare content words (e.g. add, new, track) keep weight.
BM25_QUERY_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "what",
        "which",
        "who",
        "whom",
        "this",
        "that",
        "these",
        "those",
        "am",
        "how",
        "when",
        "where",
        "why",
        "on",
        "in",
        "to",
        "for",
        "of",
        "at",
        "by",
        "with",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "under",
        "again",
        "further",
        "then",
        "once",
        "and",
        "or",
        "but",
        "if",
        "because",
        "so",
        "than",
        "too",
        "very",
        "just",
        "can",
        "could",
        "should",
        "would",
        "may",
        "might",
        "must",
    }
)


def tokenize(text: str) -> list[str]:
    """
    Lowercase alphanumeric tokens for BM25.

    Args:
        text: Raw chunk or query string.

    Returns:
        List of tokens (may be empty).
    """
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def tokenize_query_for_bm25(query: str) -> list[str]:
    """
    Tokenize a user query for BM25, dropping common English function words.

    If that removes every token, falls back to the full :func:`tokenize` output.

    Args:
        query: Raw search string.

    Returns:
        Token list for ``BM25Okapi.get_scores``.
    """
    toks = tokenize(query)
    filt = [t for t in toks if t not in BM25_QUERY_STOPWORDS]
    return filt if filt else toks
