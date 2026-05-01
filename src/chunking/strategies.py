"""
Text chunking: fixed word windows (JSON pipeline) and recursive character splits (PDF).

Generators yield chunk strings one at a time to keep memory usage predictable on
large documents.
"""

from __future__ import annotations


def chunk_words(text: str, chunk_size: int = 300, overlap: int = 50):
    """
    Split whitespace-tokenized text into overlapping word windows.

    Args:
        text: Full document plain text.
        chunk_size: Number of words per yielded segment.
        overlap: Words shared between consecutive windows (must be less than ``chunk_size``).

    Yields:
        Space-joined strings, each up to ``chunk_size`` words (last segment may be shorter).
    """
    words = text.split()
    if not words:
        return
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunk = words[i : i + chunk_size]
        yield " ".join(chunk)


def _merge_small_parts(parts: list[str], max_chars: int) -> list[str]:
    """
    Greedy merge of string parts into segments not exceeding ``max_chars``.

    Args:
        parts: Non-empty fragments (e.g. split paragraphs).
        max_chars: Maximum characters per merged segment (including single spaces).

    Returns:
        List of merged strings, each at most ``max_chars`` unless a single part exceeds it.
    """
    merged: list[str] = []
    buf = ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if not buf:
            buf = p
        elif len(buf) + 1 + len(p) <= max_chars:
            buf = f"{buf} {p}"
        else:
            merged.append(buf)
            buf = p
    if buf:
        merged.append(buf)
    return merged


def recursive_char_chunks(
    text: str,
    max_chars: int = 2000,
    overlap: int = 200,
):
    """
    Split long text on paragraph / line / sentence boundaries, then hard-window if needed.

    Separator order: ``\\n\\n``, ``\\n``, ``. ``, then spaces; if a segment is still
    longer than ``max_chars``, advance a sliding window with ``overlap`` character overlap.

    Args:
        text: Typically one PDF page worth of extracted text.
        max_chars: Soft maximum length per yielded chunk.
        overlap: Character overlap between forced windows on un-splittable runs.

    Yields:
        Substrings of ``text``, each intended to stay under embedding context noise limits.
    """
    text = text.strip()
    if not text:
        return

    def split_oversized(s: str):
        if len(s) <= max_chars:
            yield s
            return
        splitters = ("\n\n", "\n", ". ", " ")
        for sep in splitters:
            if sep not in s:
                continue
            raw_parts = [x.strip() for x in s.split(sep) if x.strip()]
            if sep == " ":
                step = max(1, max_chars - overlap)
                for i in range(0, len(s), step):
                    yield s[i : i + max_chars]
                return
            merged = _merge_small_parts(raw_parts, max_chars)
            for m in merged:
                yield from split_oversized(m)
            return
        step = max(1, max_chars - overlap)
        for i in range(0, len(s), step):
            yield s[i : i + max_chars]

    for piece in split_oversized(text):
        if len(piece) <= max_chars:
            yield piece
            continue
        step = max(1, max_chars - overlap)
        for i in range(0, len(piece), step):
            yield piece[i : i + max_chars]
