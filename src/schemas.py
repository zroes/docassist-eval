"""
Typed shapes for data crossing the chunk → index boundary.

These types document the JSONL schema consumed by :mod:`index_chroma`. Runtime code
mostly uses plain ``dict`` for flexibility; use ``ChunkRecord`` for annotations and IDE help.
"""

from typing import NotRequired, TypedDict


class ChunkRecord(TypedDict):
    """
    One row in ``chunks.jsonl`` and the logical payload mirrored in Chroma metadata.

    Attributes:
        doc_id: Stable id for the source file (JSON stem or PDF stem).
        chunk_id: Globally unique id for this chunk (used as Chroma point id).
        title: Human-readable document title for citations.
        category: Coarse grouping (e.g. product area, or the literal ``pdf`` for PDF uploads).
        source: Canonical source string (URL or ``file://`` URI).
        text: Chunk body embedded and searched.
        page_start: First PDF page for this chunk, or ``-1`` if not from a paginated PDF.
        page_end: Last PDF page (same as ``page_start`` for current chunkers) or ``-1``.
    """

    doc_id: str
    chunk_id: str
    title: str
    category: str
    source: str
    text: str
    page_start: NotRequired[int]
    page_end: NotRequired[int]
