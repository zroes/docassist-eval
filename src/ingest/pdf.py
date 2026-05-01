"""
PDF text extraction with per-page block geometry (PyMuPDF / ``fitz``).

Extracted text is passed to :mod:`chunking.strategies` for page-local splitting so
chunk metadata can retain stable page numbers for citations.
"""

from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF


def _page_text_from_blocks(page: fitz.Page) -> str:
    """
    Concatenate text blocks in top-to-then-left reading order for one PDF page.

    Args:
        page: A PyMuPDF page object.

    Returns:
        Plain text with blocks separated by blank lines.
    """
    blocks = page.get_text("blocks")
    text_blocks = [b for b in blocks if len(b) >= 5 and isinstance(b[4], str) and b[4].strip()]
    text_blocks.sort(key=lambda b: (round(b[1], 2), round(b[0], 2)))
    return "\n\n".join(b[4].strip() for b in text_blocks)


def extract_pdf_pages(pdf_path: str | Path) -> list[tuple[int, str]]:
    """
    Extract non-empty pages as (1-based page index, text).

    Args:
        pdf_path: Filesystem path to a ``.pdf`` file.

    Returns:
        List of ``(page_number, text)`` tuples; pages with no extractable text are omitted.
    """
    path = Path(pdf_path)
    doc = fitz.open(path)
    try:
        pages: list[tuple[int, str]] = []
        for i in range(doc.page_count):
            text = _page_text_from_blocks(doc.load_page(i)).strip()
            if text:
                pages.append((i + 1, text))
        return pages
    finally:
        doc.close()


def pdf_document_meta(pdf_path: str | Path) -> dict:
    """
    Build document-level metadata for chunk records (title, source URI, category).

    Args:
        pdf_path: Path to the PDF; resolved for stable ``file://`` source strings.

    Returns:
        Dict with keys ``doc_id``, ``title``, ``source``, ``category`` (always ``\"pdf\"``).
    """
    path = Path(pdf_path).resolve()
    doc = fitz.open(path)
    try:
        meta = doc.metadata or {}
        title = (meta.get("title") or "").strip()
        if not title:
            title = path.stem.replace("_", " ")
        return {
            "doc_id": path.stem,
            "title": title,
            "source": path.as_uri() if path.exists() else str(path),
            "category": "pdf",
        }
    finally:
        doc.close()
