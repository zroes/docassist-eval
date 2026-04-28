"""
Assemble the searchable corpus as JSON Lines (chunks.jsonl).

Reads normalized documents from ``data/raw/json/*.json`` (HTML scrape pipeline) and
``data/raw/pdf/*.pdf`` (layout-aware extraction), applies format-specific chunking,
and writes one JSON object per line for :mod:`index_chroma`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import config
from chunking.strategies import chunk_words, recursive_char_chunks
from ingest.pdf import extract_pdf_pages, pdf_document_meta


def _pdf_section_heading_hint(page_text: str, piece: str) -> str:
    """
    Pull a short manual-style heading for embedding prefixes.

    Prefer a ``▌`` section line that appears inside ``piece`` so sub-chunks inherit
    the correct subsection title (e.g. "CREATING A NEW CLIP FROM SONG VIEW").
    """
    for line in piece.splitlines():
        s = line.strip()
        if s.startswith("▌"):
            return s[1:].strip()[:220]
    for line in page_text.splitlines():
        s = line.strip()
        if s.startswith("▌"):
            return s[1:].strip()[:220]
    return ""


def _pdf_embed_text_prefix(title: str, page_num: int, page_text: str, piece: str) -> str:
    """Line(s) prepended to PDF chunk body for denser semantic + lexical overlap."""
    hint = _pdf_section_heading_hint(page_text, piece)
    parts = [title, f"Page {page_num}"]
    if hint:
        parts.append(hint)
    return " | ".join(parts) + "\n\n"


def _write_chunk(out_f, record: dict) -> None:
    """Serialize one chunk dict as a single UTF-8 JSON line (JSONL)."""
    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")


def process_json_file(json_path: Path, out_f, chunk_size: int, overlap: int) -> int:
    """
    Chunk one JSON document (``title``, ``source``, ``category``, ``text``) with a
    sliding word window and append rows to the open output file.

    Args:
        json_path: Path to a document JSON file under ``data/raw/json/``.
        out_f: Open text file handle opened for writing (JSONL stream).
        chunk_size: Target number of words per chunk.
        overlap: Word overlap between consecutive windows.

    Returns:
        Number of chunk rows written for this file (0 if ``text`` is missing or empty).
    """
    with open(json_path, "r", encoding="utf-8") as in_f:
        doc = json.load(in_f)
    text = doc.get("text", "")
    if not text:
        return 0
    base = json_path.stem
    n = 0
    for idx, chunk_str in enumerate(chunk_words(text, chunk_size=chunk_size, overlap=overlap)):
        _write_chunk(
            out_f,
            {
                "doc_id": base,
                "chunk_id": f"{base}_chunk_{idx}",
                "title": doc.get("title", "Unknown Title"),
                "category": doc.get("category", "Unknown Category"),
                "source": doc.get("source", "Unknown Source"),
                "text": chunk_str,
                "page_start": -1,
                "page_end": -1,
            },
        )
        n += 1
    print(f"Processed JSON '{doc.get('title', base)}' -> {n} chunks.")
    return n


def process_pdf_file(pdf_path: Path, out_f) -> int:
    """
    Extract each PDF page as ordered blocks, sub-chunk long pages with
    :func:`chunking.strategies.recursive_char_chunks`, and append rows.

    Args:
        pdf_path: Path to a ``.pdf`` under ``data/raw/pdf/``.
        out_f: Open text file handle for JSONL output.

    Returns:
        Total chunk rows written for this PDF (0 if no text could be extracted).
    """
    meta = pdf_document_meta(pdf_path)
    pages = extract_pdf_pages(pdf_path)
    if not pages:
        print(f"Skipping PDF (no extractable text): {pdf_path}")
        return 0
    doc_id = meta["doc_id"]
    n = 0
    for page_num, page_text in pages:
        for sub_i, piece in enumerate(
            recursive_char_chunks(
                page_text,
                max_chars=config.PDF_MAX_CHUNK_CHARS,
                overlap=config.PDF_CHUNK_OVERLAP_CHARS,
            )
        ):
            prefix = _pdf_embed_text_prefix(meta["title"], page_num, page_text, piece)
            body = prefix + piece
            _write_chunk(
                out_f,
                {
                    "doc_id": doc_id,
                    "chunk_id": f"{doc_id}_p{page_num}_c{sub_i}",
                    "title": meta["title"],
                    "category": meta["category"],
                    "source": meta["source"],
                    "text": body,
                    "page_start": page_num,
                    "page_end": page_num,
                },
            )
            n += 1
    print(f"Processed PDF '{meta['title']}' -> {n} chunks.")
    return n


def build_chunks(
    output_path: Path | None = None,
    word_chunk_size: int | None = None,
    word_overlap: int | None = None,
) -> int:
    """
    Rebuild the full ``chunks.jsonl`` from all JSON and PDF inputs.

    Args:
        output_path: Destination JSONL path; defaults to :data:`config.CHUNKS_JSONL`.
        word_chunk_size: Override default word window size for JSON documents.
        word_overlap: Override default word overlap for JSON documents.

    Returns:
        Total number of chunk records written across all sources.
    """
    output_path = output_path or config.CHUNKS_JSONL
    word_chunk_size = word_chunk_size or config.DEFAULT_WORD_CHUNK_SIZE
    word_overlap = word_overlap or config.DEFAULT_WORD_OVERLAP

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.RAW_JSON_DIR.mkdir(parents=True, exist_ok=True)
    config.RAW_PDF_DIR.mkdir(parents=True, exist_ok=True)

    # json_files = sorted(config.RAW_JSON_DIR.glob("*.json"))
    json_files = []
    pdf_files = sorted(config.RAW_PDF_DIR.glob("*.pdf"))

    total = 0
    print(f"Found {len(json_files)} JSON file(s), {len(pdf_files)} PDF file(s).")

    with open(output_path, "w", encoding="utf-8") as out_f:
        for jf in json_files:
            total += process_json_file(jf, out_f, word_chunk_size, word_overlap)
        for pf in pdf_files:
            total += process_pdf_file(pf, out_f)

    print(f"\nSuccess! Wrote {total} chunks to {output_path}")
    return total


if __name__ == "__main__":
    build_chunks()
