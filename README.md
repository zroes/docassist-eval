This is an RAG project for learning: it indexes a small slice of IBM watsonx documentation so you can ask grounded questions against that corpus. **Note:** Currently, the program pulls information exclusively from two large PDF files located in `data/raw/pdf/` (HTML/JSON ingestion is present in the code but bypassed by default). If these PDFs are missing, the CLI will automatically prompt you to download them when building chunks.

## Internal Logic Pipeline

1. **PDF Extraction (`src/ingest/pdf.py`)**
   - The system reads from the two large PDF files in `data/raw/pdf/`.
   - Uses PyMuPDF (`fitz`) to extract text sequentially block-by-block, preserving the reading order for each page.
   - Extracts document-level metadata (such as title and source).

2. **Chunking (`src/chunk_corpus.py`, `src/chunking/strategies.py`)**
   - The extracted text is processed into smaller segments to ensure it fits well within embedding and LLM context windows.
   - Uses a recursive character splitting strategy: it attempts to split text naturally on boundaries (`\n\n`, `\n`, `. `, ` `) to stay under a maximum character limit (e.g., 2000 characters).
   - If a section cannot be split naturally, it forces a split using an overlapping sliding window.
   - A contextual prefix (document title, page number, and any manually identified section heading hints like lines starting with "▌") is added to each chunk.
   - The finalized chunks are serialized to `data/processed/chunks.jsonl`.

3. **Indexing (`src/index_chroma.py`, `src/retrieval/bm25_index.py`)**
   - The JSONL chunks are embedded and indexed into two distinct systems:
     - **ChromaDB**: Embeds chunk texts and metadata into a local persistent vector store for semantic/dense search.
     - **BM25**: Builds a sparse index (`bm25_index.pkl`) for exact-keyword search.

4. **Retrieval (`src/retrieve.py`)**
   - When a user asks a question, the query is passed through a **hybrid retrieval** pipeline.
   - It fetches a candidate pool of hits using both ChromaDB (semantic) and BM25 (keyword).
   - Results are merged and normalized using Reciprocal Rank Fusion (RRF).
   - A local cross-encoder then reranks these merged candidates to determine the true top-K most relevant chunks.

5. **Prompting the LLM (`src/generate.py`)**
   - The top-K retrieved chunks are formatted into a context block containing the document titles, sources, page numbers, and text.
   - A prompt is assembled (typically in "grounded" mode) instructing the LLM (Gemini) to answer strictly based on the provided context and to cite its sources.
   - The API is called, and the final grounded response is returned to the user.

## Layout

| Step | Module | Role |
|------|--------|------|
| 1 | `src/extract_html_txt_to_json.py` | `data/raw/text/*.txt` → `data/raw/json/*.json` |
| 2 | `src/chunk_corpus.py` | JSON + `data/raw/pdf/*.pdf` → `data/processed/chunks.jsonl` |
| 3 | `src/index_chroma.py` | JSONL → Chroma (`chroma_db/`, collection `rag_corpus`) |
| 4 | `src/retrieve.py` | Query Chroma, format hits |
| 5 | `src/generate.py` | Build prompt, call Gemini |
| 6 | `src/qa_benchmark.py` | Batch questions → CSV in `results/` |

Shared: `src/config.py`, `src/schemas.py`, `src/chunking/`, `src/ingest/`, `src/retrieval/`, CLI `src/cli.py`.

**Retrieval:** `src/retrieve.py` merges **Chroma dense** search with **BM25** (reciprocal rank fusion), then **cross-encoder reranks** a candidate pool (model weights cache under `.hf_cache/`). Re-run **`build-chunks`** then **`index`** after changing PDFs so chunks (including per-section heading prefixes embedded in PDF text) and `bm25_index.pkl` stay in sync. Flags: `src/cli.py ask --no-hybrid` / `--no-rerank` for ablations.

## How to run

1. **Create a virtualenv and install dependencies** (from the repo root):

   ```bash
   python3 -m venv .venv
   .venv/bin/pip install -r requirements.txt
   ```

2. **Pipeline (typical order)** — use the venv’s Python:

   ```bash
   .venv/bin/python src/cli.py ingest-html     # only if you need to refresh JSON from .txt
   .venv/bin/python src/cli.py build-chunks    # rebuild chunks.jsonl (will prompt to download required PDFs if missing)
   .venv/bin/python src/cli.py index           # load Chroma (replaces collection)
   ```

   Add PDFs: copy files into `data/raw/pdf/`, or:

   ```bash
   .venv/bin/python src/cli.py ingest-pdf /path/to/file.pdf
   .venv/bin/python src/cli.py build-chunks && .venv/bin/python src/cli.py index
   ```

3. **Ask one question** (retrieval + Gemini; set `GEMINI_API_KEY` first):

   ```bash
   export GEMINI_API_KEY="your-key"
   .venv/bin/python src/cli.py ask "What is RAG useful for?"
   .venv/bin/python src/cli.py ask --show-context --top-k 8 "Summarize the main steps"
   ```

4. **Try retrieval only** (no API key):

   ```bash
   .venv/bin/python src/retrieve.py
   ```

5. **Generation / benchmark** — set `GEMINI_API_KEY`, then:

   ```bash
   export GEMINI_API_KEY="your-key"
   .venv/bin/python src/generate.py          # small built-in mock example
   .venv/bin/python src/qa_benchmark.py      # full benchmark CSV (slow, many API calls)
   ```

Individual stages can also be run directly (same `src/` path bootstrap as the CLI), e.g. `.venv/bin/python src/index_chroma.py`.
