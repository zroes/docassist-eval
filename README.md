This is an RAG project for learning: it indexes a small slice of IBM watsonx documentation (and optional PDFs) so you can ask grounded questions against that corpus.

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
   .venv/bin/python src/cli.py build-chunks    # rebuild chunks.jsonl
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
