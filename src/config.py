"""
Central configuration: filesystem layout and chunking defaults.

All ``Path`` values are absolute, derived from this file's location, so scripts behave
the same whether you run them from the repo root or from ``src/``. Import as::

    import config
    print(config.CHUNKS_JSONL)
"""

from pathlib import Path

# Directory containing this package (``.../docassist-eval/src``).
_SRC_DIR = Path(__file__).resolve().parent
# Repository root (parent of ``src``).
PROJECT_ROOT = _SRC_DIR.parent

# --- Raw corpora (user-editable inputs) ---
RAW_TEXT_DIR = PROJECT_ROOT / "data" / "raw" / "text"
RAW_JSON_DIR = PROJECT_ROOT / "data" / "raw" / "json"
RAW_PDF_DIR = PROJECT_ROOT / "data" / "raw" / "pdf"

# --- Processed artifacts ---
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CHUNKS_JSONL = PROCESSED_DIR / "chunks.jsonl"
# Lexical index (built with Chroma index; used for hybrid retrieval)
BM25_INDEX_PATH = PROCESSED_DIR / "bm25_index.pkl"

# --- Vector store ---
CHROMA_PERSIST_DIR = PROJECT_ROOT / "chroma_db"
RESULTS_DIR = PROJECT_ROOT / "results"

# Chroma collection name (vendor-neutral).
CHROMA_COLLECTION_NAME = "rag_corpus"

# Sliding window for JSON-derived plain text (word counts).
DEFAULT_WORD_CHUNK_SIZE = 300
DEFAULT_WORD_OVERLAP = 50

# PDF secondary splitter (character counts after per-page block extraction).
PDF_MAX_CHUNK_CHARS = 2000
PDF_CHUNK_OVERLAP_CHARS = 200

# --- Hybrid retrieval + rerank (defaults tuned for small–medium corpora) ---
RETRIEVAL_DENSE_POOL = 80
RETRIEVAL_BM25_POOL = 80
RETRIEVAL_RERANK_POOL = 55
RETRIEVAL_RRF_K = 60
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# Hugging Face hub cache for cross-encoder weights (under repo root for portability)
HF_CACHE_DIR = PROJECT_ROOT / ".hf_cache"
