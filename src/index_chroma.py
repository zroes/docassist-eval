"""
Load ``chunks.jsonl`` into a local Chroma persistent vector store.

Replaces the target collection on each run so embeddings stay aligned with the
current corpus (no stale duplicate IDs).
"""

import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import chromadb

import config
from retrieval.bm25_index import build_corpus_tokens_and_ids, save_bm25_index


def load_chunks(filepath: Path | None = None):
    """
    Read a JSONL file of chunk records into memory.

    Args:
        filepath: Path to ``chunks.jsonl``; defaults to :data:`config.CHUNKS_JSONL`.

    Returns:
        List of dicts, each with keys ``chunk_id``, ``text``, ``title``, ``category``,
        ``source``, ``doc_id``, and optional ``page_start`` / ``page_end``.
    """
    filepath = filepath or config.CHUNKS_JSONL
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def index_chunks_to_chroma(chunks: list[dict]) -> None:
    """
    Embed and store chunk texts in Chroma under :data:`config.CHROMA_COLLECTION_NAME`.

    Deletes any legacy ``watsonx_docs`` collection, deletes the current collection name
    if present, then recreates it and calls ``collection.add`` once for the full batch.

    Args:
        chunks: Chunk dicts as produced by :mod:`chunk_corpus` (must include required keys).

    Returns:
        None. Prints progress to stdout.
    """
    print("Initializing ChromaDB...")
    client = chromadb.PersistentClient(path=str(config.CHROMA_PERSIST_DIR))

    name = config.CHROMA_COLLECTION_NAME
    for legacy in ("watsonx_docs",):
        try:
            client.delete_collection(name=legacy)
            print(f"Removed legacy collection '{legacy}'.")
        except Exception:
            pass
    try:
        client.delete_collection(name=name)
        print(f"Removed existing collection '{name}' for a clean rebuild.")
    except Exception:
        pass

    collection = client.get_or_create_collection(name=name)

    ids = []
    documents = []
    metadatas = []

    print("Preparing data for indexing...")

    for chunk in chunks:
        ids.append(chunk["chunk_id"])
        documents.append(chunk["text"])
        metadatas.append(
            {
                "title": chunk["title"],
                "category": chunk["category"],
                "source": chunk["source"],
                "doc_id": chunk["doc_id"],
                "page_start": int(chunk.get("page_start", -1)),
                "page_end": int(chunk.get("page_end", -1)),
            }
        )

    print(f"Adding {len(documents)} chunks to the vector database...")
    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    print("Indexing complete!")

    corpus_tokens, bm25_ids = build_corpus_tokens_and_ids(chunks)
    save_bm25_index(corpus_tokens, bm25_ids)
    print(f"Saved BM25 index to {config.BM25_INDEX_PATH}")


def main() -> None:
    """
    CLI entry: load default chunks path and index into Chroma.

    Returns:
        None.
    """
    chunks = load_chunks()
    print(f"Found {len(chunks)} chunks")
    index_chunks_to_chroma(chunks)


if __name__ == "__main__":
    main()
