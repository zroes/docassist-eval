"""
Semantic retrieval against the Chroma collection built by :mod:`index_chroma`.

Default path for QA uses **hybrid BM25 + dense** fusion (RRF) and a **local cross-encoder
rerank** over a candidate pool. Legacy dense-only behavior remains available for debugging.
"""

import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import chromadb

import config
from retrieval.bm25_index import load_bm25_index, top_chunk_ids_bm25
from retrieval.cross_encoder_rerank import rerank_by_cross_encoder
from retrieval.fusion import reciprocal_rank_fusion


def get_chroma_collection():
    """
    Open the persistent Chroma client and return the configured collection.

    Returns:
        A ``chromadb.Collection`` for ``config.CHROMA_COLLECTION_NAME``.

    Raises:
        Exception: If the collection does not exist (index has not been run).
    """
    client = chromadb.PersistentClient(path=str(config.CHROMA_PERSIST_DIR))
    collection = client.get_collection(name=config.CHROMA_COLLECTION_NAME)
    return collection


def retrieve_chunks(collection, queries: list[str], top_k: int = 3):
    """
    Run a batch vector query: embed ``queries`` and return top-``top_k`` hits each.

    Args:
        collection: Chroma collection from :func:`get_chroma_collection`.
        queries: List of natural-language query strings.
        top_k: Number of chunks to return per query.

    Returns:
        Chroma ``query`` result dict (``ids``, ``documents``, ``metadatas``, ``distances``),
        shaped as nested lists indexed by query position.
    """
    return collection.query(query_texts=queries, n_results=top_k)


def _as_int(value, default=-1):
    """
    Coerce Chroma metadata values (sometimes strings) to ``int``.

    Args:
        value: Raw metadata value or ``None``.
        default: Fallback when coercion fails.

    Returns:
        Integer page index or ``default``.
    """
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _chunk_dict_from_row(chunk_id: str, document: str, meta: dict, distance: float) -> dict:
    return {
        "chunk_id": chunk_id,
        "text": document,
        "title": meta["title"],
        "source": meta["source"],
        "category": meta["category"],
        "distance": float(distance),
        "page_start": _as_int(meta.get("page_start"), -1),
        "page_end": _as_int(meta.get("page_end"), -1),
    }


def _hydrate_chunks_from_chroma(collection, ids_ordered: list[str], dense_dist_by_id: dict[str, float] | None) -> list[dict]:
    """
    Fetch documents/metadata for ids and preserve ``ids_ordered`` ordering.

    Args:
        collection: Chroma collection.
        ids_ordered: Chunk ids in candidate order.
        dense_dist_by_id: Optional map chunk_id -> dense distance for this query (fallback 1.0).

    Returns:
        List of chunk dicts suitable for :func:`format_retrieval_results` / :mod:`generate`.
    """
    if not ids_ordered:
        return []
    got = collection.get(ids=list(ids_ordered), include=["documents", "metadatas"])
    by_id: dict[str, tuple[str, dict]] = {}
    for i, cid in enumerate(got["ids"]):
        by_id[cid] = (got["documents"][i], got["metadatas"][i])
    out: list[dict] = []
    for cid in ids_ordered:
        if cid not in by_id:
            continue
        doc, meta = by_id[cid]
        dist = 1.0
        if dense_dist_by_id is not None:
            dist = float(dense_dist_by_id.get(cid, 1.0))
        out.append(_chunk_dict_from_row(cid, doc, meta, dist))
    return out


def format_retrieval_results(queries: list[str], retrieved_data: dict) -> list[dict]:
    """
    Flatten Chroma's batch response into one result object per query.

    Args:
        queries: The same query strings passed to :func:`retrieve_chunks`.
        retrieved_data: Raw return value from ``collection.query``.

    Returns:
        A list of length ``len(queries)``, each element
        ``{"query": str, "top_chunks": [ {chunk_id, text, title, source, category,
        distance, page_start, page_end}, ... ]}``.
    """
    formatted_results = []

    for q_idx, query in enumerate(queries):
        formatted_chunks = []
        for i in range(len(retrieved_data["ids"][q_idx])):
            md = retrieved_data["metadatas"][q_idx][i]
            formatted_chunks.append(
                _chunk_dict_from_row(
                    retrieved_data["ids"][q_idx][i],
                    retrieved_data["documents"][q_idx][i],
                    md,
                    retrieved_data["distances"][q_idx][i],
                )
            )

        formatted_results.append({"query": query, "top_chunks": formatted_chunks})

    return formatted_results


def retrieve_for_grounding(
    collection,
    queries: list[str],
    top_k: int = 8,
    *,
    hybrid: bool = True,
    rerank: bool = True,
    dense_pool: int | None = None,
    bm25_pool: int | None = None,
    rerank_pool: int | None = None,
    rrf_k: int | None = None,
) -> list[dict]:
    """
    Retrieve chunks for each query using dense + BM25 (RRF) and optional cross-encoder rerank.

    Args:
        collection: Chroma collection.
        queries: One or more user questions.
        top_k: Final number of chunks per query passed to the LLM.
        hybrid: If True and a BM25 index exists, merge BM25 with dense via RRF.
        rerank: If True, rerank the fused pool with the cross-encoder (local).
        dense_pool: Chroma ``n_results`` cap (default from config).
        bm25_pool: How many BM25 hits to merge (default from config).
        rerank_pool: Max unique candidates after fusion to score with the cross-encoder.
        rrf_k: RRF smoothing constant (default from config).

    Returns:
        Same structure as :func:`format_retrieval_results`: one dict per query with ``top_chunks``.
    """
    dense_pool = dense_pool or config.RETRIEVAL_DENSE_POOL
    bm25_pool = bm25_pool or config.RETRIEVAL_BM25_POOL
    rerank_pool = rerank_pool or config.RETRIEVAL_RERANK_POOL
    rrf_k = rrf_k or config.RETRIEVAL_RRF_K

    n_docs = collection.count()
    d_k = min(dense_pool, max(1, n_docs))
    dense_res = collection.query(query_texts=queries, n_results=d_k)

    bm25_state = load_bm25_index() if hybrid else None
    if hybrid and bm25_state is None:
        print("Note: BM25 index not found; run `index` after `build-chunks`. Using dense-only fusion pool.")

    results: list[dict] = []
    for q_idx, query in enumerate(queries):
        dense_ids = dense_res["ids"][q_idx]
        dense_dists = dense_res["distances"][q_idx]
        dense_dist_by_id = {cid: dense_dists[i] for i, cid in enumerate(dense_ids)}

        if hybrid and bm25_state is not None:
            bm25, chunk_ids = bm25_state
            bm25_ids = top_chunk_ids_bm25(bm25, chunk_ids, query, bm25_pool)
            fused_ids = reciprocal_rank_fusion(
                [dense_ids, bm25_ids],
                k=rrf_k,
                max_docs=rerank_pool,
            )
        else:
            fused_ids = dense_ids[:rerank_pool]

        candidates = _hydrate_chunks_from_chroma(collection, fused_ids, dense_dist_by_id)

        if rerank and candidates:
            try:
                top_chunks = rerank_by_cross_encoder(query, candidates, top_k)
            except Exception as exc:
                print(f"Cross-encoder rerank failed ({exc}); using fused order without rerank.")
                top_chunks = candidates[:top_k]
                for c in top_chunks:
                    c["distance"] = 0.0
        else:
            top_chunks = candidates[:top_k]
            for c in top_chunks:
                c["distance"] = 0.0

        results.append({"query": query, "top_chunks": top_chunks})

    return results


def run_sample_queries() -> None:
    """
    Demo: run a few fixed queries with hybrid rerank, write ``results/retrieval_examples.json``.

    Returns:
        None.
    """
    collection = get_chroma_collection()

    sample_queries = [
        "How do I use Python to chat with a model?",
        "What is RAG useful for?",
        "From a brand new machine, walk me through getting watsonx set up for a basic conversation",
    ]

    results_to_save = retrieve_for_grounding(collection, sample_queries, top_k=3)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = config.RESULTS_DIR / "retrieval_examples.json"
    with open(output_file, "w", encoding="utf-8") as out_f:
        json.dump(results_to_save, out_f, indent=4)

    print(f"\nSuccessfully saved {len(sample_queries)} query examples to {output_file}")


if __name__ == "__main__":
    run_sample_queries()
