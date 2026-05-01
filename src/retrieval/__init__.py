"""
Retrieval helpers: BM25 lexical index, RRF fusion, and cross-encoder reranking.

Used by :mod:`retrieve` after dense vectors are stored in Chroma. The BM25 artifact
is built alongside indexing in :mod:`index_chroma`.
"""
