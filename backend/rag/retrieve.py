"""Embeds a query and retrieves top-k chunks with citations.

All local. Reuses the FAISS index built by `backend.rag.index`.
"""

from __future__ import annotations

from pathlib import Path

from backend.rag.index import (
    INDEX_DIR,
    embed_texts,
    load_index,
)

_index = None
_chunks: list[dict] = []


def _ensure_loaded(index_dir: Path = INDEX_DIR) -> bool:
    """Load the index once; return True if a non-empty index is ready."""
    global _index, _chunks
    if _index is None:
        if not load_index(index_dir):
            return False
        from backend.rag.index import _index as _i, _chunks as _c
        _index, _chunks = _i, _c
    return _index.ntotal > 0 and len(_chunks) > 0


def retrieve(query: str, top_k: int = 3, index_dir: Path = INDEX_DIR) -> list[dict]:
    """Return top-k matching chunks with document/page citations.

    Each hit is a dict like:
        {"doc": "sop_material_handling.pdf", "page": 2,
         "text": "...", "score": 0.86}
    """
    if not _ensure_loaded(index_dir):
        return []

    query_vec = embed_texts([query])
    scores, idx = _index.search(query_vec, k=min(top_k, _index.ntotal))

    hits = []
    for rank in range(len(idx[0])):
        row = int(idx[0][rank])
        if row < 0 or row >= len(_chunks):
            continue
        hit = dict(_chunks[row])
        hit["score"] = round(float(scores[0][rank]), 4)
        hits.append(hit)
    return hits
