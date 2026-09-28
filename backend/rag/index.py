"""Builds a FAISS index from local PDFs using bge-small embeddings.

All local. Embedding happens on your machine (sentence-transformers loads a
small bge-small-en-v1.5 model from the local HF cache — see TODO about the
first download).

Pipeline:
  1. Walk `data/sops` for `.pdf` files.
  2. Extract text page-by-page (pypdf).
  3. Chunk each page into overlapping windows.
  4. Embed chunks with a BGE-small model.
  5. Store vectors in a FAISS index (with metadata) under `data/rag_index`.
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

# The embedding model is already cached locally after the first download, but
# sentence-transformers/huggingface_hub still phone home on every load to
# check for updated files unless told not to -- measured via a live outbound
# HTTPS connection to HF Hub's CDN during a plain index build. This is the
# fully-offline project's one real network leak; force it off here.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# NOTE: `faiss` and `numpy` are imported lazily inside functions below so that
# the rest of the workbench (main.py, agent, deliverables) still imports and
# runs even on machines where faiss-cpu isn't installed (e.g. some Python 3.13
# wheels). RAG simply reports "index unavailable" until faiss is present.

# TODO: point these at your actual corpus and output locations.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = PROJECT_ROOT / "data" / "sops"
INDEX_DIR = PROJECT_ROOT / "data" / "rag_index"

INDEX_FILE = INDEX_DIR / "index.faiss"
METADATA_FILE = INDEX_DIR / "chunks.json"

# TODO: model name. bge-small-en-v1.5 needs a one-time local download from
# HuggingFace. For a fully air-gapped box, mirror the model files first.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE = 400          # characters
CHUNK_OVERLAP = 60        # characters


# ---------------------------------------------------------------------------
# Lazily loaded singletons
# ---------------------------------------------------------------------------
_encoder = None          # sentence-transformers model
_index = None            # faiss IndexFlatIP
_chunks: list[dict] = []  # metadata parallel to index rows


def _get_encoder():
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer

        _encoder = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
    return _encoder


def embed_texts(texts: list[str]):
    """Embed a list of strings into a float32 matrix (unit normalized)."""
    import numpy as np

    model = _get_encoder()
    vecs = model.encode(texts, normalize_embeddings=True,
                        show_progress_bar=False, convert_to_numpy=True)
    return np.asarray(vecs, dtype=np.float32)


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------
def extract_pdf_text(pdf_path: Path) -> list[str]:
    """Return one string per page of a PDF."""
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return pages


def chunk_text(text: str, size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping character windows.

    Always includes a final window ending exactly at len(text) -- otherwise,
    whenever step > (len(text) - size), range() stops after the first window
    and silently drops the tail of the text (this previously cut off the
    corrosion-escalation sentence of the demo SOP, since its page text was
    only ~64 characters longer than one window).
    """
    if len(text) <= size:
        return [text] if text.strip() else []
    step = size - overlap
    starts = list(range(0, len(text) - size + 1, step))
    if starts[-1] + size < len(text):
        starts.append(len(text) - size)
    return [text[i:i + size] for i in starts]


# ---------------------------------------------------------------------------
# Index build
# ---------------------------------------------------------------------------
def build_index(docs_dir: Path = DOCS_DIR, index_dir: Path = INDEX_DIR) -> dict:
    """Build (or rebuild) the FAISS index from all PDFs in `docs_dir`."""
    docs_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(docs_dir.glob("*.pdf"))
    if not pdfs:
        # Still emit an empty, usable index so retrieval degrades gracefully.
        print(f"[index] no PDFs found under {docs_dir}; building empty index.")
        _save_empty(index_dir)
        return {"docs": 0, "chunks": 0, "files": []}

    texts: list[str] = []
    chunks: list[dict] = []

    for pdf in pdfs:
        try:
            pages = extract_pdf_text(pdf)
        except Exception as exc:  # noqa: BLE001
            print(f"[index] skipped {pdf.name}: {exc}")
            continue
        for page_no, page_text in enumerate(pages, start=1):
            for piece in chunk_text(page_text):
                if piece.strip():
                    chunks.append({
                        "doc": pdf.name,
                        "page": page_no,
                        "text": piece,
                    })
                    texts.append(piece)

    if not texts:
        print("[index] no text extracted; building empty index.")
        _save_empty(index_dir)
        return {"docs": len(pdfs), "chunks": 0, "files": [p.name for p in pdfs]}

    import faiss

    vectors = embed_texts(texts)

    dim = vectors.shape[1]
    faiss_index = faiss.IndexFlatIP(dim)      # cosine via normalized vectors
    faiss_index.add(vectors)

    faiss.write_index(faiss_index, str(index_dir / "index.faiss"))
    (index_dir / "chunks.json").write_text(
        json.dumps(chunks, indent=2), encoding="utf-8"
    )
    # Persist the loaded model reference so retrieval reuses the same index.
    with (index_dir / "state.pkl").open("wb") as fh:
        pickle.dump({"model": EMBEDDING_MODEL}, fh)

    print(f"[index] built index: {len(texts)} chunks from {len(pdfs)} PDFs.")
    return {"docs": len(pdfs), "chunks": len(texts),
            "files": [p.name for p in pdfs]}


def _save_empty(index_dir: Path) -> None:
    """Write an empty-but-valid index so retrieval never hard-crashes."""
    import faiss

    empty = faiss.IndexFlatIP(1)  # dimension 1; retrieval short-circuits on 0 rows
    faiss.write_index(empty, str(index_dir / "index.faiss"))
    (index_dir / "chunks.json").write_text("[]", encoding="utf-8")


def load_index(index_dir: Path = INDEX_DIR):
    """Load the persisted index + metadata into module globals (if present)."""
    import faiss

    global _index, _chunks
    if not (index_dir / "index.faiss").exists():
        return False
    _index = faiss.read_index(str(index_dir / "index.faiss"))
    _chunks = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
    return True


if __name__ == "__main__":
    # CLI: `python -m backend.rag.index`
    print(build_index())
