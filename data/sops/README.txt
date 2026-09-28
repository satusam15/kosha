Placeholder for RAG source documents.

Drop your Standard Operating Procedure (SOP) PDFs in this folder, then build
the index with:

    python -m backend.rag.index          # from the project root

or via the API:

    curl -X POST http://127.0.0.1:8000/index/build

Supported: .pdf files. Text is extracted page-by-page with `pypdf`,
chunked with overlap, embedded with bge-small, and stored in a local FAISS
index under `data/rag_index/`.

TODO: add real SOP PDFs here (e.g. safety_harness_sop.pdf,
     material_handling_sop.pdf, inspection_checksheet.pdf).
