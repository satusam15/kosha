# Kosh

A **local, offline-first agentic AI assistant**. It reads local documents,
answers questions with a ReAct-style agent loop, runs generated code in an
isolated sandbox, and produces real deliverables (Word / Excel / PowerPoint) —
all talking only to **localhost** endpoints. Nothing phones home.

```
mrpl-workbench/
├── backend/
│   ├── agent/loop.py           # ReAct loop: plan → call tool → observe → iterate
│   ├── agent/tools.py          # tool defs: read_file, write_file, search_docs, run_code, generate_deliverable
│   ├── router/model_router.py  # keyword-based local model selection (code/vision/general)
│   ├── rag/index.py            # FAISS index from PDFs via bge-small embeddings
│   ├── rag/retrieve.py         # embed query, retrieve top-k chunks with citations
│   ├── ocr/vision.py           # Moondream/Qwen2-VL (Ollama) OCR for images/scanned PDFs
│   ├── deliverables/schemas.py # strict pydantic models (approval_note, calc_sheet, presentation)
│   ├── deliverables/generate_docx.py / generate_xlsx.py / generate_pptx.py
│   ├── sandbox/executor.py     # runs code in Docker: --network none --memory=512m
│   ├── network_monitor.py      # live outbound-connection count (offline watchdog)
│   └── main.py                 # FastAPI app tying it together
├── frontend/index.html         # 3-pane chat UI (history · chat/steps · deliverables)
├── data/sops/                  # sample SOP PDFs for RAG (placeholder)
├── data/demo_input/            # sample scanned inspection report (placeholder)
├── requirements.txt
├── docker-compose.yml
├── Dockerfile.sandbox
└── tests/test_core.py
```

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Optional but recommended (needed for FAISS RAG + scanned-PDF OCR):

```bash
pip install faiss-cpu sentence-transformers pdf2image
sudo apt-get install -y poppler-utils   # for scanned-PDF rasterization
```

### 2. Start local models (Ollama)

Make sure Ollama is running and you have pulled the models you'll use. All
model names live in **`backend/router/model_router.py`** (and mirror in
`backend/ocr/vision.py`) — **edit them to match what you pulled**:

```bash
ollama serve &
ollama pull qwen2.5:7b          # general reasoning   -> REASONING_MODEL
ollama pull qwen2.5-coder:7b    # code                -> CODER_MODEL
ollama pull moondream           # vision/OCR          -> VISION_MODEL
```

> `TODO`s in the code mark exactly where to plug in your real model names
> and file paths.

### 3. (Optional) Build the RAG index

Drop SOP PDFs into `data/sops/`, then:

```bash
python -m backend.rag.index        # or: curl -X POST localhost:8000/index/build
```

### 4. Run the app

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000/** for the 3-pane UI. Key API endpoints:

| Method | Path              | Purpose                                        |
|--------|-------------------|------------------------------------------------|
| POST   | `/chat`           | run the agent loop on a message                |
| GET    | `/network/stats`  | live outbound-connection count (offline check) |
| GET    | `/health`         | backend + Ollama reachability                  |
| POST   | `/index/build`    | build/rebuild the RAG index                    |
| GET    | `/retrieve?q=...` | quick RAG retrieval for debugging              |
| GET    | `/sandbox`        | sandbox health                                 |
| GET    | `/ocr?path=...`   | OCR an image or scanned PDF                    |

---

## How it works

### Agent loop (`backend/agent/loop.py`)
A genuine iterate-until-done loop, not a single LLM call:

1. A system prompt describes the available tools and an `ACTION {…} END_ACTION` JSON protocol.
2. The message is routed to a local model (see routing below) and streamed back.
3. If the model emits an `ACTION`, the tool runs and the `OBSERVATION` is fed back in, then it loops.
4. When the model answers without an action (or iteration cap is hit), the loop stops.

Each step is captured into an `AgentStep` trace returned by `/chat` for the
UI's "Agent Steps" pane.

### Model routing (`backend/router/model_router.py`)
Keyword-based, fully local:
- **code** / **script** / **function** / **debug** … → coder model
- image / file upload or OCR keywords → vision model
- everything else → general reasoning model

### Sandbox (`backend/sandbox/executor.py`)
Runs generated code via:

```bash
docker run --rm --network none --memory=512m --cpus=1 <image> python -c <code>
```

and captures stdout/stderr. If Docker is missing it falls back to a **local,
UNSAFE** subprocess with a loud warning (dev convenience only — don't use for
untrusted input).

### Deliverables
Generation output is validated against **strict pydantic schemas**
(`extra="forbid"`, `strict=True`) before rendering, so malformed payloads are
rejected up front. Renderers produce:
- `approval_note.docx` (python-docx)
- `calc_sheet.xlsx` with **real formulas** (`=B2*C2`, `SUM`, `=Subtotal*rate`)
- `presentation.pptx` (python-pptx: title / bullets / table layouts)

### Offline watchdog (`backend/network_monitor.py`)
Samples outbound TCP connections with `psutil` and reports how many reach
non-local hosts. The UI shows a live "outbound non-local" pill that turns
amber if anything leaves the box.

---

## Notes & TODOs

- **All model names and file paths are placeholders** — find every `TODO:`
  in the code and plug in your real values (Ollama model names, `data/sops`
  PDF locations, workspace paths).
- `faiss` / `sentence-transformers` are imported **lazily** so the app still
  runs on boxes where they aren't installed; RAG just reports "index
  unavailable".
- For a fully air-gapped deployment, mirror the Python wheels and the
  bge-small model files ahead of time (see `backend/rag/index.py` TODO).
- The frontend is plain HTML/JS (no build step) so it works offline and is
  served by the FastAPI app at the site root.

## Tests

```bash
python -m pytest tests/ -q
```

The suite covers model routing, the tool registry, strict schema validation,
and the docx/xlsx/pptx renderers — no Ollama/Docker required.
