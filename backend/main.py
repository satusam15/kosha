"""FastAPI app tying the Kavach backend together.

All model calls go to local Ollama at `http://localhost:11434`; all docs
come from the local RAG index; code runs in the local Docker sandbox. No
external API calls anywhere.

Endpoints:
    POST /chat                 run the agent loop on a message
    GET  /network/stats        live outbound-connection count (offline watchdog)
    GET  /health               backend + Ollama reachability
    POST /index/build          build/rebuild the RAG index from data/sops
    GET  /retrieve             quick RAG retrieval for debugging
    GET  /memory               list/search persistent memory (facts/entities/relations)
    GET  /approvals            list pending/approved/rejected side-effect actions
    POST /approvals/{id}/approve   actually run a queued action
    POST /approvals/{id}/reject    permanently discard a queued action
    GET  /sandbox              sandbox health
    GET  /ocr?path=...         extract text from an image / scanned pdf
    POST /upload                upload an image/PDF, get back a path for /chat
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Optional

import ollama
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.agent.loop import run_agent
from backend.router.model_router import OLLAMA_BASE_URL, REASONING_MODEL
from backend import network_monitor

app = FastAPI(title="Kavach", version="0.1.0")

# Allow the (any-origin, local) frontend to talk to this API. Still offline.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
REPO_ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = REPO_ROOT / "backend" / "workspace" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

_ALLOWED_UPLOAD_SUFFIXES = {".png", ".jpg", ".jpeg", ".pdf"}
_MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25MB


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    has_upload: bool = False


class ChatResponse(BaseModel):
    answer: str
    model: str
    task_type: str
    iterations: int
    steps: list[dict]
    task_id: str
    saved_facts: list[dict]


class BuildIndexResponse(BaseModel):
    ok: bool
    detail: dict


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    ollama_ok = False
    try:
            client = ollama.Client(host=OLLAMA_BASE_URL)
            client.list()
            ollama_ok = True
    except Exception:  # noqa: BLE001
            ollama_ok = False
    return {
        "status": "ok",
        "ollama": ollama_ok,
        "ollama_url": OLLAMA_BASE_URL,
        "default_model": REASONING_MODEL,
        "offline_only": True,
    }


# ---------------------------------------------------------------------------
# Chat (agent loop)
# ---------------------------------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message is empty")

    try:
        result = run_agent(req.message, has_upload=req.has_upload)
    except Exception as exc:  # noqa: BLE001 - surface Ollama outage cleanly
        raise HTTPException(
            status_code=503,
            detail=(
                f"Could not reach the local model backend (Ollama at "
                f"localhost:11434). Is it running? Original error: {exc}"
            ),
        ) from exc

    steps = [
        {"kind": s.kind, "content": s.content, "tool": s.tool}
        for s in result.steps
    ]
    return ChatResponse(
        answer=result.answer,
        model=result.routed.model if result.routed else "unknown",
        task_type=result.routed.task_type if result.routed else "unknown",
        iterations=result.iterations,
        steps=steps,
        task_id=result.task_id,
        saved_facts=result.saved_facts,
    )


# ---------------------------------------------------------------------------
# Network / offline watchdog
# ---------------------------------------------------------------------------
@app.get("/network/stats")
def network_stats():
    return network_monitor.network_stats()


@app.get("/network/count")
def network_count():
    return {"outbound_nonlocal": network_monitor.outbound_nonlocal_count()}


# ---------------------------------------------------------------------------
# RAG
# ---------------------------------------------------------------------------
@app.post("/index/build", response_model=BuildIndexResponse)
def build_index():
    try:
        from backend.rag.index import build_index as _build

        detail = _build()
        return BuildIndexResponse(ok=True, detail=detail)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/retrieve")
def retrieve(q: str = Query(..., min_length=1), top_k: int = 3):
    try:
        from backend.rag.retrieve import retrieve as _retrieve

        return {"hits": _retrieve(q, top_k=top_k)}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Persistent memory (facts / entities / relations)
# ---------------------------------------------------------------------------
@app.get("/memory")
def memory(q: str = Query("", description="leave empty to list recent facts")):
    from backend.memory import store as memory_store

    if q.strip():
        return {"hits": memory_store.search_memory(q)}
    return {
        "facts": memory_store.all_facts(),
        "entities": memory_store.all_entities(),
        "relations": memory_store.all_relations(),
    }


# ---------------------------------------------------------------------------
# Approval queue for side-effecting agent actions (checkpoint 5)
# ---------------------------------------------------------------------------
@app.get("/approvals")
def list_approvals(status: Optional[str] = Query(None, description="pending|approved|rejected")):
    from backend.approvals.store import list_approvals as _list

    return {"approvals": _list(status)}


@app.post("/approvals/{approval_id}/approve")
def approve(approval_id: int):
    from backend.approvals.store import decide

    try:
        return decide(approval_id, approve=True)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/approvals/{approval_id}/reject")
def reject(approval_id: int):
    from backend.approvals.store import decide

    try:
        return decide(approval_id, approve=False)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Sandbox health
# ---------------------------------------------------------------------------
@app.get("/sandbox")
def sandbox_health():
    from backend.sandbox.executor import check_sandbox_health

    return check_sandbox_health()


# ---------------------------------------------------------------------------
# OCR / vision
# ---------------------------------------------------------------------------
@app.get("/ocr")
def ocr(path: str = Query(..., description="local path to an image or scanned PDF")):
    from backend.agent.tools import _safe_repo_path

    # CORS is wide open (allow_origins=["*"]) for local frontend convenience,
    # which means any site open in the user's browser could otherwise GET
    # this endpoint with an arbitrary path and exfiltrate OCR'd content from
    # any image file on disk. Restrict to the repo root, same as the agent's
    # own ocr_image tool.
    try:
        p = _safe_repo_path(path)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"file not found: {path}")
    from backend.ocr.vision import ocr_image, ocr_pdf

    if p.suffix.lower() == ".pdf":
        return {"text": ocr_pdf(p)}
    return {"text": ocr_image(p)}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """Save an uploaded image/PDF into the workspace and return its path.

    The returned path is relative to the repo root, so it can be dropped
    straight into a /chat message (e.g. "Analyze <path> against our SOPs")
    and read by the ocr_image tool, which is itself sandboxed to the repo
    root -- same trust boundary as every other local file this app touches.
    """
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in _ALLOWED_UPLOAD_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"unsupported file type {suffix!r}; allowed: {sorted(_ALLOWED_UPLOAD_SUFFIXES)}",
        )

    data = await file.read()
    if len(data) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="file too large (max 25MB)")

    safe_name = Path(file.filename).name  # strip any path components
    dest = UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}_{safe_name}"
    dest.write_bytes(data)

    return {
        "path": str(dest.relative_to(REPO_ROOT)),
        "filename": safe_name,
        "size": len(data),
    }


# ---------------------------------------------------------------------------
# Serve generated deliverables so the UI's download links actually resolve --
# previously there was no route for these at all, so any link the frontend
# built pointing at a deliverable's filesystem path 404'd.
# ---------------------------------------------------------------------------
DELIVERABLES_DIR = REPO_ROOT / "backend" / "workspace" / "deliverables"
DELIVERABLES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/deliverables", StaticFiles(directory=str(DELIVERABLES_DIR)), name="deliverables")


# ---------------------------------------------------------------------------
# Serve the frontend (3-pane chat UI) as static files at the site root,
# so the browser talks to /chat, /network/*, /ocr etc. same-origin (offline).
# ---------------------------------------------------------------------------
FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"


@app.get("/")
def index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    # TODO: bind host 0.0.0.0 only if you want the LAN to reach it; for a
    # purely local workbench, 127.0.0.1 is the offline-safe default.
    uvicorn.run(app, host=os.getenv("MRPL_HOST", "127.0.0.1"), port=8000)
