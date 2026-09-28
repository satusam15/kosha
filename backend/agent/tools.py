"""Tool definitions used by the ReAct agent loop.

Each tool is a plain Python callable wrapped in a small `Tool` descriptor
(name, description, args schema, func). The agent serializes these into a
prompt and executes them by name.

All tools are strictly local — no external network calls.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# Root of the workbench project (one level up from backend/).
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# TODO: point these at your real local paths / corpus location.
WORKSPACE_DIR = PROJECT_ROOT / "workspace"      # where files get written
DOCS_DIR = PROJECT_ROOT / "data" / "sops"       # RAG source PDFs
RAG_INDEX_DIR = PROJECT_ROOT / "data" / "rag_index"  # persisted index

WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

# Repo root (one level above backend/) — lets tools read project-level
# inputs like data/demo_input and data/sops without exposing the whole fs.
REPO_ROOT = PROJECT_ROOT.parent


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------
def _safe_path(path_str: str) -> Path:
    """Resolve a path but refuse to escape the workspace (basic safety)."""
    raw = Path(path_str)
    if not raw.is_absolute():
        raw = WORKSPACE_DIR / raw
    resolved = raw.resolve()
    # TODO: tighten this per your threat model (symlink attacks, etc).
    if not str(resolved).startswith(str(WORKSPACE_DIR.resolve())):
        raise PermissionError(f"path escapes workspace: {path_str}")
    return resolved


def read_file(path_str: str) -> str:
    """Read and return the contents of a text file in the workspace."""
    path = _safe_path(path_str)
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return f"[error] file not found: {path_str}"


def write_file(path_str: str, content: str) -> str:
    """Write `content` to a file in the workspace (creates parents)."""
    path = _safe_path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"[ok] wrote {len(content)} bytes to {path_str}"


def _safe_repo_path(path_str: str) -> Path:
    """Resolve a path but refuse to escape the repo root (basic safety)."""
    raw = Path(path_str)
    if not raw.is_absolute():
        raw = REPO_ROOT / raw
    resolved = raw.resolve()
    if not str(resolved).startswith(str(REPO_ROOT.resolve())):
        raise PermissionError(f"path escapes project root: {path_str}")
    return resolved


def ocr_image(path_str: str) -> str:
    """Extract text from a local image or scanned PDF via the vision model."""
    path = _safe_repo_path(path_str)
    if not path.exists():
        return f"[error] file not found: {path_str}"
    from backend.ocr.vision import ocr_image as _ocr_image, ocr_pdf as _ocr_pdf

    text = _ocr_pdf(path) if path.suffix.lower() == ".pdf" else _ocr_image(path)
    return f"[ocr] extracted text from {path_str}:\n{text}"


def search_docs(query: str, top_k: int = 3) -> str:
    """Search the local document corpus (RAG) AND remembered facts, and
    return top-k cited results from each, clearly labeled by source."""
    doc_lines: list[str] = []
    try:
        from backend.rag.retrieve import retrieve

        hits = retrieve(query, top_k=top_k)
        for i, hit in enumerate(hits, 1):
            doc_lines.append(
                f"{i}. [doc={hit['doc']} page={hit['page']}] {hit['text'][:400]}"
            )
    except Exception as exc:  # noqa: BLE001 - degrade gracefully
        doc_lines.append(f"[document index unavailable: {exc}]")

    memory_lines: list[str] = []
    try:
        from backend.memory.store import search_memory

        for i, fact in enumerate(search_memory(query, top_k=top_k), 1):
            memory_lines.append(
                f"{i}. {fact['subject']} {fact['predicate']} {fact['object']}"
                f" (remembered {fact['created_at'][:10]})"
            )
    except Exception as exc:  # noqa: BLE001 - degrade gracefully
        memory_lines.append(f"[memory unavailable: {exc}]")

    if not doc_lines and not memory_lines:
        return "[search] no relevant chunks found in the local corpus, and no matching remembered facts."

    parts = []
    parts.append(
        "[search] top document results:\n" + "\n".join(doc_lines)
        if doc_lines else "[search] no relevant document chunks found."
    )
    parts.append(
        "[search] remembered facts:\n" + "\n".join(memory_lines)
        if memory_lines else "[search] no remembered facts match."
    )
    return "\n\n".join(parts)


def run_code(code: str) -> str:
    """Execute `code` in a sandboxed local container and return its output."""
    try:
        from backend.sandbox.executor import run_in_sandbox

        result = run_in_sandbox(code, timeout=30)
        parts = []
        if result["stdout"].strip():
            parts.append(f"[stdout]\n{result['stdout']}")
        if result["stderr"].strip():
            parts.append(f"[stderr]\n{result['stderr']}")
        if result.get("exit_code") != 0 and not parts:
            parts.append(f"[error] exit code {result['exit_code']}")
        return "\n".join(parts) or "[run] produced no output."
    except Exception as exc:  # noqa: BLE001
        return f"[run] could not execute in sandbox: {exc}"


def generate_deliverable(kind: str, payload: dict) -> str:
    """Generate an approval_note / calc_sheet / presentation deliverable.

    Returns the path to the generated file (or an error string).
    """
    try:
        from backend.deliverables import schemas
        from backend.deliverables.generate_docx import render_approval_note
        from backend.deliverables.generate_xlsx import render_calc_sheet
        from backend.deliverables.generate_pptx import render_presentation

        out_dir = WORKSPACE_DIR / "deliverables"
        out_dir.mkdir(parents=True, exist_ok=True)

        if kind == "approval_note":
            note = schemas.ApprovalNote(**payload)
            path = render_approval_note(note, out_dir / "approval_note.docx")
        elif kind == "calc_sheet":
            sheet = schemas.CalcSheet(**payload)
            path = render_calc_sheet(sheet, out_dir / "calc_sheet.xlsx")
        elif kind == "presentation":
            pres = schemas.Presentation(**payload)
            path = render_presentation(pres, out_dir / "presentation.pptx")
        else:
            return f"[error] unknown deliverable kind: {kind!r}"

        return f"[ok] generated deliverable -> {path}"
    except Exception as exc:  # noqa: BLE001
        return f"[error] deliverable generation failed: {exc}"


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------
@dataclass
class Tool:
    name: str
    description: str
    args_schema: str                 # human-readable JSON-ish schema for the LLM
    func: Callable[..., Any]

    def call(self, args_json: str) -> str:
        """Parse JSON args and invoke the tool, returning a string result."""
        try:
            args = json.loads(args_json) if args_json else {}
            out = self.func(**args)
        except TypeError as exc:
            return f"[error] bad arguments for {self.name}: {exc}"
        except Exception as exc:  # noqa: BLE001
            return f"[error] {self.name} raised: {exc}"
        return out if isinstance(out, str) else json.dumps(out, default=str)


# Tools whose effects reach outside the conversation (writing a real file
# the user will actually open/use) and therefore require human approval
# before they run -- see backend/approvals/store.py and the gate in
# backend/agent/loop.py. Read-only tools (search_docs, ocr_image, read_file)
# and the sandboxed run_code (contained by Docker --network none, produces
# no persistent output of its own) are deliberately not gated.
SIDE_EFFECT_TOOLS = {"generate_deliverable", "write_file"}

# Tool registry the agent uses by name.
TOOLS: dict[str, Tool] = {t.name: t for t in [
    Tool("read_file", "Read a file from the workspace.", '{"path_str": "string"}', read_file),
    Tool("write_file", "Write a file in the workspace.", '{"path_str": "string", "content": "string"}', write_file),
    Tool("ocr_image", "Extract text from an image or scanned PDF (vision model). Use this to read inspection report scans/photos.", '{"path_str": "string"}', ocr_image),
    Tool("search_docs", "Search the local SOP/document corpus (RAG).", '{"query": "string", "top_k": 3}', search_docs),
    Tool("run_code", "Run Python code in a sandboxed container.", '{"code": "string"}', run_code),
    Tool(
        "generate_deliverable",
        "Generate a deliverable file. The `payload` shape depends on `kind`:\n"
        '    approval_note: {"title": str, "prepared_by": str, "reference"?: str, "date"?: str, '
        '"summary"?: str, "approval_required"?: bool, "approver_name"?: str, "signature_required"?: bool}\n'
        '    calc_sheet: {"title": str, "tax_rate"?: float (0-1, default 0), "currency"?: str, '
        '"rows"?: [{"item": str, "qty": float, "unit_price": float, "formula"?: str, "note"?: str}]}\n'
        '    presentation: {"title": str, "subtitle"?: str, "author"?: str, '
        '"slides"?: [{"layout": "title|bullets|table", "title": str, "bullets"?: [str], "table"?: [[str]]}]}\n'
        "Every field not marked with a trailing '?' is required. Do not add any field not listed above.",
        '{"kind": "approval_note|calc_sheet|presentation", "payload": {...}}',
        generate_deliverable,
    ),
]}


def describe_tools() -> str:
    """Return a compact description of available tools for the LLM prompt."""
    return "\n".join(
        f"- {t.name}({t.args_schema})\n    {t.description}" for t in TOOLS.values()
    )


def run_tool(name: str, args_json: str) -> str:
    """Public entry point for the agent loop to execute a tool by name."""
    tool = TOOLS.get(name)
    if tool is None:
        return f"[error] unknown tool: {name}"
    return tool.call(args_json)
