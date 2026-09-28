"""Human-in-the-loop approval queue for side-effecting agent actions.

When the agent wants to call a gated tool (see SIDE_EFFECT_TOOLS in
backend/agent/tools.py -- generate_deliverable, write_file), the loop
does NOT execute it. Instead it records a pending row here and gives the
agent an observation saying the action is queued, not done. The tool call
only actually runs once a human calls decide(..., approve=True) via
POST /approvals/{id}/approve; a reject records the decision and the tool
never runs at all.

Fully local SQLite (data/approvals.db), survives restarts. Separate from
backend/memory/store.py on purpose: that store is the agent's learned
knowledge, this one is a workflow/audit trail of side-effect decisions.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "approvals.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS approvals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    tool TEXT NOT NULL,
    args_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    result TEXT,
    created_at TEXT NOT NULL,
    decided_at TEXT
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(_SCHEMA)


def create_approval(task_id: str, tool: str, args_json: str) -> int:
    """Queue an action for approval. Idempotent per (task_id, tool, args):
    a small model that ignores the "already pending" nudge and retries the
    identical call gets back the SAME pending approval id instead of a new
    duplicate row each time -- one real-world action, one approval request."""
    with _connect() as conn:
        existing = conn.execute(
            "SELECT id FROM approvals WHERE task_id = ? AND tool = ? AND args_json = ? "
            "AND status = 'pending'",
            (task_id, tool, args_json),
        ).fetchone()
        if existing:
            return existing["id"]
        cur = conn.execute(
            "INSERT INTO approvals (task_id, tool, args_json, status, created_at) "
            "VALUES (?, ?, ?, 'pending', ?)",
            (task_id, tool, args_json, _now()),
        )
        return cur.lastrowid


def get_approval(approval_id: int) -> Optional[dict]:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM approvals WHERE id = ?", (approval_id,)
        ).fetchone()
    return dict(row) if row else None


def list_approvals(status: Optional[str] = None) -> list[dict]:
    with _connect() as conn:
        if status:
            rows = conn.execute(
                "SELECT * FROM approvals WHERE status = ? ORDER BY id DESC", (status,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM approvals ORDER BY id DESC").fetchall()
    return [dict(r) for r in rows]


def decide(approval_id: int, *, approve: bool) -> dict:
    """Approve or reject a pending action. Approving is the ONLY moment the
    original tool call actually executes -- rejecting never runs it. Raises
    ValueError for an unknown id or one that's already been decided (no
    silent re-deciding)."""
    row = get_approval(approval_id)
    if row is None:
        raise ValueError(f"no such approval: {approval_id}")
    if row["status"] != "pending":
        raise ValueError(f"approval {approval_id} already {row['status']}")

    if approve:
        from backend.agent.tools import run_tool

        result_text = run_tool(row["tool"], row["args_json"])
        status = "approved"
    else:
        result_text = "[rejected] action was not executed."
        status = "rejected"

    with _connect() as conn:
        conn.execute(
            "UPDATE approvals SET status = ?, result = ?, decided_at = ? WHERE id = ?",
            (status, result_text, _now(), approval_id),
        )

    try:
        from backend.agent.execlog import log_step

        log_step(row["task_id"], 0, f"approval_{status}", result_text, tool=row["tool"])
    except Exception:  # noqa: BLE001 - logging must never break a decision
        pass

    return get_approval(approval_id)


init_db()
