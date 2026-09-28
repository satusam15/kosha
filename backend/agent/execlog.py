"""Append-only JSONL execution log: one line per agent step.

Every Thought/Action/Observation/Final(/Stopped) step from every task is
written here as it happens -- not batched up and flushed at the end of a
task -- so the log reflects real execution order and still has a record
of a task even if it never reaches a clean final answer (crash, or the
iteration cap).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_PATH = PROJECT_ROOT / "backend" / "workspace" / "logs" / "execution.jsonl"


def log_step(task_id: str, step_index: int, kind: str, content: str,
             tool: Optional[str] = None, args: Optional[dict] = None) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "task_id": task_id,
        "step_index": step_index,
        "kind": kind,
        "tool": tool,
        "args": args,
        "content": content,
    }
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
