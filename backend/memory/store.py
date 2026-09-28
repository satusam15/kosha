"""Persistent memory: a local SQLite store for facts, entities, and
relations extracted from agent tasks.

Fully local -- a single file on disk (`data/memory.db`), no external
services, no in-memory-only mode. It survives process restarts because it
is a real file that SQLite opens fresh each connection; nothing here holds
state only in Python-process memory. Facts are indexed with SQLite's FTS5
full-text search so they can be retrieved alongside document search (see
`search_docs` in backend/agent/tools.py, which merges memory hits into its
results).

Checkpoint 1 (persistent memory) and checkpoint 2 (cross-session recall)
both live here: recall is "cross-session" simply because nothing about
retrieval is scoped to a session id -- every fact ever saved is searchable
for as long as the database file exists.
"""

from __future__ import annotations

import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from backend.util.json_repair import parse_json_loose

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "memory.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    type TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    source TEXT,
    task_id TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS relations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_entity_id INTEGER NOT NULL REFERENCES entities(id),
    predicate TEXT NOT NULL,
    object_entity_id INTEGER NOT NULL REFERENCES entities(id),
    created_at TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
    subject, predicate, object, content='facts', content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
    INSERT INTO facts_fts(rowid, subject, predicate, object)
    VALUES (new.id, new.subject, new.predicate, new.object);
END;
CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, subject, predicate, object)
    VALUES ('delete', old.id, old.subject, old.predicate, old.object);
END;
CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, subject, predicate, object)
    VALUES ('delete', old.id, old.subject, old.predicate, old.object);
    INSERT INTO facts_fts(rowid, subject, predicate, object)
    VALUES (new.id, new.subject, new.predicate, new.object);
END;
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
    """Create tables/indexes if they don't exist yet. Idempotent, safe to
    call on every process start."""
    with _connect() as conn:
        conn.executescript(_SCHEMA)


def add_entity(name: str, type_: Optional[str] = None) -> int:
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO entities (name, type, created_at) VALUES (?, ?, ?) "
            "ON CONFLICT(name) DO UPDATE SET type=COALESCE(excluded.type, entities.type) "
            "RETURNING id",
            (name, type_, _now()),
        )
        return cur.fetchone()["id"]


def add_fact(subject: str, predicate: str, object_: str, *,
             source: Optional[str] = None, task_id: Optional[str] = None) -> int:
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO facts (subject, predicate, object, source, task_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (subject, predicate, object_, source, task_id, _now()),
        )
        return cur.lastrowid


def add_relation(subject_entity: str, predicate: str, object_entity: str) -> int:
    subj_id = add_entity(subject_entity)
    obj_id = add_entity(object_entity)
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO relations (subject_entity_id, predicate, object_entity_id, created_at) "
            "VALUES (?, ?, ?, ?)",
            (subj_id, predicate, obj_id, _now()),
        )
        return cur.lastrowid


def search_memory(query: str, top_k: int = 5) -> list[dict]:
    """Full-text search over stored facts. Returns [] on no match or an
    empty store -- callers should treat that the same as an empty document
    search result, not an error."""
    if not query.strip():
        return []
    # FTS5 query syntax needs each term double-quoted to be treated as a
    # literal token rather than parsed as FTS5 query syntax (which would
    # choke on punctuation in arbitrary user text).
    terms = [w.strip('"') for w in query.split() if w.strip('"')]
    if not terms:
        return []
    fts_query = " OR ".join(f'"{w}"' for w in terms)
    with _connect() as conn:
        try:
            rows = conn.execute(
                "SELECT f.subject, f.predicate, f.object, f.source, f.task_id, "
                "f.created_at, bm25(facts_fts) AS score "
                "FROM facts_fts JOIN facts f ON f.id = facts_fts.rowid "
                "WHERE facts_fts MATCH ? ORDER BY score LIMIT ?",
                (fts_query, top_k),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
    return [dict(r) for r in rows]


def all_facts(limit: int = 200) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT subject, predicate, object, source, task_id, created_at "
            "FROM facts ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def all_entities(limit: int = 200) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT name, type, created_at FROM entities ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def all_relations(limit: int = 200) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT e1.name AS subject, r.predicate, e2.name AS object, r.created_at "
            "FROM relations r "
            "JOIN entities e1 ON e1.id = r.subject_entity_id "
            "JOIN entities e2 ON e2.id = r.object_entity_id "
            "ORDER BY r.id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


_EXTRACTION_PROMPT = """From the exchange below, extract any durable facts worth remembering for \
future conversations -- names, IDs, stated preferences, decisions, thresholds the user gave. Do \
NOT extract generic chit-chat, and do not extract things that are only true of this one document \
(a specific reading, a specific incident's details) unless the user stated them as a general fact \
to remember. If there is nothing worth remembering, return an empty list.

Respond with ONLY a JSON array, nothing else. Each item shaped exactly like:
{{"subject": "...", "predicate": "...", "object": "..."}}

User said: {user_message}
Assistant answered: {answer}

JSON array:"""

_ARRAY_RE = re.compile(r"\[.*\]", flags=re.DOTALL)


def extract_and_save_facts(user_message: str, answer: str, *,
                            model: str, base_url: str, task_id: Optional[str] = None) -> list[dict]:
    """Ask the local model to pull durable facts out of one finished task
    and persist them. Meant to be called once per task (checkpoint 1: "saved
    after each task"). Never raises -- a bad extraction should not fail the
    user's actual request.
    """
    import ollama

    prompt = _EXTRACTION_PROMPT.format(
        user_message=user_message[:1500], answer=answer[:1500]
    )
    try:
        client = ollama.Client(host=base_url)
        resp = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0},
        )
        text = resp["message"]["content"]
    except Exception:
        return []

    m = _ARRAY_RE.search(text)
    if not m:
        return []
    parsed = parse_json_loose(m.group(0))
    if not isinstance(parsed, list):
        return []

    saved = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        subj, pred, obj = item.get("subject"), item.get("predicate"), item.get("object")
        if not (isinstance(subj, str) and isinstance(pred, str) and isinstance(obj, str)):
            continue
        subj, pred, obj = subj.strip(), pred.strip(), obj.strip()
        if not (subj and pred and obj):
            continue
        add_fact(subj, pred, obj, source="chat", task_id=task_id)
        saved.append({"subject": subj, "predicate": pred, "object": obj})
    return saved


init_db()
