"""Small-model JSON parsing helpers.

Small local LLMs frequently emit near-valid JSON: Python literals
(True/False/None) instead of JSON ones, or a closing bracket of the wrong
type (e.g. closing a `[...]` array with `}`) while getting the bracket
*count* right. These helpers give parsing a few honest extra attempts
before giving up, instead of requiring perfect output on the first try.

Deliberately standalone (not imported by backend/agent/loop.py, which has
its own copy of this exact logic) so that reusing it elsewhere can never
change the already-tested agent loop's behavior.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

_PY_LITERAL_RE = re.compile(r"\b(True|False|None)\b")
_PY_TO_JSON_LITERAL = {"True": "true", "False": "false", "None": "null"}


def fix_py_literals(text: str) -> str:
    return _PY_LITERAL_RE.sub(lambda m: _PY_TO_JSON_LITERAL[m.group(0)], text)


def repair_brackets(text: str) -> str:
    """Force each closing bracket to match what's actually open, regardless
    of which character the model typed there -- fixes the common small-model
    mistake of closing a `[...]` array with `}` while getting the overall
    bracket count right."""
    stack: list[str] = []
    out: list[str] = []
    in_string = False
    escape = False
    for ch in text:
        if in_string:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
        elif ch == "{":
            stack.append("}")
            out.append(ch)
        elif ch == "[":
            stack.append("]")
            out.append(ch)
        elif ch in "}]":
            out.append(stack.pop() if stack else ch)
        else:
            out.append(ch)
    return "".join(out)


def parse_json_loose(text: str) -> Optional[Any]:
    """Try increasingly-repaired candidates; return the first that parses,
    or None if none of them do."""
    candidates = [text, fix_py_literals(text)]
    candidates.append(repair_brackets(candidates[-1]))
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None
