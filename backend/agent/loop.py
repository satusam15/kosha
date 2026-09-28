"""ReAct-style agent loop: plan -> call tool -> observe -> iterate.

The loop is a genuine iterate-until-done process (not a single LLM call):

  1. Build a system prompt describing the available tools and the output
     protocol (JSON "actions").
  2. Send the accumulated message history to the locally routed model.
  3. If the model emits an ACTION, execute the tool, append the OBSERVATION,
     and loop again.
  4. When the model emits a FINAL answer (or we hit max iterations), stop.

Everything is local. No external API calls.

TODOs:
  - The parsing in `_extract_action` is deliberately simple (JSON-block
    based). If your local model does not emit clean JSON reliably, tighten
    this or switch to a more robust structured-output strategy.
  - Wire in your own stop-criteria / guardrails as needed.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from backend.router.model_router import Route, route_task, stream_chat
from backend.agent.tools import describe_tools, run_tool

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAX_ITERATIONS = 8          # hard cap on tool calls per request
SYSTEM_PREAMBLE = """You are mrpl-workbench, a local, offline-first agentic assistant.

You have access to these tools:
{tools}

Follow this protocol STRICTLY.

To call a tool, emit exactly one JSON block like this, then STOP generating
immediately — do not write "OBSERVATION" yourself, and do not write more than
one ACTION block per turn. The real tool result will be given to you next.

ACTION
{{"tool": "search_docs", "args": {{"query": "safety harness SOP"}}}}
END_ACTION

After the tool runs, the real OBSERVATION will be given to you. Use it to
plan your next step.

When a document gives you a numeric threshold or range, explicitly state the
actual reading, the threshold/range, and whether the reading is above,
below, or inside it, before concluding a risk level. Apply every escalation
rule stated in the SOP text you retrieved (e.g. "increases risk by one
level") -- do not skip a rule just because it is at the end of a passage.

Only use run_code for actual numeric computation you cannot do yourself.
Do NOT use run_code to "generate" a document, report, or approval note --
use generate_deliverable for that once you have gathered the facts you need
(e.g. via ocr_image and/or search_docs). Reason about risk levels and
recommendations yourself in plain text; you do not need code for that.

Whenever the user asks you to write, produce, or provide Python code (a
function, script, snippet, etc.), you MUST call run_code with that code
BEFORE giving your final answer -- never hand back code you have not run.
Defining a function is not enough: the code you submit to run_code MUST also
CALL the function with concrete sample input(s) and print() the result(s),
so the sandbox observation shows real, checkable output -- "produced no
output" means you verified nothing. If the printed output looks wrong for
the inputs you chose, fix the code and run it again. Your final answer must
be the verified code plus the actual printed output from the sandbox, not a
call to any other tool.

When you have enough information to answer the user, stop and emit a final
answer as a normal paragraph. Do NOT emit an ACTION block if you are ready to
answer.

You may take up to {max_iter} steps. Be concise but complete.
"""


# ---------------------------------------------------------------------------
# Message log / state
# ---------------------------------------------------------------------------
@dataclass
class AgentStep:
    """A single logged step in the agent trace."""

    kind: str          # "plan" | "action" | "observation" | "final"
    content: str
    tool: Optional[str] = None
    args: Optional[dict] = None
    ts: float = field(default_factory=time.time)


@dataclass
class AgentResult:
    answer: str = ""
    steps: list[AgentStep] = field(default_factory=list)
    iterations: int = 0
    routed: Optional[Route] = None


# ---------------------------------------------------------------------------
# Action extraction (JSON-block based)
# ---------------------------------------------------------------------------
_ACTION_RE = re.compile(
    r"ACTION\s*(\{.*?\})\s*END_ACTION",
    flags=re.DOTALL | re.IGNORECASE,
)


_PY_LITERAL_RE = re.compile(r"\b(True|False|None)\b")
_PY_TO_JSON_LITERAL = {"True": "true", "False": "false", "None": "null"}


def _repair_json_brackets(text: str) -> str:
    """Fix a specific, observed small-model mistake: closing a `[...]` array
    with `}` instead of `]` (or vice versa) while getting the bracket
    *count* right. Walks the string with a stack of expected closers
    (skipping string literals) and forces each closer to match what's
    actually open, regardless of which character the model typed there.
    """
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


def _extract_action(text: str) -> Optional[dict]:
    """Return the first parsed action dict, or None if none present."""
    m = _ACTION_RE.search(text)
    if not m:
        return None
    raw = m.group(1)
    candidates = [
        raw,
        _PY_LITERAL_RE.sub(lambda m: _PY_TO_JSON_LITERAL[m.group(0)], raw),
    ]
    candidates.append(_repair_json_brackets(candidates[-1]))

    payload = None
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
            break
        except json.JSONDecodeError:
            continue
    if payload is None:
        return None
    if not isinstance(payload, dict) or "tool" not in payload:
        return None
    payload.setdefault("args", {})
    return _normalize_generate_deliverable_args(payload)


def _normalize_generate_deliverable_args(action: dict) -> dict:
    """generate_deliverable's {"kind": ..., "payload": {...}} shape is the
    one this model most often gets structurally wrong -- e.g. dropping the
    "args" wrapper entirely and leaving "kind" nested one level too deep
    inside a sibling "payload" dict. Scan the action's top-level dict-valued
    fields for something with a "kind" key and reshape it into the args the
    tool actually expects, rather than failing outright.
    """
    if action.get("tool") != "generate_deliverable":
        return action
    args = action.get("args") or {}
    if "kind" in args and "payload" in args:
        return action  # already correctly shaped

    candidates = [args] + [
        v for k, v in action.items()
        if k not in ("tool", "args") and isinstance(v, dict)
    ]
    for cand in candidates:
        if "kind" not in cand:
            continue
        if "payload" in cand:
            action["args"] = {"kind": cand["kind"], "payload": cand["payload"]}
            return action
        payload_fields = {k: v for k, v in cand.items() if k != "kind"}
        if payload_fields:
            action["args"] = {"kind": cand["kind"], "payload": payload_fields}
            return action
    return action


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------
def run_agent(
    user_message: str,
    *,
    has_upload: bool = False,
    history: Optional[list[dict]] = None,
    max_iterations: int = MAX_ITERATIONS,
) -> AgentResult:
    """Run the ReAct loop on a user message and return the final answer.

    Args:
        user_message: the user's request.
        has_upload: True if an image/file was attached (routes to vision).
        history: optional prior messages to keep context across turns.
        max_iterations: safety cap on tool calls.
    """
    route = route_task(user_message, has_upload=has_upload)
    result = AgentResult(routed=route, iterations=0)

    messages: list[dict] = list(history or [])
    if not messages or messages[0].get("role") != "system":
        messages.insert(
            0,
            {
                "role": "system",
                "content": SYSTEM_PREAMBLE.format(
                    tools=describe_tools(), max_iter=max_iterations
                ),
            },
        )
    messages.append({"role": "user", "content": user_message})

    last_action_signature: Optional[str] = None
    repeat_count = 0

    for iteration in range(max_iterations):
        result.iterations = iteration + 1

        # --- PLAN: stream the model's response ---
        plan_parts: list[str] = []
        for chunk in stream_chat(route.model, messages):
            plan_parts.append(chunk)
        plan = "".join(plan_parts).strip()

        # Store a compact plan trace (avoid storing the full model text twice).
        result.steps.append(AgentStep(kind="plan", content=plan[:2000]))

        action = _extract_action(plan)
        if action is None:
            # The model answered without needing a tool -> we are done.
            result.steps.append(AgentStep(kind="final", content=plan))
            result.answer = plan
            return result

        tool_name = action["tool"]
        args = action.get("args", {})
        args_json = json.dumps(args, default=str)
        result.steps.append(
            AgentStep(kind="action", content=f"{tool_name}({args_json})",
                      tool=tool_name, args=args)
        )

        # --- ACT + OBSERVE ---
        observation = run_tool(tool_name, args_json)
        result.steps.append(AgentStep(kind="observation", content=observation[:2000]))

        # Greedy decoding is deterministic: if the exact same call is retried
        # after failing, it will just fail identically forever. Detect that
        # and force a course change instead of burning the rest of the loop.
        signature = f"{tool_name}:{args_json}"
        repeat_count = repeat_count + 1 if signature == last_action_signature else 0
        last_action_signature = signature

        nudge = ""
        if repeat_count >= 1:
            nudge = (
                "\n\nYou already called this exact tool with these exact "
                "arguments and it failed. Repeating it will fail the same "
                "way again. Do NOT call it again — either try a different "
                "tool/approach, or if you already have enough information, "
                "give your final answer or call generate_deliverable now."
            )

        # Feed the observation back into the conversation and loop again.
        messages.append(
            {"role": "assistant", "content": plan[:2000]}
        )
        messages.append(
            {
                "role": "user",
                "content": f"OBSERVATION:\n{observation}\n\n"
                           "Continue. If you are done, give your final answer. "
                           f"Otherwise emit your next ACTION block.{nudge}",
            }
        )

    # Loop exhausted without an explicit final answer.
    result.answer = (
        f"[stopped] reached max iterations ({max_iterations}) before a final "
        "answer. Last model text was appended to the trace."
    )
    return result
