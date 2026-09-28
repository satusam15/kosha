"""Keyword-based local model router.

Picks which local Ollama model to use based on the nature of the task,
*without* calling any external service. Everything talks to
`http://localhost:11434` only.

Routing rules:
  * "code" / "script" / "function" / "debug" / "test" keywords -> coder model
  * image / file / scanned-document uploads                 -> vision model
  * everything else                                         -> general reasoning model

TODOs:
  - Plug in the *actual* model names you have pulled into Ollama.
    e.g. `ollama pull qwen2.5-coder:7b`, `ollama pull moondream`,
    `ollama pull qwen2.5:7b`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# TODO: replace these placeholders with model names installed in your Ollama.
#        Run `ollama list` to see what you actually have pulled locally.
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = "http://localhost:11434"
CODER_MODEL = "qwen2.5-coder:1.5b"          # TODO: your local coder model
VISION_MODEL = "moondream:latest"                # TODO: your local vision model (moondream / qwen2-vl)
REASONING_MODEL = "qwen2.5:3b-instruct-q4_K_M"            # TODO: your local general reasoning model

# Keywords that push a task toward the coding model.
_CODER_KEYWORDS = [
    "code", "script", "function", "debug", "debugging", "implement",
    "refactor", "fix bug", "exception", "traceback", "regex", "algorithm",
    "test case", "unit test", "function definition", "snippet",
]

# Image/file-like words that push a task toward the vision model.
_VISION_KEYWORDS = [
    "image", "photo", "picture", "scanned", "scan", "ocr", "read the text",
    "extract text", "screenshot", "diagram", "figure", "document scan",
    "png", "jpg", "jpeg", "pdf scan", "handwritten",
]


@dataclass(frozen=True)
class Route:
    """The result of routing a task to a model."""

    task_type: str          # one of: "code", "vision", "general"
    model: str
    reason: str


def _has_any(text: str, keywords: list[str]) -> bool:
    lowered = text.lower()
    return any(k in lowered for k in keywords)


def route_task(message: str, *, has_upload: bool = False) -> Route:
    """Decide which local model should handle a task.

    Args:
        message: the raw user message / task description.
        has_upload: True when the request carries an image or file attachment
                    (set by the caller after inspecting the request).

    Returns:
        A Route describing the chosen model and why.
    """
    # 1) Vision: explicit uploads always win.
    #
    # NOTE: the orchestrating model here is still REASONING_MODEL, not
    # VISION_MODEL. The agent loop is a ReAct tool-calling loop (see
    # backend/agent/loop.py) that needs a model able to follow the
    # ACTION/END_ACTION JSON protocol; moondream is a small captioning/VQA
    # model that cannot do that; it free-associates a generic answer and
    # never emits a tool call. Moondream is invoked correctly elsewhere,
    # inside the `ocr_image` tool (backend/agent/tools.py), which the
    # reasoning model calls when it needs to read an image. `task_type`
    # stays "vision" purely as a label for observability/UI.
    if has_upload:
        return Route(
            task_type="vision",
            model=REASONING_MODEL,
            reason="user attached an image/scanned file -> reasoning model "
                   "(will call the ocr_image tool for the actual vision work)",
        )

    # 2) Vision keywords (even without an upload) next.
    if _has_any(message, _VISION_KEYWORDS):
        return Route(
            task_type="vision",
            model=REASONING_MODEL,
            reason="task mentions image/OCR content -> reasoning model "
                   "(will call the ocr_image tool for the actual vision work)",
        )

    # 3) Coding keywords next.
    if _has_any(message, _CODER_KEYWORDS):
        return Route(
            task_type="code",
            model=CODER_MODEL,
            reason="task mentions code/script/function -> coder model",
        )

    # 4) Everything else -> general reasoning.
    return Route(
        task_type="general",
        model=REASONING_MODEL,
        reason="general reasoning task -> default model",
    )


def stream_chat(model: str, messages: list[dict], *, base_url: str = OLLAMA_BASE_URL):
    """Yield text chunks from a local Ollama chat completion (streaming).

    Uses only `localhost`. Thin wrapper over the official `ollama` python
    client so the rest of the code base never touches an external host.

    TODO: verify the model name in `model` is actually pulled locally,
          otherwise this raises `ResponseError`.
    """
    import ollama  # the `ollama` python client (see requirements.txt)

    client = ollama.Client(host=base_url)  # localhost only
    stream = client.chat(
        model=model,
        messages=messages,
        stream=True,
        # Greedy decoding: this loop depends on the model reliably emitting
        # well-formed ACTION/JSON blocks. Sampling at higher temperature
        # measurably increases malformed/off-task tool calls on small models.
        options={"temperature": 0},
    )
    for chunk in stream:
        delta = chunk.get("message", {}).get("content", "")
        if delta:
            yield delta
