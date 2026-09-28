"""Runs generated code in Docker with no network, bounded memory.

Command shape:
    docker run --rm --network none --memory=512m <image> python -c <code>

Captures stdout/stderr and returns them to the caller. Falls back to a
local subprocess (still network-isolated via a manual warning) if Docker
is unavailable — this is only a dev convenience, NOT a substitute for a
real sandbox.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass

# TODO: the image must be present locally (`docker pull python:3.11-slim`)
#       for this to work offline.
SANDBOX_IMAGE = "python:3.11-slim"
MEMORY_LIMIT = "512m"
DEFAULT_TIMEOUT = 30  # seconds


@dataclass
class ExecResult:
    exit_code: int
    stdout: str
    stderr: str


def _docker_available() -> bool:
    return shutil.which("docker") is not None


def run_in_sandbox(code: str, timeout: int = DEFAULT_TIMEOUT) -> dict:
    """Run `code` in a Docker sandbox and return {"exit_code","stdout","stderr"}."""
    if _docker_available():
        try:
            return _run_docker(code, timeout)
        except Exception as exc:  # noqa: BLE001
            # Fall through to degraded local mode but surface the reason.
            return _run_local(code, timeout, reason=str(exc))

    # No Docker: run locally with a hard warning (NOT network-isolated).
    return _run_local(code, timeout)


def _run_docker(code: str, timeout: int) -> dict:
    cmd = [
        "docker", "run", "--rm",
        "--network", "none",           # no outbound/inbound networking
        "--memory", MEMORY_LIMIT,
        "--cpus", "1",
        SANDBOX_IMAGE,
        "python", "-c", code,
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return {
        "exit_code": proc.returncode,
        "stdout": proc.stdout or "",
        "stderr": proc.stderr or "",
        "sandbox": "docker",
    }


def _run_local(code: str, timeout: int, reason: str = "docker not installed") -> dict:
    """WARN: local execution is NOT isolated. Only for dev/demo convenience."""
    proc = subprocess.run(
        ["python", "-c", code],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return {
        "exit_code": proc.returncode,
        "stdout": proc.stdout or "",
        "stderr": proc.stderr or "",
        "sandbox": "local (UNSAFE fallback)",
        "warning": (
            f"ran without Docker isolation because: {reason}. "
            "Only use for trusted local experimentation."
        ),
    }


def check_sandbox_health() -> dict:
    """Return a small health dict used by the FastAPI /sandbox endpoint."""
    if not _docker_available():
        return {"docker": False, "message": "docker CLI not found on PATH"}
    try:
        proc = subprocess.run(
            ["docker", "image", "inspect", SANDBOX_IMAGE],
            capture_output=True, text=True, timeout=10,
        )
        image_ready = proc.returncode == 0
    except Exception:  # noqa: BLE001
        image_ready = False
    return {
        "docker": True,
        "image": SANDBOX_IMAGE,
        "image_pulled": image_ready,
        "message": "image ready" if image_ready
                   else f"TODO: run `docker pull {SANDBOX_IMAGE}`",
    }
