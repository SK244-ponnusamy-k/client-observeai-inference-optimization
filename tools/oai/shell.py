"""
Helpers for invoking external tooling (bash, kubectl, aws) with friendly errors.

The generated deploy/stop/download scripts are bash. On Windows we run them via
`bash` (Git Bash / WSL). If bash is missing we say so clearly instead of failing
with a cryptic OS error.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from . import paths, ui


def have(tool: str) -> bool:
    return shutil.which(tool) is not None


def require_tool(tool: str, why: str) -> None:
    if not have(tool):
        install = {
            "bash": "Install Git for Windows (git-scm.com) or enable WSL, then re-run.",
            "kubectl": "Install kubectl and point it at the cluster (aws eks update-kubeconfig ...).",
            "aws": "Install AWS CLI v2 and configure credentials for the account.",
            "envsubst": "Install gettext (provides envsubst); it ships with Git Bash.",
        }.get(tool, f"Install '{tool}' and re-run.")
        ui.fail(f"'{tool}' is required to {why}, but it was not found on PATH.", install)


def run_bash(script: Path, args: list[str] | None = None, env: dict[str, str] | None = None) -> int:
    """Run a bash script from the framework root, streaming its output live."""
    require_tool("bash", f"run {script.name}")
    args = args or []
    import os

    merged = os.environ.copy()
    if env:
        merged.update(env)
    # Use a POSIX-style path so bash on Windows resolves it correctly.
    posix = script.as_posix()
    cmd = ["bash", posix, *args]
    ui.step(f"Running: bash {script.relative_to(paths.ROOT).as_posix()} {' '.join(args)}".rstrip())
    proc = subprocess.run(cmd, cwd=str(paths.ROOT), env=merged, check=False)  # noqa: S603
    return proc.returncode


def kubectl_json(args: list[str]) -> str | None:
    """Run a kubectl command returning stdout, or None if kubectl/call fails."""
    if not have("kubectl"):
        return None
    try:
        proc = subprocess.run(  # noqa: S603
            ["kubectl", *args], capture_output=True, text=True, timeout=30, check=False
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    return proc.stdout if proc.returncode == 0 else None
