"""
Helpers for invoking external tooling (bash, kubectl, aws) with friendly errors.

The generated deploy/stop/download scripts are bash. On Windows we run them via
a REAL bash (Git Bash / WSL). A common trap on Windows: the first `bash` on PATH
is `C:\\Windows\\System32\\bash.exe`, the WSL launcher stub. When WSL has no
distro installed it fails with "WSL ... /bin/bash: No such file or directory".
_resolve_bash() deliberately skips that stub and prefers a real Git Bash.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from . import paths, ui

# Windows "bash" launcher stubs we must NOT use to run the framework's scripts:
#   - C:\Windows\System32\bash.exe        (WSL launcher; fails with no distro)
#   - ...\WindowsApps\bash.exe            (Microsoft Store app-execution alias)
_STUB_MARKERS = ("system32\\bash", "windowsapps\\bash")

# Preferred real Git Bash locations. Checked BEFORE PATH so we never pick a stub.
_GIT_BASH_CANDIDATES = (
    r"C:\Program Files\Git\bin\bash.exe",
    r"C:\Program Files\Git\usr\bin\bash.exe",
    r"C:\Program Files (x86)\Git\bin\bash.exe",
)


def _is_stub(path: str) -> bool:
    p = path.replace("/", "\\").lower()
    return any(marker in p for marker in _STUB_MARKERS)


def have(tool: str) -> bool:
    return shutil.which(tool) is not None


def have_in_bash(tool: str) -> bool:
    """
    True if `tool` is callable inside the resolved bash.

    On Windows, tools like helm/kubectl/eksctl often live only on Git Bash's PATH,
    not the Windows PATH that this Python process sees. Since the framework's
    scripts RUN inside that bash, the honest availability check is 'can bash find
    it', not 'can shutil.which find it'.
    """
    bash = _resolve_bash()
    if bash is None or _is_stub(bash):
        return False
    try:
        proc = subprocess.run(  # noqa: S603
            [bash, "-lc", f"command -v {tool}"],
            capture_output=True, text=True, timeout=15, check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return proc.returncode == 0 and bool(proc.stdout.strip())


def _resolve_bash() -> str | None:
    """
    Return a path to a REAL bash, or None if only the WSL stub / nothing exists.

    Order: OAI_BASH override -> a PATH bash that is NOT the WSL stub -> a known
    Git Bash install location.
    """
    override = os.environ.get("OAI_BASH")
    if override and Path(override).exists():
        return override

    # Prefer a known real Git Bash BEFORE trusting PATH (PATH often exposes only
    # the WSL / Store stubs first on Windows).
    for cand in _GIT_BASH_CANDIDATES:
        if Path(cand).exists():
            return cand

    # Then any PATH bash that is not a known stub.
    for found in _all_on_path("bash"):
        if not _is_stub(found):
            return found

    # Last resort: whatever `bash` resolves to (may be a stub — caller handles failure).
    which = shutil.which("bash")
    return None if (which and _is_stub(which)) else which


def _all_on_path(tool: str) -> list[str]:
    """All matches for `tool` on PATH, in order (shutil.which returns only the first)."""
    results: list[str] = []
    exts = os.environ.get("PATHEXT", ".EXE").split(os.pathsep) if os.name == "nt" else [""]
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        if not directory:
            continue
        base = Path(directory) / tool
        for ext in ([""] + exts if os.name == "nt" else [""]):
            cand = str(base) + ext
            if Path(cand).is_file() and cand not in results:
                results.append(cand)
    return results


# Tools that the framework's bash scripts invoke themselves. For these, the real
# question is whether the RESOLVED BASH can find them (Git Bash PATH), not whether
# this Windows Python process can. So we also check inside bash before failing.
_BASH_RUN_TOOLS = ("kubectl", "eksctl", "helm", "envsubst", "aws")


def require_tool(tool: str, why: str) -> None:
    if tool == "bash":
        ok = _resolve_bash() is not None
    elif tool in _BASH_RUN_TOOLS:
        ok = have(tool) or have_in_bash(tool)
    else:
        ok = have(tool)
    if not ok:
        install = {
            "bash": "Install Git for Windows (git-scm.com), or set OAI_BASH to your bash.exe, then re-run.",
            "kubectl": "Install kubectl and point it at the cluster (aws eks update-kubeconfig ...).",
            "aws": "Install AWS CLI v2 and configure credentials for the account.",
            "helm": "Install Helm (helm.sh) and ensure it is on your Git Bash PATH.",
            "eksctl": "Install eksctl (eksctl.io) and ensure it is on your Git Bash PATH.",
            "envsubst": "Install gettext (provides envsubst); it ships with Git Bash.",
        }.get(tool, f"Install '{tool}' and re-run.")
        ui.fail(f"'{tool}' is required to {why}, but it was not found.", install)


def run_bash(script: Path, args: list[str] | None = None, env: dict[str, str] | None = None) -> int:
    """Run a bash script from the framework root, streaming its output live."""
    bash = _resolve_bash()
    if bash is None or _is_stub(bash):
        ui.fail(
            "No usable bash found (only the WSL stub is on PATH).",
            "Install Git for Windows, or set OAI_BASH to its bash.exe, e.g.\n"
            "   -> export OAI_BASH='/c/Program Files/Git/bin/bash.exe'",
        )
    args = args or []

    merged = os.environ.copy()
    if env:
        merged.update(env)
    # Use a POSIX-style path so bash resolves the script correctly.
    posix = script.as_posix()
    cmd = [bash, posix, *args]
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
