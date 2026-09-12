"""
Console UI helpers - colored, plain-language output.

Everything the user sees goes through here so the tone is consistent and every
error is actionable. Colors auto-disable when output is not a TTY (e.g. piped to
a file or a CI log) or when NO_COLOR is set.
"""

from __future__ import annotations

import os
import sys

_USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None

_GREEN = "\033[0;32m" if _USE_COLOR else ""
_YELLOW = "\033[1;33m" if _USE_COLOR else ""
_RED = "\033[0;31m" if _USE_COLOR else ""
_BLUE = "\033[0;34m" if _USE_COLOR else ""
_BOLD = "\033[1m" if _USE_COLOR else ""
_NC = "\033[0m" if _USE_COLOR else ""


def info(msg: str) -> None:
    print(f"{_GREEN}[ok]{_NC}   {msg}")


def step(msg: str) -> None:
    print(f"{_BLUE}[..]{_NC}   {msg}")


def warn(msg: str) -> None:
    print(f"{_YELLOW}[warn]{_NC} {msg}")


def error(msg: str) -> None:
    # Written to stdout on purpose: on Windows PowerShell, text on stderr is
    # rendered as a red "NativeCommandError" record, which looks alarming even
    # for a handled, expected error. We keep all user-facing output on stdout so
    # it reads cleanly; the process exit code still signals success/failure.
    print(f"{_RED}[error]{_NC} {msg}")


def hint(msg: str) -> None:
    """A 'do this next' line - the most important part of any failure."""
    print(f"{_YELLOW}   -> {msg}{_NC}")


def banner(title: str) -> None:
    line = "=" * 66
    print(f"\n{_BOLD}{line}\n  {title}\n{line}{_NC}")


def kv(key: str, value: str) -> None:
    print(f"  {key:<18}: {value}")


def bold(text: str) -> str:
    return f"{_BOLD}{text}{_NC}"


class OaiError(Exception):
    """A user-facing error. `hint` is the actionable next step (may be multi-line)."""

    def __init__(self, message: str, hint_text: str | None = None) -> None:
        super().__init__(message)
        self.hint_text = hint_text


def fail(message: str, hint_text: str | None = None) -> None:
    """Raise a user-facing error with an optional actionable hint."""
    raise OaiError(message, hint_text)
