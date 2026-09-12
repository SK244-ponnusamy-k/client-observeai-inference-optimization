"""
Reader for config/config.env.

config.env is a bash file (`export KEY="value"`). Rather than execute it (which
would require bash on Windows), we parse the `export` lines directly and resolve
simple ${VAR:-default} / ${VAR} references against what we have already parsed
plus the current process environment. This is intentionally a small, safe subset
of shell expansion - enough for this config file, with no command execution.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache

from . import paths

_EXPORT_RE = re.compile(r'^\s*export\s+([A-Za-z_][A-Za-z0-9_]*)=(.*)$')
_VAR_RE = re.compile(r'\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}')


def _strip_inline_comment(value: str) -> str:
    """Remove a trailing ' # comment' that is outside of quotes."""
    out = []
    in_single = in_double = False
    i = 0
    while i < len(value):
        ch = value[i]
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            # comment starts only if preceded by whitespace or at column start
            if i == 0 or value[i - 1].isspace():
                break
        out.append(ch)
        i += 1
    return "".join(out).strip()


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


class Config:
    """Parsed config.env with dict-like access and safe .get()."""

    def __init__(self, values: dict[str, str]) -> None:
        self._values = values

    def get(self, key: str, default: str | None = None) -> str | None:
        return self._values.get(key, os.environ.get(key, default))

    def require(self, key: str) -> str:
        val = self.get(key)
        if val is None or val == "":
            from .ui import OaiError

            raise OaiError(
                f"Required setting '{key}' is not set in config/config.env.",
                f"Open config/config.env and set {key}.",
            )
        return val

    def as_dict(self) -> dict[str, str]:
        return dict(self._values)


@lru_cache(maxsize=1)
def load() -> Config:
    """Parse config/config.env once and cache it."""
    values: dict[str, str] = {}
    path = paths.CONFIG_ENV
    if not path.exists():
        return Config(values)

    for raw in path.read_text(encoding="utf-8").splitlines():
        m = _EXPORT_RE.match(raw)
        if not m:
            continue
        key, rhs = m.group(1), m.group(2)
        rhs = _strip_inline_comment(rhs)
        rhs = _unquote(rhs)

        def repl(mo: re.Match[str]) -> str:
            var, dflt = mo.group(1), mo.group(2)
            if var in values:
                return values[var]
            if var in os.environ:
                return os.environ[var]
            return dflt if dflt is not None else ""

        rhs = _VAR_RE.sub(repl, rhs)
        values[key] = rhs

    return Config(values)
