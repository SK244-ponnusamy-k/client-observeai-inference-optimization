"""
Tiny, dependency-free template renderer.

We deliberately avoid Jinja2 to add no new dependencies. Templates use Python's
str.format via a thin wrapper: placeholders are written as {{name}} so that the
generated files can freely contain shell/bash ${VAR} and Kubernetes syntax
without escaping. Only our own {{...}} markers are substituted.

Rendering is intentionally strict: an unknown placeholder raises, so a template
typo fails loudly at generate time rather than producing a broken YAML.
"""

from __future__ import annotations

import re
from typing import Any

_PLACEHOLDER = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")


def render(template: str, context: dict[str, Any]) -> str:
    """Replace every {{name}} in `template` with str(context[name])."""
    missing: list[str] = []

    def repl(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in context:
            missing.append(key)
            return match.group(0)
        return str(context[key])

    result = _PLACEHOLDER.sub(repl, template)
    if missing:
        uniq = sorted(set(missing))
        raise KeyError(f"Template referenced unknown placeholder(s): {', '.join(uniq)}")
    return result
