"""Small helpers shared by the package (no torch import)."""

from __future__ import annotations

import difflib
from collections.abc import Collection

__all__ = ["suggest"]


def suggest(name: str, choices: Collection[str]) -> str:
    """Return " (did you mean 'x'?)" for the closest choice, or an empty string."""
    close = difflib.get_close_matches(name, list(choices), n=1, cutoff=0.6)
    return f" (did you mean {close[0]!r}?)" if close else ""
