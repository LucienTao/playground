"""Detect whether a dialogue context has switched topics."""

from __future__ import annotations
from ..utils.text import tokenize, jaccard

def is_context_switch(current: str, previous_summary: str, threshold: float) -> bool:
    """Return ``True`` if similarity is below ``threshold``."""
    current_tokens = tokenize(current)
    prev_tokens = tokenize(previous_summary)
    sim = jaccard(current_tokens, prev_tokens)

    return sim < threshold
