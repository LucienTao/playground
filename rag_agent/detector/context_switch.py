"""Detect whether a dialogue context has switched topics."""

from __future__ import annotations

from ..memory.memory_store import _tokenize, _jaccard


def is_context_switch(current: str, previous_summary: str, threshold: float) -> bool:
    """Return ``True`` if similarity is below ``threshold``.

    Similarity is computed using Jaccard similarity of token sets.
    """
    current_tokens = _tokenize(current)
    prev_tokens = _tokenize(previous_summary)
    sim = _jaccard(current_tokens, prev_tokens)
    return sim < threshold
