"""Naive text summarization utilities."""

import re


def summarize(text: str, max_sentences: int = 3) -> str:
    """Return the first ``max_sentences`` sentences from ``text``.

    This is a placeholder summarizer which simply splits the text by sentence
    boundaries and returns the first few sentences.  It is deterministic and
    requires no external models.
    """

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(sentences[:max_sentences])
