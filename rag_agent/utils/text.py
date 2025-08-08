"""Text helpers: tokenization, similarity and embeddings."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Iterable, List


def tokenize(text: str) -> List[str]:
    """Return a list of lowercase tokens (Chinese-aware)."""
    tokens = re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z]+", text)
    return [t.lower() for t in tokens]


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    """Jaccard similarity between two token iterables."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)

_DEF_DIM = 128


def hash_embedding(text: str, dim: int = _DEF_DIM) -> List[float]:
    """Compute a deterministic embedding by hashing tokens into a vector."""
    vec = [0.0] * dim
    for tok in tokenize(text):
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec
