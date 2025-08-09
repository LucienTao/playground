"""Text helpers: tokenization, similarity and embeddings."""

from __future__ import annotations

import hashlib
import math
import random
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


def _token_embedding(token: str, dim: int) -> List[float]:
    """Return a pseudo-random embedding vector for a token."""
    seed = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
    rng = random.Random(seed)
    return [rng.uniform(-1.0, 1.0) for _ in range(dim)]


def model_embedding(text: str, dim: int = _DEF_DIM) -> List[float]:
    """Embed text by averaging deterministic token embeddings."""
    tokens = tokenize(text)
    if not tokens:
        return [0.0] * dim
    vecs = [_token_embedding(tok, dim) for tok in tokens]
    avg = [sum(vals) / len(tokens) for vals in zip(*vecs)]
    norm = math.sqrt(sum(v * v for v in avg))
    if norm > 0:
        avg = [v / norm for v in avg]
    return avg
