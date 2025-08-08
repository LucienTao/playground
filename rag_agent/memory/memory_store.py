"""Simple JSON-based memory store for the agent.

The store keeps a list of conversation "memory chunks" each containing a
summary, the original content and a naive bag-of-words embedding.  This is
sufficient for demonstrating retrieval and context switch detection without
requiring heavy ML dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Iterable
import json
import time


def _tokenize(text: str) -> List[str]:
    """Return a list of lowercase tokens (Chinese-aware)."""
    import re
    tokens = re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z]+", text)
    return [t.lower() for t in tokens]


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


@dataclass
class MemoryChunk:
    id: str
    timestamp: float
    summary: str
    content: str
    topic: str
    embedding: List[str] = field(default_factory=list)


class MemoryStore:
    """Persist memories to a JSON file and allow simple similarity search."""

    def __init__(self, path: Path) -> None:
        self.path = path
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            self.memories: List[MemoryChunk] = [MemoryChunk(**m) for m in data]
        else:
            self.memories = []

    # ------------------------------------------------------------------
    def add_memory(self, summary: str, content: str, topic: str) -> MemoryChunk:
        chunk_id = f"chunk-{int(time.time())}-{len(self.memories)+1:03d}"
        embedding = _tokenize(summary)
        chunk = MemoryChunk(
            id=chunk_id,
            timestamp=time.time(),
            summary=summary,
            content=content,
            topic=topic,
            embedding=embedding,
        )
        self.memories.append(chunk)
        self.save()
        return chunk

    # ------------------------------------------------------------------
    def retrieve(self, query: str, top_k: int = 3) -> List[MemoryChunk]:
        tokens = _tokenize(query)
        scored = [
            ( _jaccard(tokens, m.embedding), m) for m in self.memories
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for score, m in scored[:top_k] if score > 0]

    # ------------------------------------------------------------------
    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            json.dump([m.__dict__ for m in self.memories], f, ensure_ascii=False, indent=2)
