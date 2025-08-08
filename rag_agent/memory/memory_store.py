"""Memory store backed by a FAISS index."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

from ..utils.text import hash_embedding
from ..utils.faiss_utils import faiss


@dataclass
class MemoryChunk:
    id: str
    timestamp: float
    summary: str
    content: str
    topic: str


class MemoryStore:
    """Persist memories to a JSON file and allow vector similarity search."""

    def __init__(self, path: Path, dim: int = 128) -> None:
        self.path = path
        self.dim = dim
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            self.memories: List[MemoryChunk] = [MemoryChunk(**m) for m in data]
        else:
            self.memories = []
        self.index = faiss.IndexFlatIP(dim)
        if self.memories:
            vecs = [hash_embedding(m.summary, dim) for m in self.memories]
            self.index.add(vecs)

    # ------------------------------------------------------------------
    def add_memory(self, summary: str, content: str, topic: str) -> MemoryChunk:
        chunk_id = f"chunk-{int(time.time())}-{len(self.memories)+1:03d}"
        chunk = MemoryChunk(
            id=chunk_id,
            timestamp=time.time(),
            summary=summary,
            content=content,
            topic=topic,
        )
        vec = hash_embedding(summary, self.dim)
        self.index.add([vec])
        self.memories.append(chunk)
        self.save()
        return chunk

    # ------------------------------------------------------------------
    def retrieve(self, query: str, top_k: int = 3) -> List[MemoryChunk]:
        if not self.memories:
            return []
        qvec = hash_embedding(query, self.dim)
        scores, idx = self.index.search([qvec], top_k)
        results: List[MemoryChunk] = []
        for score, i in zip(scores[0], idx[0]):
            if i < 0 or score <= 0:
                continue
            results.append(self.memories[int(i)])
        return results

    # ------------------------------------------------------------------
    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            json.dump([m.__dict__ for m in self.memories], f, ensure_ascii=False, indent=2)
