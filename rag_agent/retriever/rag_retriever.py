"""Document retriever using a FAISS index."""

from __future__ import annotations

from pathlib import Path
from typing import List

from ..utils.text import model_embedding
from ..utils.faiss_utils import faiss


class DocumentRetriever:
    """Load plain text documents and return similar ones via vector search."""

    def __init__(self, documents_dir: Path, dim: int = 128) -> None:
        self.dim = dim
        self.docs: List[str] = []
        embeddings: List[List[float]] = []
        for path in documents_dir.glob("*.txt"):
            text = path.read_text(encoding="utf-8")
            self.docs.append(text)
            embeddings.append(model_embedding(text, dim))
        self.index = faiss.IndexFlatIP(dim)
        if embeddings:
            self.index.add(embeddings)

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        if not self.docs:
            return []
        qvec = model_embedding(query, self.dim)
        scores, idx = self.index.search([qvec], top_k)
        results: List[str] = []
        for score, i in zip(scores[0], idx[0]):
            if i < 0 or score <= 0:
                continue
            results.append(self.docs[int(i)])
        return results
