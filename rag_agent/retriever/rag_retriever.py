"""Simple document retriever used for RAG."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from ..memory.memory_store import _tokenize, _jaccard


class DocumentRetriever:
    """Load plain text documents and return the most similar ones."""

    def __init__(self, documents_dir: Path) -> None:
        self.docs: List[Tuple[str, List[str]]] = []
        for path in documents_dir.glob("*.txt"):
            text = path.read_text(encoding="utf-8")
            tokens = _tokenize(text)
            self.docs.append((text, tokens))

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        tokens = _tokenize(query)
        scored = [(_jaccard(tokens, doc_tokens), text) for text, doc_tokens in self.docs]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [text for score, text in scored[:top_k] if score > 0]
