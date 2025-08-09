"""Minimal FAISS wrapper with a pure Python fallback."""

from __future__ import annotations

from typing import List, Tuple

try:  # pragma: no cover - real FAISS if available
    import faiss as _faiss  # type: ignore
    faiss = _faiss
except Exception:  # pragma: no cover - fallback
    class IndexFlatIP:
        def __init__(self, dim: int) -> None:
            self.dim = dim
            self.vectors: List[List[float]] = []

        def add(self, vecs: List[List[float]]) -> None:
            self.vectors.extend(vecs)

        def search(self, query: List[List[float]], k: int) -> Tuple[List[List[float]], List[List[int]]]:
            scores_list: List[List[float]] = []
            idx_list: List[List[int]] = []
            for q in query:
                sims = [sum(qi * vi for qi, vi in zip(q, v)) for v in self.vectors]
                order = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:k]
                scores = [sims[i] for i in order]
                indices = order
                # pad results
                while len(scores) < k:
                    scores.append(0.0)
                    indices.append(-1)
                scores_list.append(scores)
                idx_list.append(indices)
            return scores_list, idx_list

    class _FaissModule:
        IndexFlatIP = IndexFlatIP

    faiss = _FaissModule()
