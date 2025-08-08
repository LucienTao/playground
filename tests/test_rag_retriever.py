from pathlib import Path
from rag_agent.retriever.rag_retriever import DocumentRetriever


def test_document_retrieval() -> None:
    retriever = DocumentRetriever(Path("rag_agent/data/documents"))
    results = retriever.retrieve("RAG 是什么?", top_k=1)
    assert results and "RAG" in results[0]
