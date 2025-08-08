from pathlib import Path
from rag_agent.memory.memory_store import MemoryStore


def test_add_and_retrieve(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "mem.json")
    store.add_memory("关于RAG的讨论", "用户和助手讨论RAG", topic="RAG")
    results = store.retrieve("请问RAG是什么?")
    assert results, "should retrieve at least one memory"
    assert results[0].summary.startswith("关于RAG")
