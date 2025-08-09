"""Entry point providing a simple `Agent` class for chatting."""

from __future__ import annotations

from .config import Config
from .memory.memory_store import MemoryStore
from .memory.summarizer import summarize
from .retriever.rag_retriever import DocumentRetriever
from .prompt.rag_prompt import build_prompt
from .detector.context_switch import is_context_switch
from .model.llm_runner import LLMRunner


class Agent:
    """Stateful assistant with naive memory and RAG capabilities."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self.memory_store = MemoryStore(self.config.memory_path)
        self.retriever = DocumentRetriever(self.config.documents_dir)
        self.llm = LLMRunner()
        self.topic_buffer: str = ""
        self.last_user_input: str | None = None

    def chat(self, user_input: str) -> str:
        memories = [m.summary for m in self.memory_store.retrieve(user_input)]
        documents = self.retriever.retrieve(user_input)
        prompt = build_prompt(user_input, memories, documents)
        response = self.llm.generate(prompt)

        # context switch detection
        if self.last_user_input and is_context_switch(
            user_input, self.last_user_input, self.config.context_switch_threshold
        ):
            if self.topic_buffer.strip():
                summary = summarize(self.topic_buffer)
                self.memory_store.add_memory(summary, self.topic_buffer, topic=summary[:20])
            self.topic_buffer = ""

        self.topic_buffer += f"用户: {user_input}\n助手: {response}\n"
        self.last_user_input = user_input
        return response


__all__ = ["Agent"]
