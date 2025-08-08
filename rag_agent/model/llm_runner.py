"""Very small LLM runner abstraction.

In a real deployment this module would interface with a vLLM server hosting
`Qwen3-8B`.  For repository purposes we provide a deterministic stub that
returns the prompt appended with a canned acknowledgement.  This keeps tests
lightweight while exercising the control flow of the agent.
"""

from __future__ import annotations


class LLMRunner:
    def generate(self, prompt: str) -> str:
        # In reality this would call into a local LLM.  Here we simply echo a
        # confirmation to keep examples deterministic.
        return f"[LLM 回复] {prompt.splitlines()[-1]}"
