"""Prompt construction utilities."""

from __future__ import annotations

from typing import List


SYSTEM_PROMPT = "你是一个有长期记忆的智能助手。"


def build_prompt(user_input: str, memories: List[str], documents: List[str]) -> str:
    """Combine system prompt, memories and retrieved documents."""
    lines = [SYSTEM_PROMPT, "", "以下是你得到的记忆:"]
    if memories:
        for m in memories:
            lines.append(f"- {m}")
    else:
        lines.append("- (无)")
    lines.append("")
    if documents:
        lines.append("知识:")
        for i, doc in enumerate(documents, 1):
            snippet = doc.strip().replace("\n", " ")[:200]
            lines.append(f"[文档{i}] {snippet}")
        lines.append("")
    lines.append(f"当前用户输入:\n{user_input}\n")
    lines.append("请合理结合记忆和当前问题给出回复")
    return "\n".join(lines)
