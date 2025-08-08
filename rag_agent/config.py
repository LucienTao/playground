"""Configuration for RAG agent.

The config uses simple constants to avoid external dependencies.
"""

from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    # Path to memory json file
    memory_path: Path = Path(__file__).resolve().parent / "memory" / "memory_store.json"
    # Directory containing knowledge documents
    documents_dir: Path = Path(__file__).resolve().parent / "data" / "documents"
    # Threshold for context switch based on Jaccard similarity
    context_switch_threshold: float = 0.3
