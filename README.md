# Playground

This repository contains a minimal demonstration of a Retrieval-Augmented
Generation (RAG) agent with long-term memory and context-switch detection.

The demo now persists memories and documents in a FAISS-like vector store. Each
text is hashed into a dense vector for efficient similarity search without
external services. A lightweight LLM runner stub is provided to simulate
responses; in a real deployment this would call a model such as Qwen3-8B.

## Structure

```
rag_agent/
  app.py                # Agent entry point
  config.py             # Basic configuration values
  memory/               # Memory store and summarizer
  retriever/            # Document retriever for RAG
  prompt/               # Prompt construction
  detector/             # Context switch detection
  model/                # Lightweight LLM runner stub
  utils/                # Tokenization, embeddings and FAISS helpers
  data/documents/       # Example knowledge documents
```

## Testing

Tests can be executed with:

```
pytest -q
```
