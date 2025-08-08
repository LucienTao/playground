# Playground

This repository contains a minimal demonstration of a Retrieval-Augmented
Generation (RAG) agent with long-term memory and context-switch detection.

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
  data/documents/       # Example knowledge documents
```

## Testing

Tests can be executed with:

```
pytest -q
```
