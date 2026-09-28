# Hybrid Search Engine

A custom search engine built from scratch using Python, NumPy, and FastAPI that combines semantic vector search with exact keyword matching.

## Features
- **Semantic Vector Search:** Computes cosine similarity using vector embeddings to capture underlying meaning and context.
- **Exact Keyword Matching:** Implements traditional text-scoring (BM25-style relevance) to catch specific terms.
- **Hybrid Fusion:** Merges both search methodologies to deliver balanced, highly relevant ranking results.
- **FastAPI Backend:** Provides clean, real-time API endpoints with automated documentation via Swagger UI.

## Tech Stack
- **Python**
- **FastAPI**
- **NumPy**
- **Pydantic**

## Project Structure
```text
HybridSearchEngine/
│
├── src/
│   ├── __init__.py
│   ├── engine.py       # Core hybrid search and scoring logic
│   └── main.py         # FastAPI application and route handlers
│
├── .gitignore
└── README.md
