# Hybrid Search Engine

A custom search engine built from scratch using Python, NumPy, and FastAPI that combines semantic vector search with exact keyword matching.

## Features
- **Semantic Vector Search:** Computes cosine similarity using vector embeddings to capture underlying meaning and context.
- **Exact Keyword Matching:** Implements traditional text-scoring relevance to catch specific terms.
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

```

## Getting Started

1. Clone the repository:
```bash
git clone [https://github.com/asthaap5347/HybridSearchEngine.git](https://github.com/asthaap5347/HybridSearchEngine.git)

```


2. Navigate into the project and install dependencies:
```bash
cd HybridSearchEngine
pip install fastapi uvicorn numpy pydantic

```


3. Run the local server:
```bash
uvicorn src.main:app --reload

```


4. Open your browser and explore the interactive API docs at `http://127.0.0.1:8000/docs`.

```

