from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.engine import HybridSearchEngine

app = FastAPI(title="Mini Vector Database & Hybrid Search Engine", version="1.0")

# Initialize our vector database instance globally
db = HybridSearchEngine()

# Preload a couple of sample documents on startup so the DB isn't empty
db.add_document(1, "Introduction to vector databases and machine learning embeddings")
db.add_document(2, "Building high-performance backend systems with Python and FastAPI")
db.add_document(3, "Natural language processing and semantic similarity search engines")

# Pydantic models to validate incoming HTTP request payloads
class DocumentInput(BaseModel):
    id: int
    text: str

class SearchInput(BaseModel):
    query: str
    top_k: int = 2

@app.get("/")
def home():
    return {"message": "Welcome to your Vector Database & Search Engine API!"}

@app.post("/documents/")
def add_document(doc: DocumentInput):
    """API endpoint to add a new document into our vector store."""
    try:
        db.add_document(doc.id, doc.text)
        return {"status": "success", "message": f"Document {doc.id} added successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/")
def search_documents(payload: SearchInput):
    """API endpoint to search the database using vector similarity."""
    try:
        results = db.search(payload.query, payload.top_k)
        return {
            "query": payload.query,
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))