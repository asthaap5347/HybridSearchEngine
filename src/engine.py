import numpy as np
from collections import Counter

class HybridSearchEngine:
    def __init__(self):
        # Stores our documents: {"id": 1, "text": "...", "vector": [...]}
        self.documents = []

    def text_to_vector(self, text: str) -> np.ndarray:
        """Converts text into a semantic numerical vector using word hashing."""
        words = text.lower().split()
        vector = np.zeros(64)
        for word in words:
            idx = hash(word) % 64
            vector[idx] += 1.0
            
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector

    def add_document(self, doc_id: int, text: str):
        """Adds a document to our database with its vector and raw words."""
        vector = self.text_to_vector(text)
        self.documents.append({
            "id": doc_id,
            "text": text,
            "vector": vector,
            "words": text.lower().split()
        })

    def keyword_score(self, query: str, doc_words: list) -> float:
        """Calculates a simple exact-match keyword overlap score."""
        query_words = query.lower().split()
        if not query_words or not doc_words:
            return 0.0
        
        # Count how many query words appear in the document
        match_count = sum(1 for qw in query_words if qw in doc_words)
        return float(match_count / len(query_words))

    def search(self, query: str, top_k: int = 2):
        """
        Performs Hybrid Search by combining Semantic Vector Search 
        and Exact Keyword Matching.
        """
        if not self.documents:
            return []

        query_vector = self.text_to_vector(query)
        results = []

        for doc in self.documents:
            # 1. Semantic Score (Cosine Similarity)
            semantic_score = float(np.dot(doc["vector"], query_vector))
            
            # 2. Keyword Score (Exact Match)
            kw_score = self.keyword_score(query, doc["words"])
            
            # 3. Hybrid Combination (50% Meaning + 50% Exact Keywords)
            hybrid_score = (0.5 * semantic_score) + (0.5 * kw_score)

            results.append({
                "id": doc["id"],
                "text": doc["text"],
                "score": round(hybrid_score, 4),
                "semantic_score": round(semantic_score, 4),
                "keyword_score": round(kw_score, 4)
            })

        # Sort by highest hybrid score
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]