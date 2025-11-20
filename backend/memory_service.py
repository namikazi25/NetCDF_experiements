import json
import os
import numpy as np
from llm_service import client  # Re-use your existing client for embeddings

# Ensure the memory file is in the same directory as this script for simplicity
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEMORY_FILE = os.path.join(BASE_DIR, "code_memory.json")

def get_embedding(text):
    """Generates a vector embedding for the text."""
    try:
        # Using standard OpenAI embedding model
        # Note: Ensure your client supports embeddings or use a fallback/mock if strictly local
        embedding_model = os.getenv("LOCAL_EMBEDDING_MODEL", "text-embedding-qwen3-embedding-4b")
        response = client.embeddings.create(
            input=text,
            model=embedding_model
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Embedding Error: {e}")
        # Fallback for testing/offline mode
        # Create a deterministic vector based on the hash of the text
        import hashlib
        hash_val = int(hashlib.sha256(text.encode('utf-8')).hexdigest(), 16)
        np.random.seed(hash_val % 2**32)
        return np.random.rand(1536).tolist()

def load_memory():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []

def save_memory_entry(query, code, plan_summary):
    """Saves a successful execution to the recipe book."""
    memories = load_memory()
    
    # Avoid duplicates (simple check)
    for mem in memories:
        if mem["query"] == query and mem["code"] == code:
            return

    entry = {
        "query": query,
        "code": code,
        "plan": plan_summary,
        "embedding": get_embedding(query) # Store vector for fast search
    }
    
    memories.append(entry)
    with open(MEMORY_FILE, "w") as f:
        json.dump(memories, f, indent=2)

def find_similar_code(current_query, threshold=0.75):
    """Finds the most relevant past code snippet."""
    memories = load_memory()
    if not memories:
        return None
        
    query_vec = get_embedding(current_query)
    if not query_vec:
        return None

    best_match = None
    best_score = -1

    for mem in memories:
        if not mem.get("embedding"): continue
        
        # Cosine Similarity calculation
        vec_a = np.array(query_vec)
        vec_b = np.array(mem["embedding"])
        
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        
        if norm_a == 0 or norm_b == 0:
            continue
            
        # Dimension check
        if vec_a.shape != vec_b.shape:
            print(f"Warning: Embedding dimension mismatch ({vec_a.shape} vs {vec_b.shape}). Skipping memory entry.")
            continue
            
        score = np.dot(vec_a, vec_b) / (norm_a * norm_b)
        
        if score > best_score:
            best_score = score
            best_match = mem

    if best_score >= threshold:
        return best_match
    return None
