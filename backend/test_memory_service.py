import os
import sys
import json
from memory_service import save_memory_entry, find_similar_code, load_memory

# Add current directory to path
sys.path.append(os.getcwd())

def test_memory_service():
    print("Testing Memory Service...")
    
    # Clear memory file for testing
    if os.path.exists("backend/code_memory.json"):
        os.remove("backend/code_memory.json")
    
    # 1. Test Saving
    print("\n--- Test Case 1: Saving Memory ---")
    query = "Calculate max wave height"
    code = "print(ds['wsh'].max())"
    plan = "Load file, calculate max."
    
    save_memory_entry(query, code, plan)
    
    memories = load_memory()
    print(f"Memories count: {len(memories)}")
    assert len(memories) == 1
    assert memories[0]["query"] == query
    print("Save successful.")
    
    # 2. Test Retrieval (Exact Match)
    print("\n--- Test Case 2: Retrieval (Exact) ---")
    match = find_similar_code("Calculate max wave height")
    if match:
        print(f"Found match: {match['query']}")
        assert match['code'] == code
    else:
        print("No match found (Unexpected).")
        
    # 3. Test Retrieval (Semantic Match)
    print("\n--- Test Case 3: Retrieval (Semantic) ---")
    # Note: This depends on the embedding model. 
    # "Find maximum wave height" should be close to "Calculate max wave height"
    match = find_similar_code("Find maximum wave height")
    if match:
        print(f"Found match: {match['query']}")
        print(f"Code: {match['code']}")
    else:
        print("No match found (Might be due to mock embeddings or low threshold).")

if __name__ == "__main__":
    test_memory_service()
