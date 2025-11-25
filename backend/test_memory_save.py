import json
import os
import sys

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from memory_service import save_memory_entry

def test_save_memory():
    query = "test_query_unique_123"
    code = "print('test_code')"
    plan = "test_plan"
    
    # Save memory
    save_memory_entry(query, code, plan)
    
    # Check file
    memory_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "code_memory.json")
    with open(memory_file, "r") as f:
        data = json.load(f)
        
    # Verify entry exists
    found = False
    for entry in data:
        if entry["query"] == query and entry["code"] == code:
            found = True
            break
            
    if found:
        print("SUCCESS: Memory saved correctly.")
    else:
        print("FAILURE: Memory not found.")
        sys.exit(1)

if __name__ == "__main__":
    test_save_memory()
