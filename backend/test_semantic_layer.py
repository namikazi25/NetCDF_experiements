import os
import sys
import json
from semantic_layer import resolve_concepts_for_schema, format_semantic_context

# Add current directory to path
sys.path.append(os.getcwd())

def test_semantic_layer():
    print("Testing Semantic Layer Resolution...")
    
    # 1. Test SCHISM Case (hvel_x, hvel_y, elev)
    print("\n--- Test Case 1: SCHISM Model ---")
    schism_schema = {
        "variables": {
            "hvel_x": {}, "hvel_y": {}, "elev": {}, "temp": {}
        }
    }
    concepts = resolve_concepts_for_schema(schism_schema)
    print("Resolved Concepts:", list(concepts.keys()))
    print(format_semantic_context(concepts))
    
    # 2. Test ROMS Case (u, v, zeta)
    print("\n--- Test Case 2: ROMS Model ---")
    roms_schema = {
        "variables": {
            "u": {}, "v": {}, "zeta": {}
        }
    }
    concepts = resolve_concepts_for_schema(roms_schema)
    print("Resolved Concepts:", list(concepts.keys()))
    print(format_semantic_context(concepts))
    
    # 3. Test WRF Case (U10, V10, T2)
    print("\n--- Test Case 3: WRF Model ---")
    wrf_schema = {
        "variables": {
            "U10": {}, "V10": {}, "T2": {}
        }
    }
    concepts = resolve_concepts_for_schema(wrf_schema)
    print("Resolved Concepts:", list(concepts.keys()))
    print(format_semantic_context(concepts))

if __name__ == "__main__":
    test_semantic_layer()
