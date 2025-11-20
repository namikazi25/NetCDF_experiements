import json
import numpy as np
import os

# Load the knowledge base once
# Ensure we look in the same directory as this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KB_PATH = os.path.join(BASE_DIR, "knowledge_base.json")

try:
    with open(KB_PATH, "r") as f:
        KNOWLEDGE_BASE = json.load(f)
except FileNotFoundError:
    print(f"Warning: Knowledge base not found at {KB_PATH}")
    KNOWLEDGE_BASE = {"concepts": {}}

def resolve_concepts_for_schema(schema: dict) -> dict:
    """
    Matches the 'Universal Concepts' to the specific variables 
    available in the provided file schema.
    """
    file_vars = set(schema["variables"].keys())
    available_concepts = {}

    for concept_name, details in KNOWLEDGE_BASE["concepts"].items():
        
        # Check all known definitions for this concept
        for definition in details["definitions"]:
            
            # STRATEGY 1: Vector Magnitude (Calculated)
            if definition["type"] == "vector_magnitude":
                comp_x, comp_y = definition["components"]
                if comp_x in file_vars and comp_y in file_vars:
                    available_concepts[concept_name] = {
                        "type": "derived",
                        "formula": f"np.sqrt(ds['{comp_x}']**2 + ds['{comp_y}']**2)",
                        "source_vars": [comp_x, comp_y]
                    }
                    break # Found a match, stop looking

            # STRATEGY 2: Direct Variable Mapping
            elif definition["type"] == "direct":
                target_var = definition["variable"]
                if target_var in file_vars:
                    available_concepts[concept_name] = {
                        "type": "direct",
                        "variable": target_var,
                        "desc": details["description"]
                    }
                    break

    return available_concepts

def format_semantic_context(available_concepts: dict) -> str:
    """
    Formats the resolved concepts for the Planner Prompt.
    """
    if not available_concepts:
        return ""

    out = "### SEMANTIC CONCEPTS (USER TERMINOLOGY):\n"
    out += "Use these mappings to understand user terms:\n"
    
    for name, info in available_concepts.items():
        if info["type"] == "derived":
            out += f"- **{name.title()}**: Calculate using `{info['formula']}`\n"
        elif info["type"] == "direct":
            out += f"- **{name.title()}**: Use variable `{info['variable']}`\n"
            
    return out
