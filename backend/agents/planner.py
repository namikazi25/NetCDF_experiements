import json
from llm_service import client, MODEL
# Import the formatter we just made
from schema_registry import format_context_for_planner
from semantic_layer import format_semantic_context
from memory_service import find_similar_code

def plan_task(query: str, metadata_bundle: dict) -> dict:
    
    # 1. Check Memory First
    similar_task = find_similar_code(query)
    
    memory_context = ""
    if similar_task:
        memory_context = f"""
        ### 🧠 RELEVANT MEMORY (PREVIOUS SOLUTION):
        The user asked a similar question before: "{similar_task['query']}"
        Here is the code that worked efficiently:
        
        ```python
        {similar_task['code']}
        ```
        
        **INSTRUCTION:** 1. Use the code above as a template.
        2. Adapt variable names (e.g. 'elev' vs 'zeta') to match the current file schema.
        3. Do not reinvent the logic; reuse the Pandas/Xarray pattern shown above.
        """

    # Generate the text using the new smart formatter
    context_str = format_context_for_planner(metadata_bundle)
    
    # Add Level 2 Context
    # Assume we pull 'concepts' from the metadata_bundle
    # The app puts 'concepts' inside the 'baseline' dict
    concepts = metadata_bundle.get('baseline', {}).get('concepts', {})
    semantic_context = format_semantic_context(concepts)
    
    # Check if we are in comparison mode to inject specific rules
    is_comparison = 'scenario' in metadata_bundle
    
    comparison_rules = ""
    if is_comparison:
        comparison_rules = """
        5. **Comparison Logic:**
           - You have TWO datasets: `ds_base` and `ds_scen`.
           - If the user asks for "difference", "change", or "impact":
             1. Align: `ds_base, ds_scen = xr.align(ds_base, ds_scen, join='override')`
             2. Subtract: `diff = ds_scen['var'] - ds_base['var']`
             3. Plot the `diff`.
           - If the user asks for a "Map", generate THREE plots: Baseline, Scenario, and Difference.
        """

    system_prompt = f"""You are a Senior Data Scientist acting as a Planner.
    
    {context_str}
    
    {semantic_context}
    
    {memory_context}

    ### PLANNING RULES:
    1. **Terminology:** If the user asks for a Concept (e.g. "Plot Velocity"), CHECK the "SEMANTIC CONCEPTS" list.
       - If it says "Calculate using...", write that code.
       - If it says "Use variable...", use that variable name.
    
    2. **Variable Selection:** Use "CALCULABLE CONCEPTS" if available.
    3. **Spatial Filtering:** Use `ds.sel(..., method='nearest')` for specific points.
    4. **Data Structures:** For sorting or tables, use `.to_dataframe()`.
    {comparison_rules}
    
    OUTPUT FORMAT (JSON):
    {{ "thought": "...", "steps": [...] }}
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"User Query: {query}"}
    ]

    # ... (Rest of the standard API call logic remains the same)
    try:
        response = client.chat.completions.create(model=MODEL, messages=messages)
        content = response.choices[0].message.content
        
        # JSON Cleanup
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
            
        return json.loads(content)
    except Exception as e:
        return {"thought": f"Error: {e}", "steps": []}
