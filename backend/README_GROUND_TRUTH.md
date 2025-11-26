# Ground Truth Tool System for NetCDF Query

## Overview

This system converts proven query patterns from the reference notebook into callable function tools. When a user asks a query that matches a known pattern, the system executes the ground truth code directly instead of generating new code via LLM.

## Benefits

1. **Consistent Results** - Same query always produces same output
2. **Faster Execution** - No LLM code generation step
3. **Reliability** - Proven code, no hallucination risk
4. **Fallback Safety** - Unknown queries still use LLM agent workflow

## Files

```
├── ground_truth_tools.py      # Tool registry + implementations
├── tool_caller.py             # Query matching + parameter extraction  
├── orchestrator_enhanced.py   # Drop-in replacement for orchestrator.py
├── test_ground_truth.py       # Test suite
└── README_GROUND_TRUTH.md     # This file
```

## Registered Tools

| Tool Name | Query Pattern Examples |
|-----------|----------------------|
| `get_parameter_shapes` | "Show shape and size of all parameters" |
| `calculate_average_depth` | "Calculate average water depth" |
| `find_min_max_depth` | "Find min and max depth locations" |
| `calculate_max_wave_velocity` | "Maximum wave velocity map" |
| `calculate_elevation_difference` | "Difference in elevation between files" |
| `get_point_time_series` | "Show elevation over time for point (40.7, -8.7)" |
| `find_max_variable_location` | "Find maximum tp value" |
| `find_velocity_above_average` | "Locations where velocity exceeds average" |
| `calculate_time_filtered_elevation` | "Elevation between 10 AM and 3 PM" |
| `analyze_bounding_box` | "Data within lat 40.7-40.8, lon -9.0 to -8.8" |
| `plot_average_elevation` | "Plot average elevation map" |

## Integration

### Option 1: Replace orchestrator.py (Recommended)

Copy `orchestrator_enhanced.py` content into `backend/orchestrator.py`:

```python
# In backend/orchestrator.py
from orchestrator_enhanced import run_orchestrator
```

Or simply copy the file:

```bash
cp orchestrator_enhanced.py backend/orchestrator.py
```

### Option 2: Import in app.py

```python
# In app.py, change:
from orchestrator import run_orchestrator

# To:
from orchestrator_enhanced import run_orchestrator
```

### Required File Locations

Place these files in `backend/`:

```bash
cp ground_truth_tools.py backend/
cp tool_caller.py backend/
cp orchestrator_enhanced.py backend/
```

## Usage Flow

```
User Query
    │
    ▼
┌─────────────────────┐
│ match_query_to_tool │  ──── Match keywords to tool
└─────────────────────┘
    │
    ├── Match Found ──► resolve_parameters() ──► execute_tool() ──► Response
    │
    └── No Match ──► Original Agent Workflow (Plan → Evaluate → Execute → Synthesize)
```

## Adding New Tools

1. **Add implementation to `ground_truth_tools.py`**:

```python
@register_tool(
    name="my_new_tool",
    description="Description of what this tool does",
    parameters={
        "netcdf_path": {"type": "string", "description": "Path to file"},
        "my_param": {"type": "number", "description": "Some parameter", "default": 10}
    }
)
def my_new_tool(netcdf_path: str, my_param: int = 10) -> Dict[str, Any]:
    # Your ground truth implementation
    table = generate_full_node_table(netcdf_path)
    # ... analysis code ...
    return {"table": results, "stats": {...}}
```

2. **Add query patterns to `QUERY_PATTERNS`**:

```python
QUERY_PATTERNS = {
    # ... existing tools ...
    "my_new_tool": [
        "keyword1", "keyword2", "phrase to match"
    ]
}
```

3. **Add parameter extraction logic in `tool_caller.py`** (if needed):

```python
def resolve_parameters(tool_name, query, base_path, scenario_path):
    # ... existing code ...
    
    # Add custom extraction for your tool
    if "my_param" in tool_params:
        match = re.search(r'my_param\s*[:=]?\s*(\d+)', query)
        if match:
            params["my_param"] = int(match.group(1))
```

4. **Add summary formatter in `orchestrator_enhanced.py`**:

```python
summaries = {
    # ... existing summaries ...
    "my_new_tool": lambda r: (
        f"**My Analysis**\n\n"
        f"Result: {r['stats']['value']}"
    ),
}
```

## Testing

Run the test suite:

```bash
cd backend
python test_ground_truth.py
```

Test individual queries:

```python
from ground_truth_tools import match_query_to_tool, execute_tool
from tool_caller import resolve_parameters

query = "Calculate average water depth"
tool = match_query_to_tool(query)
print(f"Matched: {tool}")

params = resolve_parameters(tool, query, "/path/to/file.nc")
print(f"Parameters: {params}")

result = execute_tool(tool, **params)
print(f"Result keys: {result.keys()}")
```

## Query Matching Strategy

The current system uses keyword matching. For production, consider:

1. **Embedding-based matching** - Use sentence transformers for semantic similarity
2. **LLM classification** - Let the LLM decide which tool to use
3. **Hybrid approach** - Keywords for obvious matches, LLM for ambiguous cases

Example embedding-based enhancement:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

# Pre-compute tool description embeddings
tool_embeddings = {
    name: model.encode(tool["description"])
    for name, tool in TOOL_REGISTRY.items()
}

def match_query_semantic(query: str, threshold: float = 0.7) -> Optional[str]:
    query_vec = model.encode(query)
    
    best_match = None
    best_score = 0
    
    for name, vec in tool_embeddings.items():
        score = cosine_similarity(query_vec, vec)
        if score > best_score and score > threshold:
            best_score = score
            best_match = name
    
    return best_match
```

## Performance Notes

- `generate_full_node_table()` loads entire dataset into memory
- For large files, consider lazy loading or chunking
- The table generation is cached per query but could be cached per file

## Troubleshooting

**Q: Tool matched but execution failed**
- Check if all required parameters are resolved
- Verify the NetCDF file has expected variables
- Check the steps_log for detailed error messages

**Q: Query not matching expected tool**
- Add more keyword patterns to `QUERY_PATTERNS`
- Check for typos in query keywords
- Consider using semantic matching for better coverage

**Q: Different results than notebook**
- Verify the same aggregation method (mean vs max)
- Check groupby columns match
- Ensure time/layer handling is consistent
