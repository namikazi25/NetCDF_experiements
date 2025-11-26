"""
Tool Caller Integration for NetCDF Query System

This module integrates the ground truth tools with the LLM agent system.
It provides:
1. Query classification to detect if a ground truth tool matches
2. Tool execution with proper parameter extraction
3. Fallback to the original agent workflow if no match
"""

import json
import re
from typing import Dict, Any, Optional, Tuple
from ground_truth_tools import (
    TOOL_REGISTRY,
    execute_tool,
    get_tool_definitions,
    match_query_to_tool,
    QUERY_PATTERNS
)
from llm_service import client, MODEL


# ============================================================================
# ENHANCED QUERY CLASSIFIER (Using LLM for parameter extraction)
# ============================================================================

def classify_and_extract(query: str, metadata: dict) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Use LLM to classify query and extract parameters for tool calling.
    Returns (tool_name, parameters) or (None, {}) if no match.
    """
    
    # First, try simple keyword matching
    simple_match = match_query_to_tool(query)
    
    if not simple_match:
        return None, {}
    
    # If we have a match, use LLM to extract parameters
    tool_info = TOOL_REGISTRY[simple_match]
    
    system_prompt = f"""You are a parameter extraction assistant.
    
Given a user query about NetCDF data analysis, extract the parameters needed for this tool:

TOOL: {simple_match}
DESCRIPTION: {tool_info['description']}
PARAMETERS: {json.dumps(tool_info['parameters'], indent=2)}

AVAILABLE FILE PATHS FROM CONTEXT:
{json.dumps(metadata, indent=2)}

RULES:
1. Extract ONLY the parameters defined for this tool
2. Use exact parameter names from the schema
3. For file paths, use the paths from the metadata context
4. For coordinates, extract numbers from the query
5. Return valid JSON only

OUTPUT FORMAT:
{{
    "tool_name": "{simple_match}",
    "parameters": {{...extracted parameters...}},
    "confidence": 0.0-1.0
}}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Query: {query}"}
    ]
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages
        )
        content = response.choices[0].message.content
        
        # Clean JSON
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        result = json.loads(content)
        
        if result.get("confidence", 0) >= 0.7:
            return result["tool_name"], result["parameters"]
        
    except Exception as e:
        print(f"Parameter extraction failed: {e}")
    
    # Fallback: return tool with minimal parameters
    return simple_match, {}


def extract_coordinates_from_query(query: str) -> Dict[str, float]:
    """Extract lat/lon coordinates from natural language query."""
    coords = {}
    
    # Pattern: (40.699429, -8.756169) or 40.699429 lat, -8.756169 lon
    lat_patterns = [
        r'(\d+\.\d+)\s*(?:lat|latitude)',
        r'lat(?:itude)?\s*[:=]?\s*(\d+\.\d+)',
        r'\((\d+\.\d+)\s*,',
    ]
    
    lon_patterns = [
        r'(-?\d+\.\d+)\s*(?:lon|longitude)',
        r'lon(?:gitude)?\s*[:=]?\s*(-?\d+\.\d+)',
        r',\s*(-?\d+\.\d+)\)',
    ]
    
    for pattern in lat_patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            coords['lat'] = float(match.group(1))
            break
    
    for pattern in lon_patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            coords['lon'] = float(match.group(1))
            break
    
    return coords


def extract_bounding_box_from_query(query: str) -> Dict[str, float]:
    """Extract bounding box coordinates from query."""
    bbox = {}
    
    # Pattern: Latitude: 40.7 to 40.8
    lat_range = re.search(r'lat(?:itude)?[:\s]*(\d+\.?\d*)\s*to\s*(\d+\.?\d*)', query, re.IGNORECASE)
    lon_range = re.search(r'lon(?:gitude)?[:\s]*(-?\d+\.?\d*)\s*to\s*(-?\d+\.?\d*)', query, re.IGNORECASE)
    
    if lat_range:
        bbox['lat_min'] = float(lat_range.group(1))
        bbox['lat_max'] = float(lat_range.group(2))
    
    if lon_range:
        bbox['lon_min'] = float(lon_range.group(1))
        bbox['lon_max'] = float(lon_range.group(2))
    
    return bbox


def extract_time_range_from_query(query: str) -> Dict[str, int]:
    """Extract hour range from query."""
    time_range = {}
    
    # Pattern: between 10:00 AM and 3:00 PM
    pattern = r'(\d{1,2})(?::00)?\s*(?:AM|am)?\s*(?:and|to)\s*(\d{1,2})(?::00)?\s*(?:PM|pm)?'
    match = re.search(pattern, query)
    
    if match:
        start = int(match.group(1))
        end = int(match.group(2))
        # Convert PM to 24h
        if 'pm' in query.lower() and end < 12:
            end += 12
        time_range['start_hour'] = start
        time_range['end_hour'] = end
    
    return time_range


def extract_top_n_from_query(query: str) -> Optional[int]:
    """Extract top N value from query."""
    match = re.search(r'top\s*(\d+)', query, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


# ============================================================================
# SMART PARAMETER RESOLVER
# ============================================================================

def resolve_parameters(
    tool_name: str,
    query: str,
    base_path: Optional[str] = None,
    scenario_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Intelligently resolve parameters for a tool based on query and context.
    """
    params = {}
    tool_info = TOOL_REGISTRY.get(tool_name, {})
    tool_params = tool_info.get("parameters", {})
    
    # File paths
    if "netcdf_path" in tool_params:
        # Determine which file to use based on query
        if "scenario" in query.lower() or "sch1" in query.lower():
            params["netcdf_path"] = scenario_path or base_path
        else:
            params["netcdf_path"] = base_path
    
    if "base_path" in tool_params and base_path:
        params["base_path"] = base_path
    
    if "scenario_path" in tool_params and scenario_path:
        params["scenario_path"] = scenario_path
    
    # Coordinates
    coords = extract_coordinates_from_query(query)
    if coords:
        params.update(coords)
    
    # Bounding box
    bbox = extract_bounding_box_from_query(query)
    if bbox:
        params.update(bbox)
    
    # Time range
    time_range = extract_time_range_from_query(query)
    if time_range:
        params.update(time_range)
    
    # Top N
    top_n = extract_top_n_from_query(query)
    if top_n and "top_n" in tool_params:
        params["top_n"] = top_n
    
    # Variable extraction
    variable_keywords = {
        "tp": ["tp", "wave period"],
        "elev": ["elevation", "elev", "surface elevation"],
        "depth": ["depth", "water depth"],
        "wave_velocity": ["wave velocity", "wsh"]
    }
    
    if "variable" in tool_params:
        query_lower = query.lower()
        for var, keywords in variable_keywords.items():
            if any(kw in query_lower for kw in keywords):
                params["variable"] = var
                break
    
    # Default plot generation to True
    if "generate_plot" in tool_params:
        params["generate_plot"] = True
    
    return params


# ============================================================================
# MAIN TOOL CALLER
# ============================================================================

class GroundTruthToolCaller:
    """
    Routes queries to ground truth tools when applicable.
    Falls back to agent workflow otherwise.
    """
    
    def __init__(self, base_path: str = None, scenario_path: str = None):
        self.base_path = base_path
        self.scenario_path = scenario_path
    
    def set_paths(self, base_path: str, scenario_path: str = None):
        """Update file paths."""
        self.base_path = base_path
        self.scenario_path = scenario_path
    
    def try_ground_truth(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Attempt to answer query using ground truth tools.
        Returns result dict if successful, None if no matching tool.
        """
        # Match query to tool
        tool_name = match_query_to_tool(query)
        
        if not tool_name:
            return None
        
        print(f"[GroundTruth] Matched tool: {tool_name}")
        
        # Resolve parameters
        params = resolve_parameters(
            tool_name,
            query,
            self.base_path,
            self.scenario_path
        )
        
        print(f"[GroundTruth] Resolved parameters: {params}")
        
        # Validate required parameters
        tool_info = TOOL_REGISTRY[tool_name]
        required = [k for k, v in tool_info["parameters"].items() if "default" not in v]
        
        missing = [r for r in required if r not in params]
        if missing:
            print(f"[GroundTruth] Missing required parameters: {missing}")
            return None
        
        # Execute tool
        result = execute_tool(tool_name, **params)
        
        if "error" in result:
            print(f"[GroundTruth] Execution error: {result['error']}")
            return None
        
        # Format response
        return self._format_response(tool_name, result, query)
    
    def _format_response(self, tool_name: str, result: Dict, query: str) -> Dict[str, Any]:
        """Format tool result into standardized response."""
        images = []
        if "image" in result:
            images.append(result["image"])
        
        # Generate natural language summary
        summary = self._generate_summary(tool_name, result, query)
        
        return {
            "response": summary,
            "images": images,
            "tool_used": tool_name,
            "raw_result": result,
            "steps_log": [
                {"stage": "Tool Matching", "status": "complete", "output": f"Matched: {tool_name}"},
                {"stage": "Execution", "status": "complete", "output": "Ground truth tool executed"},
                {"stage": "Synthesis", "status": "complete", "output": "Response generated"}
            ]
        }
    
    def _generate_summary(self, tool_name: str, result: Dict, query: str) -> str:
        """Generate natural language summary of results."""
        
        if tool_name == "get_parameter_shapes":
            return f"Found {len(result.get('table', []))} parameters in the file. See the table for details."
        
        elif tool_name == "calculate_average_depth":
            stats = result.get("stats", {})
            return (f"Average depth analysis complete.\n"
                   f"- Mean depth: {stats.get('mean_depth', 'N/A'):.2f}m\n"
                   f"- Max depth: {stats.get('max_depth', 'N/A'):.2f}m\n"
                   f"- Min depth: {stats.get('min_depth', 'N/A'):.2f}m\n"
                   f"- Total points: {stats.get('num_points', 0)}")
        
        elif tool_name == "find_min_max_depth":
            max_rows = result.get("max_depth", [])
            min_rows = result.get("min_depth", [])
            return (f"Depth extremes found:\n"
                   f"- Maximum: {max_rows[0]['depth']:.2f}m at ({max_rows[0]['lat']:.4f}, {max_rows[0]['lon']:.4f})\n"
                   f"- Minimum: {min_rows[0]['depth']:.2f}m at ({min_rows[0]['lat']:.4f}, {min_rows[0]['lon']:.4f})")
        
        elif tool_name == "calculate_max_wave_velocity":
            stats = result.get("stats", {})
            return (f"Maximum wave velocity analysis complete.\n"
                   f"- Peak velocity: {stats.get('max_velocity', 'N/A'):.4f}\n"
                   f"- Mean velocity: {stats.get('mean_velocity', 'N/A'):.4f}")
        
        elif tool_name == "calculate_elevation_difference":
            stats = result.get("stats", {})
            return (f"Elevation difference (Base - Scenario):\n"
                   f"- Mean difference: {stats.get('mean_diff', 'N/A'):.4f}m\n"
                   f"- Max difference: {stats.get('max_diff', 'N/A'):.4f}m\n"
                   f"- Min difference: {stats.get('min_diff', 'N/A'):.4f}m")
        
        elif tool_name == "find_velocity_above_average":
            avg = result.get("average_velocity", 0)
            count = result.get("num_points_above_avg", 0)
            return (f"Found {count} locations with wave velocity above average ({avg:.4f}).\n"
                   f"See the plot and table for details.")
        
        else:
            # Generic response
            return f"Analysis complete using {tool_name}. See the attached results."


# ============================================================================
# INTEGRATION WITH ORCHESTRATOR
# ============================================================================

def run_with_ground_truth(
    query: str,
    metadata_bundle: dict,
    base_path: str,
    scenario_path: str = None
) -> Dict[str, Any]:
    """
    Enhanced orchestrator that tries ground truth tools first.
    Falls back to original agent workflow if no match.
    """
    from orchestrator import run_orchestrator
    
    # Initialize tool caller
    caller = GroundTruthToolCaller(base_path, scenario_path)
    
    # Try ground truth first
    result = caller.try_ground_truth(query)
    
    if result:
        print("[Router] Using ground truth tool")
        return result
    
    # Fallback to original agent workflow
    print("[Router] No ground truth match, using agent workflow")
    return run_orchestrator(query, metadata_bundle, base_path, scenario_path)


# ============================================================================
# OPENAI FUNCTION CALLING FORMAT
# ============================================================================

def get_openai_tools_schema() -> list:
    """Return tools in OpenAI function calling format."""
    return get_tool_definitions()


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    # Test parameter extraction
    test_cases = [
        "Show the shape and size of all parameters",
        "Calculate average water depth",
        "Find the maximum wave velocity for scenario file",
        "For this point (40.699429 lat, -8.756169 lon) show elevation over time",
        "Calculate elevation between 10:00 AM and 3:00 PM",
        "Find top 100 locations where velocity exceeds average",
        "Analyze data within Latitude: 40.7 to 40.8, Longitude: -9.0 to -8.8"
    ]
    
    print("=" * 60)
    print("PARAMETER EXTRACTION TEST")
    print("=" * 60)
    
    for query in test_cases:
        print(f"\nQuery: {query}")
        tool = match_query_to_tool(query)
        if tool:
            params = resolve_parameters(tool, query, "/path/to/base.nc", "/path/to/scenario.nc")
            print(f"Tool: {tool}")
            print(f"Parameters: {json.dumps(params, indent=2)}")
        else:
            print("No tool match")
