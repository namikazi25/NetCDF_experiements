"""
Test Suite for Ground Truth Tools

Run this to verify the tool registry and query matching work correctly.
"""

import os
import sys

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ground_truth_tools import (
    TOOL_REGISTRY,
    list_tools,
    match_query_to_tool,
    get_tool_definitions
)
from tool_caller import (
    resolve_parameters,
    extract_coordinates_from_query,
    extract_bounding_box_from_query,
    extract_time_range_from_query,
    extract_top_n_from_query,
    GroundTruthToolCaller
)


def test_tool_registry():
    """Verify all tools are registered correctly."""
    print("\n" + "=" * 60)
    print("TEST: Tool Registry")
    print("=" * 60)
    
    expected_tools = [
        "get_parameter_shapes",
        "calculate_average_depth", 
        "find_min_max_depth",
        "calculate_max_wave_velocity",
        "calculate_elevation_difference",
        "get_point_time_series",
        "find_max_variable_location",
        "find_velocity_above_average",
        "calculate_time_filtered_elevation",
        "analyze_bounding_box",
        "plot_average_elevation"
    ]
    
    for tool in expected_tools:
        if tool in TOOL_REGISTRY:
            print(f"✅ {tool}")
        else:
            print(f"❌ {tool} - MISSING")
    
    print(f"\nTotal registered: {len(TOOL_REGISTRY)}")


def test_query_matching():
    """Test query to tool matching."""
    print("\n" + "=" * 60)
    print("TEST: Query Matching")
    print("=" * 60)
    
    test_cases = [
        # (query, expected_tool)
        ("Show the shape and size of all parameters", "get_parameter_shapes"),
        ("What are the dimensions of this file?", "get_parameter_shapes"),
        ("Calculate average water depth", "calculate_average_depth"),
        ("Make a plot for water depth", "calculate_average_depth"),
        ("Find the min and max depth", "find_min_max_depth"),
        ("What is the maximum depth?", "find_min_max_depth"),
        ("Calculate maximum wave velocity", "calculate_max_wave_velocity"),
        ("Plot wave orbital velocity from wsh_x and wsh_y", "calculate_max_wave_velocity"),
        ("Plot the difference in surface elevation between two files", "calculate_elevation_difference"),
        ("Compare elevation between base and scenario", "calculate_elevation_difference"),
        ("Show elevation over time for this point", "get_point_time_series"),
        ("For point (40.7, -8.7) show change over time", "get_point_time_series"),
        ("Find maximum tp value", "find_max_variable_location"),
        ("Where is the max tp?", "find_max_variable_location"),
        ("Find locations where velocity exceeds average", "find_velocity_above_average"),
        ("Top 100 locations with highest velocity", "find_velocity_above_average"),
        ("Calculate elevation between 10:00 AM and 3:00 PM", "calculate_time_filtered_elevation"),
        ("Average elevation from 10 to 15 hours", "calculate_time_filtered_elevation"),
        ("Analyze within bounding box lat 40.7 to 40.8", "analyze_bounding_box"),
        ("Plot average elevation map", "plot_average_elevation"),
    ]
    
    passed = 0
    failed = 0
    
    for query, expected in test_cases:
        result = match_query_to_tool(query)
        if result == expected:
            print(f"✅ '{query[:40]}...' -> {result}")
            passed += 1
        else:
            print(f"❌ '{query[:40]}...' -> {result} (expected: {expected})")
            failed += 1
    
    print(f"\nPassed: {passed}/{len(test_cases)}")


def test_parameter_extraction():
    """Test parameter extraction from queries."""
    print("\n" + "=" * 60)
    print("TEST: Parameter Extraction")
    print("=" * 60)
    
    # Coordinates
    print("\n--- Coordinate Extraction ---")
    coord_tests = [
        ("point (40.699429 lat, -8.756169 lon)", {"lat": 40.699429, "lon": -8.756169}),
        ("latitude: 40.7, longitude: -8.8", {"lat": 40.7, "lon": -8.8}),
        ("at (40.5, -9.0)", {"lat": 40.5, "lon": -9.0}),
    ]
    
    for query, expected in coord_tests:
        result = extract_coordinates_from_query(query)
        match = all(abs(result.get(k, 0) - v) < 0.001 for k, v in expected.items())
        status = "✅" if match else "❌"
        print(f"{status} '{query}' -> {result}")
    
    # Bounding box
    print("\n--- Bounding Box Extraction ---")
    bbox_tests = [
        ("Latitude: 40.7 to 40.8, Longitude: -9.0 to -8.8", 
         {"lat_min": 40.7, "lat_max": 40.8, "lon_min": -9.0, "lon_max": -8.8}),
    ]
    
    for query, expected in bbox_tests:
        result = extract_bounding_box_from_query(query)
        print(f"'{query}' -> {result}")
    
    # Time range
    print("\n--- Time Range Extraction ---")
    time_tests = [
        ("between 10:00 AM and 3:00 PM", {"start_hour": 10, "end_hour": 15}),
        ("from 10 to 15", {"start_hour": 10, "end_hour": 15}),
    ]
    
    for query, expected in time_tests:
        result = extract_time_range_from_query(query)
        print(f"'{query}' -> {result}")
    
    # Top N
    print("\n--- Top N Extraction ---")
    topn_tests = [
        ("top 100 locations", 100),
        ("find top 50", 50),
    ]
    
    for query, expected in topn_tests:
        result = extract_top_n_from_query(query)
        status = "✅" if result == expected else "❌"
        print(f"{status} '{query}' -> {result}")


def test_full_parameter_resolution():
    """Test complete parameter resolution for tools."""
    print("\n" + "=" * 60)
    print("TEST: Full Parameter Resolution")
    print("=" * 60)
    
    base = "/data/base.nc"
    scenario = "/data/scenario.nc"
    
    test_cases = [
        ("Calculate average depth for base file", "calculate_average_depth"),
        ("Plot wave velocity for scenario file", "calculate_max_wave_velocity"),
        ("Show point (40.699429, -8.756169) elevation over time", "get_point_time_series"),
        ("Find top 100 locations above average velocity", "find_velocity_above_average"),
        ("Calculate elevation between 10 AM and 3 PM", "calculate_time_filtered_elevation"),
    ]
    
    for query, tool in test_cases:
        params = resolve_parameters(tool, query, base, scenario)
        print(f"\nQuery: {query}")
        print(f"Tool: {tool}")
        print(f"Parameters: {params}")


def test_openai_schema():
    """Verify OpenAI function calling schema is valid."""
    print("\n" + "=" * 60)
    print("TEST: OpenAI Function Schema")
    print("=" * 60)
    
    schema = get_tool_definitions()
    
    print(f"Generated {len(schema)} function definitions")
    
    for func in schema[:3]:  # Show first 3
        print(f"\n📌 {func['function']['name']}")
        print(f"   {func['function']['description'][:60]}...")
        print(f"   Required: {func['function']['parameters'].get('required', [])}")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("GROUND TRUTH TOOLS TEST SUITE")
    print("=" * 60)
    
    test_tool_registry()
    test_query_matching()
    test_parameter_extraction()
    test_full_parameter_resolution()
    test_openai_schema()
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
