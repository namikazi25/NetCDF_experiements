"""
Enhanced Orchestrator with Ground Truth Tool Priority

This orchestrator:
1. First checks if a ground truth tool matches the query
2. If match found, executes the proven code directly
3. Falls back to LLM agent workflow if no match

This gives you:
- Consistent results for known query patterns
- Faster execution (no LLM code generation needed)
- Guaranteed correct output for matching queries
"""

from typing import Dict, Any, Optional
from ground_truth_tools import (
    TOOL_REGISTRY,
    execute_tool,
    match_query_to_tool
)
from tool_caller import (
    resolve_parameters,
    GroundTruthToolCaller
)

# Import original agents for fallback
from agents.planner import plan_task
from agents.evaluator import evaluate_plan
from agents.executor import generate_and_execute_code
from agents.synthesizer import synthesize_response
from memory_service import save_memory_entry


def run_orchestrator_enhanced(
    query: str,
    metadata_bundle: dict,
    netcdf_path: str,
    scenario_path: str = None
) -> Dict[str, Any]:
    """
    Enhanced orchestrator with ground truth tool priority.
    
    Flow:
    1. Try to match query to a ground truth tool
    2. If match + valid params → Execute ground truth
    3. If no match → Run original multi-agent workflow
    """
    steps_log = []
    
    # =========================================================================
    # PHASE 1: Ground Truth Tool Matching
    # =========================================================================
    steps_log.append({"stage": "Tool Matching", "status": "running"})
    
    tool_name = match_query_to_tool(query)
    
    if tool_name:
        steps_log[-1]["status"] = "complete"
        steps_log[-1]["output"] = f"Matched ground truth tool: {tool_name}"
        
        # Try to execute with ground truth
        result = _try_ground_truth_execution(
            tool_name, query, netcdf_path, scenario_path, steps_log
        )
        
        if result:
            return result
        
        # Ground truth failed, fall through to agent workflow
        steps_log.append({
            "stage": "Ground Truth Fallback",
            "status": "skipped",
            "output": "Missing parameters or execution failed, using agent workflow"
        })
    else:
        steps_log[-1]["status"] = "complete"
        steps_log[-1]["output"] = "No ground truth match, using agent workflow"
    
    # =========================================================================
    # PHASE 2: Original Multi-Agent Workflow (Fallback)
    # =========================================================================
    return _run_agent_workflow(query, metadata_bundle, netcdf_path, scenario_path, steps_log)


def _try_ground_truth_execution(
    tool_name: str,
    query: str,
    netcdf_path: str,
    scenario_path: str,
    steps_log: list
) -> Optional[Dict[str, Any]]:
    """
    Attempt to execute a ground truth tool.
    Returns result dict if successful, None if failed.
    """
    steps_log.append({"stage": "Parameter Resolution", "status": "running"})
    
    # Resolve parameters from query
    params = resolve_parameters(tool_name, query, netcdf_path, scenario_path)
    
    # Check required parameters
    tool_info = TOOL_REGISTRY[tool_name]
    required = [k for k, v in tool_info["parameters"].items() if "default" not in v]
    missing = [r for r in required if r not in params or params[r] is None]
    
    if missing:
        steps_log[-1]["status"] = "failed"
        steps_log[-1]["output"] = f"Missing required parameters: {missing}"
        return None
    
    steps_log[-1]["status"] = "complete"
    steps_log[-1]["output"] = f"Resolved: {params}"
    
    # Execute tool
    steps_log.append({"stage": "Ground Truth Execution", "status": "running"})
    
    try:
        result = execute_tool(tool_name, **params)
        
        if "error" in result:
            steps_log[-1]["status"] = "failed"
            steps_log[-1]["output"] = result["error"]
            return None
        
        steps_log[-1]["status"] = "complete"
        steps_log[-1]["output"] = "Execution successful"
        
        # Format response
        return _format_ground_truth_response(tool_name, result, query, steps_log)
        
    except Exception as e:
        steps_log[-1]["status"] = "failed"
        steps_log[-1]["output"] = str(e)
        return None


def _format_ground_truth_response(
    tool_name: str,
    result: Dict,
    query: str,
    steps_log: list
) -> Dict[str, Any]:
    """Format ground truth result into standard response format."""
    
    steps_log.append({"stage": "Response Formatting", "status": "running"})
    
    # Extract images
    images = []
    if "image" in result:
        images.append(result["image"])
    
    # Generate summary based on tool type
    summary = _generate_tool_summary(tool_name, result)
    
    steps_log[-1]["status"] = "complete"
    
    return {
        "response": summary,
        "images": images,
        "steps_log": steps_log,
        "source": "ground_truth",
        "tool_used": tool_name,
        "raw_data": result  # Include raw data for debugging/display
    }


def _generate_tool_summary(tool_name: str, result: Dict) -> str:
    """Generate natural language summary for tool results."""
    
    summaries = {
        "get_parameter_shapes": lambda r: (
            f"The file contains {len(r.get('table', []))} parameters.\n\n"
            f"| Parameter | Shape |\n|---|---|\n" +
            "\n".join(f"| {row['Parameter']} | {row['Shape']} |" 
                     for row in r.get('table', [])[:15]) +
            ("\n| ... | ... |" if len(r.get('table', [])) > 15 else "")
        ),
        
        "calculate_average_depth": lambda r: (
            f"**Average Depth Analysis**\n\n"
            f"- Mean depth: {r['stats']['mean_depth']:.2f}m\n"
            f"- Maximum depth: {r['stats']['max_depth']:.2f}m\n"
            f"- Minimum depth: {r['stats']['min_depth']:.2f}m\n"
            f"- Total points analyzed: {r['stats']['num_points']}\n\n"
            f"See the plot for spatial distribution."
        ),
        
        "find_min_max_depth": lambda r: (
            f"**Depth Extremes**\n\n"
            f"Maximum depth: {r['max_depth'][0]['depth']:.2f}m "
            f"at ({r['max_depth'][0]['lat']:.4f}, {r['max_depth'][0]['lon']:.4f})\n\n"
            f"Minimum depth: {r['min_depth'][0]['depth']:.2f}m "
            f"at ({r['min_depth'][0]['lat']:.4f}, {r['min_depth'][0]['lon']:.4f})"
        ),
        
        "calculate_max_wave_velocity": lambda r: (
            f"**Maximum Wave Velocity Analysis**\n\n"
            f"- Peak velocity: {r['stats']['max_velocity']:.4f} m/s\n"
            f"- Mean velocity: {r['stats']['mean_velocity']:.4f} m/s\n"
            f"- Points analyzed: {r['stats']['num_points']}\n\n"
            f"Wave velocity calculated as: √(wsh_x² + wsh_y²)"
        ),
        
        "calculate_elevation_difference": lambda r: (
            f"**Elevation Difference (Base - Scenario)**\n\n"
            f"- Mean difference: {r['stats']['mean_diff']:.4f}m\n"
            f"- Max difference: {r['stats']['max_diff']:.4f}m\n"
            f"- Min difference: {r['stats']['min_diff']:.4f}m\n"
            f"- Std deviation: {r['stats']['std_diff']:.4f}m\n\n"
            f"Positive values indicate base elevation is higher than scenario."
        ),
        
        "get_point_time_series": lambda r: (
            f"**Time Series at ({r['point']['lat']:.4f}, {r['point']['lon']:.4f})**\n\n"
            f"Variable: {r['variable']}\n"
            f"Time steps found: {r['num_timesteps']}\n\n"
            f"See the plot for temporal variation."
        ),
        
        "find_max_variable_location": lambda r: (
            f"**Maximum {r['variable']} Location**\n\n"
            f"Maximum value: {r['max_value']:.4f}\n\n"
            f"| Lat | Lon | Time | Value |\n|---|---|---|---|\n" +
            "\n".join(f"| {row['lat']:.4f} | {row['lon']:.4f} | {row['day_time']} | {row[r['variable']]:.4f} |"
                     for row in r.get('table', [])[:5])
        ),
        
        "find_velocity_above_average": lambda r: (
            f"**Locations with Velocity Above Average**\n\n"
            f"Average velocity: {r['average_velocity']:.4f} m/s\n"
            f"Points above average: {r['num_points_above_avg']}\n\n"
            f"See the plot for spatial distribution."
        ),
        
        "calculate_time_filtered_elevation": lambda r: (
            f"**Time-Filtered Elevation Analysis**\n\n"
            f"Time range: {r['time_range']['start_hour']}:00 to {r['time_range']['end_hour']}:00\n"
            f"Points analyzed: {r['num_points']}\n\n"
            f"See the plot for spatial distribution."
        ),
        
        "analyze_bounding_box": lambda r: (
            f"**Bounding Box Analysis**\n\n"
            f"Region: Lat [{r['bounding_box']['lat_min']}, {r['bounding_box']['lat_max']}], "
            f"Lon [{r['bounding_box']['lon_min']}, {r['bounding_box']['lon_max']}]\n\n"
            f"Variable: {r['variable']}\n"
            f"- Mean: {r['stats']['mean']:.4f}\n"
            f"- Max: {r['stats']['max']:.4f}\n"
            f"- Min: {r['stats']['min']:.4f}\n"
            f"- Points: {r['num_points']}"
        ),
        
        "plot_average_elevation": lambda r: (
            f"**Average Surface Elevation**\n\n"
            f"- Mean elevation: {r['stats']['mean_elevation']:.4f}m\n"
            f"- Max elevation: {r['stats']['max_elevation']:.4f}m\n"
            f"- Min elevation: {r['stats']['min_elevation']:.4f}m\n\n"
            f"See the plot for spatial distribution."
        ),
    }
    
    formatter = summaries.get(tool_name)
    if formatter:
        try:
            return formatter(result)
        except Exception as e:
            return f"Analysis complete. Results available in data output. (Format error: {e})"
    
    return f"Analysis complete using {tool_name}. See attached results."


def _run_agent_workflow(
    query: str,
    metadata_bundle: dict,
    netcdf_path: str,
    scenario_path: str,
    steps_log: list
) -> Dict[str, Any]:
    """
    Run the original multi-agent workflow.
    This is the fallback when no ground truth tool matches.
    """
    
    # 1. Planning
    steps_log.append({"stage": "Planning", "status": "running"})
    plan = plan_task(query, metadata_bundle)
    steps_log[-1]["status"] = "complete"
    steps_log[-1]["output"] = plan
    
    # 2. Evaluation
    steps_log.append({"stage": "Evaluation", "status": "running"})
    evaluation = evaluate_plan(query, plan, metadata_bundle)
    steps_log[-1]["status"] = "complete"
    steps_log[-1]["output"] = evaluation
    
    if not evaluation.get("approved", True):
        feedback = evaluation.get("feedback", "Plan needs improvement")
        steps_log.append({
            "stage": "Re-Planning",
            "status": "running",
            "output": f"Plan rejected. Feedback: {feedback}"
        })
        
        plan = plan_task(
            f"{query}\n\nPREVIOUS PLAN FEEDBACK: {feedback}\nPlease revise.",
            metadata_bundle
        )
        steps_log[-1]["status"] = "complete"
        steps_log[-1]["output"] = plan
    
    # 3. Execution
    steps_log.append({"stage": "Execution", "status": "running"})
    exec_result = generate_and_execute_code(query, plan, netcdf_path, scenario_path)
    steps_log[-1]["status"] = "complete" if exec_result["success"] else "failed"
    steps_log[-1]["output"] = {
        "stdout": exec_result.get("stdout"),
        "stderr": exec_result.get("stderr")
    }
    
    # Save to memory if successful
    if exec_result["success"]:
        code_to_save = exec_result.get("code_generated", "")
        if code_to_save:
            save_memory_entry(query, code_to_save, plan.get("thought", ""))
            steps_log.append({
                "stage": "Learning",
                "status": "complete",
                "output": "Saved successful code to memory."
            })
    
    # 4. Synthesis
    steps_log.append({"stage": "Synthesis", "status": "running"})
    final_response = synthesize_response(query, plan, exec_result)
    steps_log[-1]["status"] = "complete"
    
    return {
        "response": final_response,
        "images": exec_result.get("images", []),
        "steps_log": steps_log,
        "source": "agent_workflow"
    }


# ============================================================================
# BACKWARD COMPATIBILITY
# ============================================================================

def run_orchestrator(
    query: str,
    metadata_bundle: dict,
    netcdf_path: str,
    scenario_path: str = None
) -> Dict[str, Any]:
    """
    Drop-in replacement for original orchestrator.
    Uses enhanced version with ground truth priority.
    """
    return run_orchestrator_enhanced(query, metadata_bundle, netcdf_path, scenario_path)


# ============================================================================
# STATISTICS / DEBUGGING
# ============================================================================

def get_ground_truth_coverage() -> Dict[str, Any]:
    """Return statistics about ground truth tool coverage."""
    from ground_truth_tools import QUERY_PATTERNS
    
    return {
        "total_tools": len(TOOL_REGISTRY),
        "tools": list(TOOL_REGISTRY.keys()),
        "query_patterns": {
            name: len(patterns) for name, patterns in QUERY_PATTERNS.items()
        }
    }


if __name__ == "__main__":
    # Quick test
    print("Ground Truth Coverage:")
    coverage = get_ground_truth_coverage()
    print(f"  Tools: {coverage['total_tools']}")
    for tool in coverage['tools']:
        print(f"    - {tool}")
