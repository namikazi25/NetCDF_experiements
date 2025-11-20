from agents.planner import plan_task
from agents.evaluator import evaluate_plan
from agents.executor import generate_and_execute_code
from agents.synthesizer import synthesize_response
from memory_service import save_memory_entry

def run_orchestrator(query: str, metadata: dict, netcdf_path: str, scenario_path: str = None) -> dict:
    """
    Manages the multi-agent workflow: Plan -> Evaluate -> Execute -> Synthesize.
    Returns a dict with 'response', 'images', and 'steps' (for UI visualization).
    """
    steps_log = []
    
    # 1. Planning
    steps_log.append({"stage": "Planning", "status": "running"})
    plan = plan_task(query, metadata)
    steps_log[-1]["status"] = "complete"
    steps_log[-1]["output"] = plan
    
    # 2. Evaluation
    steps_log.append({"stage": "Evaluation", "status": "running"})
    evaluation = evaluate_plan(query, plan, metadata)
    steps_log[-1]["status"] = "complete"
    steps_log[-1]["output"] = evaluation
    
    if not evaluation.get("approved", True):
        # In a real system, we would loop back to planner with feedback.
        # For prototype, we'll just warn and proceed or stop.
        # Let's proceed but note the warning.
        steps_log.append({"stage": "Warning", "status": "warning", "output": "Plan was flagged but proceeding."})
    
    # 3. Execution
    steps_log.append({"stage": "Execution", "status": "running"})
    exec_result = generate_and_execute_code(query, plan, netcdf_path, scenario_path)
    steps_log[-1]["status"] = "complete" if exec_result["success"] else "failed"
    steps_log[-1]["output"] = {"stdout": exec_result.get("stdout"), "stderr": exec_result.get("stderr")}
    
    # === NEW: MEMORY STORAGE ===
    if exec_result["success"]:
        # If the code ran without crashing, we assume it's a "good recipe"
        # We save the code associated with this query
        code_to_save = exec_result.get("code_generated", "")
        if code_to_save:
            save_memory_entry(query, code_to_save, plan.get("thought", ""))
            # Log learning
            steps_log.append({"stage": "Learning", "status": "complete", "output": "Saved successful code to memory."})
    # ===========================
    
    # 4. Synthesis
    steps_log.append({"stage": "Synthesis", "status": "running"})
    final_response = synthesize_response(query, plan, exec_result)
    steps_log[-1]["status"] = "complete"
    
    return {
        "response": final_response,
        "images": exec_result.get("images", []),
        "steps_log": steps_log
    }
