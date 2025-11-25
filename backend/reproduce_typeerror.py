
def simulate_executor_error():
    plan = {
        "thought": "Plan to plot water depth",
        "steps": [
            {"step": 1, "description": "Load the dataset"},
            {"step": 2, "description": "Calculate average depth"}
        ]
    }
    
    try:
        plan_str = "\n".join(plan.get("steps", []))
        print("Success:", plan_str)
    except TypeError as e:
        print("Caught expected error:", e)

if __name__ == "__main__":
    simulate_executor_error()
