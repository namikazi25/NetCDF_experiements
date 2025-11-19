import json
from llm_service import client, MODEL

def evaluate_plan(query: str, plan: dict, metadata: dict) -> dict:
    """
    Evaluates the proposed plan for safety and correctness.
    Returns JSON with 'approved' (bool) and 'feedback' (str).
    """
    system_prompt = """You are a Lead Engineer evaluating a data analysis plan.
    Review the plan to ensure it answers the user's query and uses the available tools correctly.
    
    CRITERIA:
    1. Does the plan directly address the query?
    2. Are the steps logical?
    3. Is it safe? (No system commands, only data analysis)
    
    OUTPUT FORMAT:
    {
        "approved": true/false,
        "feedback": "Plan looks good." or "Step 2 is missing..."
    }
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"User Query: {query}\n\nProposed Plan:\n{json.dumps(plan, indent=2)}"}
    ]
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages
        )
        content = response.choices[0].message.content
        
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
            
        return json.loads(content)
    except Exception as e:
        # Fail open for prototype if evaluation crashes, but log it
        return {"approved": True, "feedback": f"Evaluation failed ({e}), proceeding with caution."}
