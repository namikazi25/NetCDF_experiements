from llm_service import client, MODEL

def synthesize_response(query: str, plan: dict, execution_result: dict) -> str:
    """
    Synthesizes the final natural language response based on execution results.
    """
    system_prompt = """You are a Data Analyst.
    Synthesize a final answer for the user based on the analysis results.
    
    - Summarize the findings from the code output.
    - If plots were generated, mention them (e.g., "As shown in the plot...").
    - Be concise and professional.
    """
    
    context = f"""
    User Query: {query}
    
    Plan Executed: {plan.get('steps')}
    
    Code Output:
    {execution_result.get('stdout')}
    
    Errors (if any):
    {execution_result.get('stderr')}
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": context}
    ]
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error synthesizing response: {e}"
