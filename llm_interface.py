import ollama
import re
import json

def ask_llm(user_prompt, df_columns):
    column_str = ", ".join([f"'{col}'" for col in df_columns])

    system_prompt = f"""
You are a helpful data analysis assistant. The dataset contains the following columns: {column_str}.

Your job is to:
1. Identify the user's intent: one of:
   - "regression"
   - "multi_regression"
   - "basic stats"
   - "rank_correlation"

2. Identify relevant columns:
   - For 'regression' or 'rank_correlation': provide one 'x_col' (list of one string) and one 'y_col' (string).
   - For 'multi_regression': provide multiple 'x_col' variables (list of strings) and one 'y_col' (string).
   - For 'basic stats': set both 'x_col' and 'y_col' to null.

⚠️ Do NOT set 'y_col' to null for regression/multi_regression/rank_correlation.

Only output a JSON object like this:

{{
  "operation": "multi_regression",
  "x_col": ["height_cm", "weight_kg"],
  "y_col": "performance_score"
}}

Now interpret the user query below and respond ONLY with the JSON:
"""

    full_prompt = f"{system_prompt}\nUser Query: {user_prompt}"

    response = ollama.chat(
        model="llama3:latest", 
        messages=[{"role": "user", "content": full_prompt}]
    )

    try:
        content = response['message']['content']
        match = re.search(r'\{[\s\S]*?\}', content)
        if match:
            json_like = match.group()
            parsed = json.loads(json_like)

            # Defensive fallback
            if parsed.get("operation") in ["regression", "multi_regression", "rank_correlation"]:
                if not parsed.get("y_col"):
                    raise ValueError("Missing y_col for statistical operation.")

            return parsed
        else:
            raise ValueError("No valid JSON found in model output.")
    except Exception as e:
        print(f"LLM parsing error: {e}")
        print("Raw response:", response['message']['content'])
        return None
