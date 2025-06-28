import pandas as pd
import janitor
import os
from operation_creator import OperationCreator
from llm_interface import ask_llm

def load_and_clean_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    print("📥 Loading and cleaning dataset...")

    # Load based on file extension
    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    elif ext == ".csv":
        df = pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file type. Please provide a .csv or .xlsx file.")

    # Clean and process
    df = (
        df.clean_names()
          .remove_empty()
    )

    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            pass  # Keep as-is if conversion fails

        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].mean())

    return df.convert_dtypes()


# === Main Program ===

file_path = input("📁 Enter path to your data file (.csv or .xlsx): ").strip()

try:
    df = load_and_clean_file(file_path)
    print("\n🧾 Columns available:", list(df.columns))

    user_query = input("🗣️ What would you like to do with your data? (e.g., 'Perform regression on height and weight'): ")

    parsed = ask_llm(user_query, list(df.columns))

    if not parsed or not parsed.get("operation"):
        raise ValueError("⚠️ Could not understand your request.")

    operation = parsed["operation"]
    x_col = parsed.get("x_col")
    y_col = parsed.get("y_col")

   
    if operation == "multi_regression":
        if not isinstance(x_col, list) or not x_col:
            raise ValueError("❌ No independent variables (x_col) provided.")
        if not isinstance(y_col, str) or not y_col.strip():
            print("⚠️ The LLM couldn't fully understand your query.")
            y_col = input("🔸 Please enter the dependent variable: ").strip()

    elif operation in ["regression", "rank_correlation"]:
        if not x_col or not y_col:
            raise ValueError("❌ Missing x or y column for regression or correlation.")

    op = OperationCreator(operation)
    result = op.stat_operation(df, x_col=x_col, y_col=y_col)

    print("\n✅ === Result ===")
    if isinstance(result, dict):
        for key, value in result.items():
            if isinstance(value, dict):
                print(f"\n🔹 {key}:")
                for sub_key, sub_val in value.items():
                    print(f"   • {sub_key}: {sub_val}")
            else:
                print(f"{key}: {value}")
    else:
        print(result)

except Exception as e:
    print(f"\n❌ Error: {e}")
