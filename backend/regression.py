import pandas as pd
from scipy.stats import linregress

def compute(df, x_col=None, y_col=None):
    """
    Perform linear regression on two numeric columns.
    """
    # Handle x_col/y_col as lists (from LLM)
    if isinstance(x_col, list):
        if len(x_col) != 1:
            raise ValueError("Regression requires exactly one independent variable.")
        x_col = x_col[0]

    if isinstance(y_col, list):
        if len(y_col) != 1:
            raise ValueError("Regression requires exactly one dependent variable.")
        y_col = y_col[0]

    # Prompt user if not provided
    if x_col is None or y_col is None:
        print("Available columns:", list(df.columns))
        x_col = input("Enter the column name for X (independent variable): ").strip()
        y_col = input("Enter the column name for Y (dependent variable): ").strip()

    # Validate existence
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"One or both columns '{x_col}' or '{y_col}' do not exist in the DataFrame.")

    # Validate numeric type
    if not pd.api.types.is_numeric_dtype(df[x_col]) or not pd.api.types.is_numeric_dtype(df[y_col]):
        raise TypeError(f"Columns '{x_col}' and '{y_col}' must be numeric for regression.")

    # Drop NaN rows
    data = df[[x_col, y_col]].dropna()
    if data.empty or len(data) < 2:
        raise ValueError("Not enough valid data points for regression after dropping missing values.")

    print(f"📊 Running regression on {len(data)} valid rows.")

    # Perform linear regression
    result = linregress(data[x_col], data[y_col])

    return {
        "X column": x_col,
        "Y column": y_col,
        "intercept (a)": float(result.intercept),
        "slope (b)": float(result.slope),
        "r_squared": float(result.rvalue ** 2),
        "p_value": float(result.pvalue),
        "std_err": float(result.stderr)
    }
