import pandas as pd

def compute(df, x_col=None, y_col=None):
    """
    Computes correlation between two numeric columns in the dataframe.
    """
    # Ask user to choose columns if not specified
    if x_col is None or y_col is None:
        print("Available columns:", list(df.columns))
        x_col = input("Enter the column name for X: ").strip()
        y_col = input("Enter the column name for Y: ").strip()

    # Validate column existence
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"One or both columns '{x_col}' or '{y_col}' do not exist in the DataFrame.")

    # Validate numeric types
    if not pd.api.types.is_numeric_dtype(df[x_col]) or not pd.api.types.is_numeric_dtype(df[y_col]):
        raise TypeError("Selected columns must be numeric for correlation.")

    # Compute pairwise correlation
    correlation_value = df[[x_col, y_col]].corr().iloc[0, 1]

    return {
        "X column": x_col,
        "Y column": y_col,
        "correlation": float(correlation_value)
    }
