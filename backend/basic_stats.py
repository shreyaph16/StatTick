import pandas as pd

def compute(df, cols=None):
    """
    Returns descriptive statistics (count, mean, std, etc.) for selected numeric columns.
    """
    # Ask user to select columns if not provided
    if cols is None:
        print("Available columns:", list(df.columns))
        input_str = input("Enter column names for stats (comma-separated): ")
        cols = [col.strip() for col in input_str.split(",")]

    # Validate existence
    for col in cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' does not exist in the DataFrame.")

    # Validate numeric types
    for col in cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"Column '{col}' must be numeric to compute statistics.")

    # Compute and return stats
    stats = df[cols].describe().to_dict()
    return stats
