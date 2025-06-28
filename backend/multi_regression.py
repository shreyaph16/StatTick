import pandas as pd
import statsmodels.api as sm

def compute(df, x_cols, y_col):
    if not isinstance(x_cols, list):
        raise TypeError("x_cols must be a list.")
    if not isinstance(y_col, str):
        raise TypeError("y_col must be a string.")

    missing = [col for col in x_cols + [y_col] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    data = df[x_cols + [y_col]].copy()

    # Force convert all to numeric
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # Drop NaNs
    data = data.dropna()

    if data.empty:
        raise ValueError("No data left after cleaning.")

    # Debug: Show dtypes
    print("\n📊 Data types used in regression:")
    print(data.dtypes)

    # Debug: Show top few rows
    print("\n🔍 Sample data:")
    print(data.head())

    # Extract X and Y
    X = data[x_cols]
    Y = data[y_col]

    # Add intercept
    X = sm.add_constant(X, has_constant='add')

    # Check that X and Y are clean numeric dtypes
    if not all(pd.api.types.is_numeric_dtype(X[col]) for col in X.columns):
        raise TypeError("X contains non-numeric columns.")
    if not pd.api.types.is_numeric_dtype(Y):
        raise TypeError("Y is not numeric.")

    # Final conversion to numpy arrays just to be extra sure
    X_np = X.to_numpy(dtype=float)
    Y_np = Y.to_numpy(dtype=float)

    # Fit model
    model = sm.OLS(Y_np, X_np).fit()

    return {
        "Dependent Variable": y_col,
        "Independent Variables": x_cols,
        "R-squared": round(model.rsquared, 4),
        "Adjusted R-squared": round(model.rsquared_adj, 4),
        "F-statistic": round(model.fvalue, 4),
        "p-value (overall)": round(model.f_pvalue, 4),
        "Coefficients": dict(zip(X.columns, model.params.round(4))),
        "Summary": str(model.summary())
    }
