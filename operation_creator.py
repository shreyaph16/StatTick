from backend import regression, basic_stats, rank_correlation, multi_regression

class OperationCreator:
    def __init__(self, operation: str):
        # Normalize operation name to match keys in operations_map
        self.operation = operation.lower().replace("_", " ")

        self.operations_map = {
            "regression": regression.compute,
            "basic stats": basic_stats.compute,
            "rank correlation": rank_correlation.compute,  # Also normalize here
            "multi regression": multi_regression.compute,
        }

    def stat_operation(self, df, x_col=None, y_col=None):
        compute_fn = self.operations_map.get(self.operation)

        if not compute_fn:
            raise ValueError(f"Unsupported operation: {self.operation}")

        # basic stats requires only df, others use x_col and y_col
        if self.operation == "basic stats":
            return compute_fn(df)
        else:
            return compute_fn(df, x_col, y_col)
