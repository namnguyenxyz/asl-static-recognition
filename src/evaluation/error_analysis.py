"""Build auditable error-analysis tables from per-image predictions."""
from __future__ import annotations

import pandas as pd


REQUIRED_PREDICTION_COLUMNS = {
    "relative_path", "true_label", "predicted_label", "confidence", "correct"
}


def summarize_errors(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return error examples, confusion pairs, and per-class error rates.

    The function intentionally uses only exported predictions.  It therefore
    keeps the qualitative review traceable to a specific evaluation artifact.
    """
    missing = REQUIRED_PREDICTION_COLUMNS - set(predictions.columns)
    if missing:
        raise ValueError(f"Predictions are missing required columns: {sorted(missing)}")
    if predictions.empty:
        raise ValueError("Predictions must contain at least one evaluated image.")

    frame = predictions.copy()
    frame["correct"] = frame["correct"].astype(bool)
    errors = frame.loc[~frame["correct"]].copy()
    errors = errors.sort_values(["true_label", "predicted_label", "confidence", "relative_path"])

    support = frame.groupby("true_label", as_index=False).size().rename(columns={"size": "support"})
    per_class = (
        errors.groupby("true_label", as_index=False)
        .size()
        .rename(columns={"size": "errors"})
        .merge(support, on="true_label", how="right")
        .fillna({"errors": 0})
    )
    per_class["errors"] = per_class["errors"].astype(int)
    per_class["error_rate"] = per_class["errors"] / per_class["support"]
    per_class = per_class.sort_values(["error_rate", "errors", "true_label"], ascending=[False, False, True])

    pairs = (
        errors.groupby(["true_label", "predicted_label"], as_index=False)
        .size()
        .rename(columns={"size": "errors"})
        .merge(support, on="true_label", how="left")
    )
    pairs["rate_within_true_label"] = pairs["errors"] / pairs["support"]
    pairs = pairs.sort_values(["errors", "rate_within_true_label", "true_label", "predicted_label"], ascending=[False, False, True, True])
    return errors, pairs, per_class
