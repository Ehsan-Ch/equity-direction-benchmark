"""Every feature at target date t uses observations strictly before t."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import COLUMNS


def make_features(data: pd.DataFrame) -> pd.DataFrame:
    if list(data.columns) != COLUMNS:
        raise ValueError("Expected ordered factor columns")
    if data.index.has_duplicates or not data.index.is_monotonic_increasing:
        raise ValueError("Dates must be unique and sorted")
    if not np.isfinite(data.to_numpy()).all():
        raise ValueError("Factors contain missing/non-finite values")
    features = {}
    for factor in ["Mkt-RF", "SMB", "HML"]:
        for lag in [1, 2, 5, 10]:
            features[f"{factor}_lag_{lag}"] = data[factor].shift(lag)
        for window in [5, 21, 63]:
            history = data[factor].shift(1).rolling(window, min_periods=window)
            features[f"{factor}_mean_{window}"] = history.mean()
            features[f"{factor}_std_{window}"] = history.std(ddof=1)
    features["RF_lag_1"] = data["RF"].shift(1)
    return pd.DataFrame(features, index=data.index)


def make_dataset(data: pd.DataFrame, start: str, end: str) -> tuple[pd.DataFrame, pd.Series]:
    """Label: market excess return strictly positive; zero is class 0."""
    x = make_features(data).dropna().loc[start:end]
    y = (data.loc[x.index, "Mkt-RF"] > 0).astype(int).rename("up")
    if x.empty or not np.isfinite(x.to_numpy()).all():
        raise ValueError("No valid model observations")
    return x, y
