"""Proper probability scores, calibration bins and time-block uncertainty."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, log_loss, roc_auc_score


def score(y: np.ndarray, p: np.ndarray) -> dict:
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    if len(y) == 0 or y.shape != p.shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid probability vector")
    return {
        "n": len(y), "positive_rate": float(y.mean()),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.column_stack([1-p, p]), labels=[0, 1])),
        "accuracy": float(accuracy_score(y, p >= 0.5)),
        "balanced_accuracy": float(balanced_accuracy_score(y, p >= 0.5)),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
    }


def reliability(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict]:
    y, p = np.asarray(y), np.asarray(p)
    assignment = np.minimum((p * bins).astype(int), bins-1)
    result = []
    for i in range(bins):
        mask = assignment == i
        if mask.any():
            result.append({"bin": i, "count": int(mask.sum()),
                           "mean_probability": float(p[mask].mean()),
                           "observed_frequency": float(y[mask].mean())})
    return result


def paired_block_interval(predictions: pd.DataFrame, left: str, right: str,
                          block: int = 20, repeats: int = 2000, seed: int = 42) -> dict:
    """Paired circular blocks within years, conditional on already fitted models."""
    if block < 1 or repeats < 2:
        raise ValueError("Invalid bootstrap configuration")
    rng = np.random.default_rng(seed)
    groups = []
    for _, group in predictions.groupby("year", sort=True):
        y = group["actual_up"].to_numpy()
        groups.append((group[left].to_numpy()-y)**2 - (group[right].to_numpy()-y)**2)
    estimates = []
    for _ in range(repeats):
        sample = []
        for differences in groups:
            n = len(differences)
            starts = rng.integers(0, n, size=int(np.ceil(n/block)))
            ids = ((starts[:, None] + np.arange(block)) % n).ravel()[:n]
            sample.append(differences[ids])
        estimates.append(float(np.concatenate(sample).mean()))
    differences = np.concatenate(groups)
    return {"comparison": f"{left} minus {right}", "mean_brier_difference": float(differences.mean()),
            "ci95_low": float(np.quantile(estimates, .025)), "ci95_high": float(np.quantile(estimates, .975)),
            "block_observations": block, "replicates": repeats,
            "interpretation": "Negative favors left; conditional on fitted models and this data vintage, not a test of tradability."}


def population_stability(reference: np.ndarray, current: np.ndarray) -> float:
    """Descriptive PSI with reference-only quantile bins; no automated decision."""
    inner = np.unique(np.quantile(reference, np.linspace(.1, .9, 9)))
    edges = np.concatenate(([-np.inf], inner, [np.inf]))
    a = np.histogram(reference, edges)[0] + .5
    b = np.histogram(current, edges)[0] + .5
    a, b = a/a.sum(), b/b.sum()
    return float(np.sum((b-a) * np.log(b/a)))
