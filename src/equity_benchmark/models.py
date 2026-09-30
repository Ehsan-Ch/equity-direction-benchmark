"""Small predeclared candidate grids; preprocessing fitted inside each fold."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def candidates(seed: int) -> dict:
    models = {f"logistic_C{c:g}": make_pipeline(
        StandardScaler(), LogisticRegression(C=c, max_iter=2000, random_state=seed))
        for c in [0.01, 0.1, 1.0]}
    for leaves in [3, 7]:
        models[f"histgb_leaves{leaves}"] = HistGradientBoostingClassifier(
            max_iter=100, learning_rate=0.05, max_leaf_nodes=leaves,
            min_samples_leaf=100, l2_regularization=10.0,
            early_stopping=False, random_state=seed)
    return models


def clip_probability(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)


@dataclass
class SigmoidCalibrator:
    """Fit logit(p) -> y on the separate, recent calibration year only."""
    estimator: LogisticRegression | None = None

    def fit(self, probability: np.ndarray, target: np.ndarray) -> "SigmoidCalibrator":
        if len(np.unique(target)) != 2:
            raise ValueError("Calibration requires both classes")
        p = clip_probability(probability)
        self.estimator = LogisticRegression(C=1000.0, max_iter=2000)
        self.estimator.fit(np.log(p / (1 - p)).reshape(-1, 1), target)
        return self

    def predict(self, probability: np.ndarray) -> np.ndarray:
        if self.estimator is None:
            raise ValueError("Calibrator is not fitted")
        p = clip_probability(probability)
        return self.estimator.predict_proba(np.log(p / (1 - p)).reshape(-1, 1))[:, 1]
