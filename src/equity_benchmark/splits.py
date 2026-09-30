"""Expanding training, separate tuning and calibration years, annual test."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    year: int
    train: np.ndarray
    tune: np.ndarray
    calibrate: np.ndarray
    test: np.ndarray

    def audit(self, index: pd.DatetimeIndex) -> dict:
        return {name: {"rows": len(getattr(self, name)),
                       "start": str(index[getattr(self, name)][0].date()),
                       "end": str(index[getattr(self, name)][-1].date())}
                for name in ["train", "tune", "calibrate", "test"]}


def annual_fold(index: pd.DatetimeIndex, year: int, gap: int = 5) -> Fold:
    if gap < 0 or index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("Invalid gap or date index")
    def rows(mask: np.ndarray, trim: bool = True) -> np.ndarray:
        selected = np.flatnonzero(mask)
        return selected[:-gap] if trim and gap else selected
    train = rows(index.year < year - 2)
    tune = rows(index.year == year - 2)
    calibrate = rows(index.year == year - 1)
    test = rows(index.year == year, trim=False)
    if len(train) < 500 or min(len(tune), len(calibrate), len(test)) < 30:
        raise ValueError(f"Insufficient observations for test year {year}")
    stages = [train, tune, calibrate, test]
    if any(a[-1] + gap >= b[0] for a, b in zip(stages, stages[1:])):
        raise ValueError("Time stages overlap or embargo is too short")
    return Fold(year, train, tune, calibrate, test)
