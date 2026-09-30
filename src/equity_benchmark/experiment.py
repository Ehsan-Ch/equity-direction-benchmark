"""Run the complete fixed-protocol benchmark and export auditable predictions."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from threadpoolctl import threadpool_limits

from .data import load_data
from .features import make_dataset
from .metrics import paired_block_interval, population_stability, reliability, score
from .models import SigmoidCalibrator, candidates
from .splits import annual_fold


@dataclass(frozen=True)
class Config:
    start: str = "1990-01-01"
    end: str = "2025-12-31"
    first_test_year: int = 2020
    last_test_year: int = 2025
    gap: int = 5
    seed: int = 42
    bootstrap_repeats: int = 2000

    def validate(self) -> None:
        if self.first_test_year > self.last_test_year or self.gap < 0:
            raise ValueError("Invalid years or gap")
        if pd.Timestamp(self.end) < pd.Timestamp(f"{self.last_test_year}-12-31"):
            raise ValueError("End must include the whole final test year")
        if pd.Timestamp(self.start).year > self.first_test_year - 5:
            raise ValueError("Need at least three training years before tuning and calibration")


def run(archive: Path, output: Path, artifact_dir: Path, config: Config,
        expected_sha256: str | None = None) -> dict:
    config.validate()
    data, provenance = load_data(archive, expected_sha256)
    if data.index[-1] < pd.Timestamp(f"{config.last_test_year}-12-28"):
        raise ValueError("Provider snapshot does not cover the final test year")
    x, y = make_dataset(data, config.start, config.end)
    output.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    all_predictions, fold_reports = [], []
    candidate_bank = candidates(config.seed)
    final_artifact = None
    for year in range(config.first_test_year, config.last_test_year+1):
        fold = annual_fold(x.index, year, config.gap)
        print(f"Evaluating {year}: train={len(fold.train)}, tune={len(fold.tune)}, "
              f"calibrate={len(fold.calibrate)}, test={len(fold.test)}", flush=True)
        validation = []
        train_prior = float(y.iloc[fold.train].mean())
        validation.append({"candidate": "historical_prior", "family": "prior",
                           **score(y.iloc[fold.tune], np.full(len(fold.tune), train_prior))})
        with threadpool_limits(limits=1):
            for name, template in candidate_bank.items():
                model = clone(template)
                start_time = time.perf_counter()
                model.fit(x.iloc[fold.train], y.iloc[fold.train])
                p = model.predict_proba(x.iloc[fold.tune])[:, 1]
                validation.append({"candidate": name, "family": name.split("_")[0],
                                   **score(y.iloc[fold.tune], p),
                                   "fit_seconds": time.perf_counter()-start_time})
            # Candidate/family decisions are completed before calibration/test labels are used.
            best_by_family = {family: min([r for r in validation if r["family"] == family],
                                          key=lambda r: (r["brier"], r["candidate"]))
                              for family in ["logistic", "histgb"]}
            selected = min(validation, key=lambda r: (r["brier"], r["candidate"]))
            refit = np.concatenate([fold.train, fold.tune])
            known = np.concatenate([refit, fold.calibrate])
            prior = float(y.iloc[known].mean())
            table = pd.DataFrame({"date": x.index[fold.test].strftime("%Y-%m-%d"),
                                  "year": year, "actual_up": y.iloc[fold.test].to_numpy(),
                                  "coin_flip": .5, "historical_prior": prior})
            fitted = {}
            for family, best in best_by_family.items():
                model = clone(candidate_bank[best["candidate"]])
                model.fit(x.iloc[refit], y.iloc[refit])
                calibration_p = model.predict_proba(x.iloc[fold.calibrate])[:, 1]
                calibrator = SigmoidCalibrator().fit(calibration_p, y.iloc[fold.calibrate].to_numpy())
                raw_test = model.predict_proba(x.iloc[fold.test])[:, 1]
                table[f"{family}_raw"] = raw_test
                table[f"{family}_sigmoid"] = calibrator.predict(raw_test)
                fitted[family] = {"model": model, "calibrator": calibrator,
                                  "candidate": best["candidate"]}
        selected_column = "historical_prior" if selected["family"] == "prior" else selected["family"]+"_sigmoid"
        table["selected"] = table[selected_column]
        all_predictions.append(table)
        drift = sorted([{"feature": column, "psi": population_stability(
            x.iloc[refit][column].to_numpy(), x.iloc[fold.test][column].to_numpy())}
            for column in x.columns], key=lambda r: -r["psi"])
        fold_reports.append({"test_year": year, "windows": fold.audit(x.index),
                             "validation_candidates": validation,
                             "selected_candidate": selected["candidate"],
                             "selected_output": selected_column,
                             "test_metrics": {column: score(table.actual_up, table[column])
                                              for column in probability_columns(table)},
                             "feature_psi": drift})
        final_artifact = {"schema_version": 1, "test_year": year,
                          "fit_data_through": str(x.index[fold.calibrate][-1].date()),
                          "earliest_target_date": str(x.index[fold.test][0].date()),
                          "feature_names": list(x.columns), "selected_family": selected["family"],
                          "historical_prior": prior, "estimators": fitted,
                          "source_sha256": provenance["archive_sha256"], "config": asdict(config)}
    predictions = pd.concat(all_predictions, ignore_index=True)
    metrics = {c: score(predictions.actual_up, predictions[c]) for c in probability_columns(predictions)}
    report = {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "config": asdict(config), "data": provenance,
              "data_rows_used": len(x), "features": list(x.columns),
              "environment": {"python": platform.python_version(), **{
                  p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scikit-learn", "matplotlib", "joblib"]}},
              "aggregate_metrics": metrics, "folds": fold_reports,
              "paired_uncertainty": paired_block_interval(predictions, "selected", "historical_prior",
                                                            repeats=config.bootstrap_repeats, seed=config.seed),
              "reliability": {c: reliability(predictions.actual_up, predictions[c])
                              for c in probability_columns(predictions)}}
    predictions.to_csv(output/"predictions.csv", index=False, float_format="%.12g")
    (output/"metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    (output/"data_manifest.json").write_text(json.dumps(provenance, indent=2)+"\n")
    joblib.dump(final_artifact, artifact_dir/"final_fold.joblib")
    from .reporting import write_report
    write_report(report, predictions, output)
    return report


def probability_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in ["date", "year", "actual_up"]]
