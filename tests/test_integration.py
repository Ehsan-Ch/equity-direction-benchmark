"""Offline end-to-end fixture and independent audit of committed real results."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from equity_benchmark.cli import replay
from equity_benchmark.experiment import Config, probability_columns, run
from equity_benchmark.metrics import score
from test_research import archive, fixture


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        data = fixture(1600).loc[:"2020-12-31"] * 100
        data.index = data.index.strftime("%Y%m%d")
        data.index.name = ""
        cls.path = cls.root/"fixture.zip"
        cls.path.write_bytes(archive(data.to_csv()))
        with contextlib.redirect_stdout(io.StringIO()):
            cls.result = run(cls.path, cls.root/"reports", cls.root/"artifacts",
                             Config(start="2015-01-01", end="2020-12-31", first_test_year=2020,
                                    last_test_year=2020, bootstrap_repeats=50))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_saved_artifact_replays_exact_forward_prediction(self):
        row = pd.read_csv(self.root/"reports/predictions.csv").iloc[20]
        replayed = replay(self.path, self.root/"artifacts/final_fold.joblib", row.date)
        self.assertAlmostEqual(replayed["positive_excess_return_probability"], row.selected, places=10)
        self.assertLess(replayed["model_fit_data_through"], row.date)
        self.assertLess(replayed["features_through"], row.date)

    def test_replay_rejects_training_dates(self):
        with self.assertRaisesRegex(ValueError, "precedes"):
            replay(self.path, self.root/"artifacts/final_fold.joblib", "2019-06-03")

    def test_full_pipeline_exports_required_artifacts(self):
        for file in ["metrics.json", "predictions.csv", "data_manifest.json", "RESULTS.md",
                     "figures/benchmark.svg", "figures/calibration.svg", "figures/rolling_loss.svg"]:
            self.assertTrue((self.root/"reports"/file).is_file())
        self.assertEqual(len(self.result["features"]), 31)


class PublishedResultAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]/"reports"
        cls.predictions = pd.read_csv(root/"predictions.csv")
        cls.report = json.loads((root/"metrics.json").read_text())

    def test_independently_reconstructed_scores(self):
        for model in probability_columns(self.predictions):
            actual = score(self.predictions.actual_up, self.predictions[model])
            expected = self.report["aggregate_metrics"][model]
            for metric in ["brier", "log_loss", "accuracy", "balanced_accuracy", "roc_auc"]:
                self.assertAlmostEqual(actual[metric], expected[metric], places=9)

    def test_unique_dates_fixed_horizon_and_selection_trace(self):
        p = self.predictions
        self.assertFalse(p.date.duplicated().any())
        self.assertTrue(p.date.is_monotonic_increasing)
        self.assertEqual(set(p.year), set(range(2020, 2026)))
        for fold in self.report["folds"]:
            rows = p[p.year == fold["test_year"]]
            np.testing.assert_array_equal(rows.selected, rows[fold["selected_output"]])
            winner = min(fold["validation_candidates"], key=lambda r: (r["brier"], r["candidate"]))
            self.assertEqual(winner["candidate"], fold["selected_candidate"])
            self.assertEqual(len(rows), fold["windows"]["test"]["rows"])


if __name__ == "__main__":
    unittest.main()
