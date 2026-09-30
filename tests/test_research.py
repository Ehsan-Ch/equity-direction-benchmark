"""Integrity tests use generated fixtures only; published results use provider data."""

import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

from equity_benchmark.data import COLUMNS, load_data, parse_archive
from equity_benchmark.features import make_dataset, make_features
from equity_benchmark.metrics import paired_block_interval, population_stability, reliability, score
from equity_benchmark.models import SigmoidCalibrator, candidates
from equity_benchmark.splits import annual_fold


def fixture(n=2200):
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(rng.normal(0, .01, (n, 4)),
                         index=pd.bdate_range("2015-01-01", periods=n, name="date"), columns=COLUMNS)
    frame["RF"] = .0001
    return frame


def archive(text):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as z:
        z.writestr("factors.csv", text)
    return stream.getvalue()


class DataTests(unittest.TestCase):
    def test_parser_units_header_and_footer(self):
        frame, _ = parse_archive(archive("provider header\n,Mkt-RF,SMB,HML,RF\n20200102,1,2,3,.01\n20200103,-1,0,0,.01\nCopyright provider\n"))
        self.assertEqual(len(frame), 2)
        self.assertAlmostEqual(frame.iloc[0, 0], .01)

    def test_missing_sentinel_rejected(self):
        for sentinel in [-99.99, -999]:
            with self.assertRaises(ValueError):
                parse_archive(archive(f",Mkt-RF,SMB,HML,RF\n20200102,{sentinel},0,0,.01\n"))

    def test_duplicate_and_unsorted_rejected(self):
        for dates in [("20200102", "20200102"), ("20200103", "20200102")]:
            with self.assertRaises(ValueError):
                parse_archive(archive(",Mkt-RF,SMB,HML,RF\n"+"\n".join(f"{d},1,0,0,0" for d in dates)))

    def test_unexpected_schema_rejected(self):
        with self.assertRaises(ValueError):
            parse_archive(archive(",Market,SMB,HML,RF\n20200102,1,2,3,4\n"))

    def test_snapshot_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"data.zip"
            path.write_bytes(archive(",Mkt-RF,SMB,HML,RF\n20200102,1,2,3,4\n"))
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                load_data(path, "0"*64)


class TimeIntegrityTests(unittest.TestCase):
    def test_future_and_current_observations_cannot_change_features(self):
        data = fixture(200)
        before = make_features(data)
        changed = data.copy()
        changed.iloc[100:] = 10.0
        after = make_features(changed)
        pd.testing.assert_frame_equal(before.iloc[:101], after.iloc[:101])
        self.assertFalse(before.iloc[101:].equals(after.iloc[101:]))

    def test_lag_and_rolling_mean_align_with_prior_rows(self):
        data = fixture(100)
        x = make_features(data)
        self.assertEqual(x.iloc[70]["Mkt-RF_lag_1"], data.iloc[69]["Mkt-RF"])
        self.assertAlmostEqual(x.iloc[70]["Mkt-RF_mean_21"], data.iloc[49:70]["Mkt-RF"].mean())

    def test_label_is_same_date_excess_return_not_next_row(self):
        data = fixture(100)
        data.iloc[70, 0] = 0
        x, y = make_dataset(data, "2015-01-01", "2016-01-01")
        self.assertEqual(y.loc[data.index[70]], 0)
        np.testing.assert_array_equal(y, (data.loc[x.index, "Mkt-RF"] > 0).astype(int))

    def test_split_order_and_embargo(self):
        index = fixture().index
        fold = annual_fold(index, 2022, 5)
        self.assertEqual(set(index[fold.tune].year), {2020})
        self.assertEqual(set(index[fold.calibrate].year), {2021})
        self.assertEqual(set(index[fold.test].year), {2022})
        stages = [fold.train, fold.tune, fold.calibrate, fold.test]
        for a, b in zip(stages, stages[1:]):
            self.assertGreaterEqual(b[0]-a[-1]-1, 5)
        self.assertEqual(len(set(np.concatenate(stages))), sum(map(len, stages)))

    def test_future_dates_do_not_change_prior_split(self):
        index = fixture().index
        a, b = annual_fold(index, 2021), annual_fold(index[index.year <= 2021], 2021)
        for name in ["train", "tune", "calibrate", "test"]:
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name))

    def test_insufficient_split_fails(self):
        with self.assertRaises(ValueError):
            annual_fold(fixture(100).index, 2022)


class ModelAndMetricTests(unittest.TestCase):
    def test_scaler_fits_training_only(self):
        x, y = make_dataset(fixture(200), "2015-01-01", "2016-01-01")
        model = candidates(42)["logistic_C0.01"]
        model.fit(x.iloc[:80], y.iloc[:80])
        scaler = model.named_steps["standardscaler"]
        np.testing.assert_allclose(scaler.mean_, x.iloc[:80].mean())
        saved_mean = scaler.mean_.copy()
        model.predict_proba(x.iloc[80:] * 100)
        np.testing.assert_array_equal(saved_mean, scaler.mean_)

    def test_model_bank_is_deterministic_and_no_random_early_stopping(self):
        bank = candidates(42)
        self.assertEqual(len(bank), 5)
        self.assertFalse(bank["histgb_leaves3"].early_stopping)
        self.assertEqual(clone(bank["histgb_leaves3"]).random_state, 42)

    def test_calibrator_bounded_and_requires_two_classes(self):
        p = np.linspace(.1, .9, 100)
        y = (np.arange(100) % 3 != 0).astype(int)
        calibrator = SigmoidCalibrator().fit(p, y)
        pred = calibrator.predict(np.array([0, 1, .5]))
        self.assertTrue(np.all((pred >= 0) & (pred <= 1)))
        with self.assertRaises(ValueError):
            SigmoidCalibrator().fit(p, np.zeros(100))

    def test_known_scores_and_invalid_probability(self):
        metrics = score(np.array([0, 1]), np.array([.5, .5]))
        self.assertAlmostEqual(metrics["brier"], .25)
        self.assertAlmostEqual(metrics["log_loss"], np.log(2))
        with self.assertRaises(ValueError):
            score(np.array([0, 1]), np.array([.5, 1.1]))

    def test_reliability_counts_include_extreme_probabilities(self):
        bins = reliability(np.array([0, 0, 1, 1]), np.array([0, .2, .8, 1]))
        self.assertEqual(sum(b["count"] for b in bins), 4)

    def test_block_bootstrap_paired_identical_forecasts(self):
        frame = pd.DataFrame({"year": [2020]*40+[2021]*40, "actual_up": [0, 1]*40,
                              "a": [.4, .6]*40, "b": [.4, .6]*40})
        result = paired_block_interval(frame, "a", "b", repeats=100)
        self.assertEqual(result["mean_brier_difference"], 0)
        self.assertEqual(result["ci95_low"], 0)
        self.assertEqual(result["ci95_high"], 0)

    def test_population_stability(self):
        ref = np.arange(100, dtype=float)
        self.assertEqual(population_stability(ref, ref), 0)
        self.assertGreater(population_stability(ref, ref+100), 0)


if __name__ == "__main__":
    unittest.main()
