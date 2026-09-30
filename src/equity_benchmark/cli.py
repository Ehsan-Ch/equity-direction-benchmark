"""Small CLI for downloading, running research and replaying saved predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .data import download, load_data
from .experiment import Config, run
from .features import make_features


def replay(archive: Path, artifact: Path, date: str) -> dict:
    # joblib uses pickle: this CLI is for artifacts generated locally by `run` only.
    bundle = joblib.load(artifact)
    frame, metadata = load_data(archive, bundle["source_sha256"])
    target = pd.Timestamp(date)
    if target < pd.Timestamp(bundle["earliest_target_date"]):
        raise ValueError("Replay date precedes this model's forward-test window")
    if target not in frame.index:
        raise ValueError("Replay target must be an observed provider date")
    x = make_features(frame).loc[[target], bundle["feature_names"]]
    if not np.isfinite(x.to_numpy()).all():
        raise ValueError("Insufficient history for requested date")
    family = bundle["selected_family"]
    if family == "prior":
        p = bundle["historical_prior"]
    else:
        saved = bundle["estimators"][family]
        with threadpool_limits(limits=1):
            raw = saved["model"].predict_proba(x)[:, 1]
            p = float(saved["calibrator"].predict(raw)[0])
    return {"target_date": date, "features_through": str(frame.index[frame.index.get_loc(target)-1].date()),
            "model_fit_data_through": bundle["fit_data_through"], "positive_excess_return_probability": p,
            "selected_family": family, "data_sha256": metadata["archive_sha256"],
            "mode": "Historical replay on revised research data; not a live market forecast"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("download", help="Cache the provider archive; no raw data are committed")
    fetch.add_argument("--data", type=Path, default=Path("data/raw/factors.zip"))
    fetch.add_argument("--refresh", action="store_true")
    research = commands.add_parser("run", help="Execute the predeclared annual forward tests")
    research.add_argument("--data", type=Path, default=Path("data/raw/factors.zip"))
    research.add_argument("--output", type=Path, default=Path("reports"))
    research.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    research.add_argument("--expected-sha256")
    research.add_argument("--first-test-year", type=int, default=2020)
    research.add_argument("--last-test-year", type=int, default=2025)
    research.add_argument("--end", default="2025-12-31")
    predict = commands.add_parser("replay", help="Replay one historical target with the locally generated final model")
    predict.add_argument("--data", type=Path, default=Path("data/raw/factors.zip"))
    predict.add_argument("--artifact", type=Path, default=Path("artifacts/final_fold.joblib"))
    predict.add_argument("--date", required=True)
    args = parser.parse_args()
    if args.command == "download":
        result = download(args.data, args.refresh)
    elif args.command == "run":
        result = run(args.data, args.output, args.artifacts,
                     Config(end=args.end, first_test_year=args.first_test_year, last_test_year=args.last_test_year),
                     args.expected_sha256)
        result = {"aggregate_metrics": result["aggregate_metrics"], "paired_uncertainty": result["paired_uncertainty"]}
    else:
        result = replay(args.data, args.artifact, args.date)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
