"""Fetch, hash and validate the provider's daily research factor archive."""

from __future__ import annotations

import hashlib
import io
import json
import re
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
SOURCE_PAGE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html"
COLUMNS = ["Mkt-RF", "SMB", "HML", "RF"]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_archive(payload: bytes) -> tuple[pd.DataFrame, str]:
    """Read just YYYYMMDD rows; reject missing data rather than invent returns."""
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        if len(names) != 1:
            raise ValueError("Expected exactly one CSV in factor ZIP")
        text = archive.read(names[0]).decode("utf-8-sig")
    lines = text.splitlines()
    headers = [line.strip() for line in lines if line.strip().startswith(",Mkt-RF")]
    if len(headers) != 1 or headers[0].replace(" ", "") != ",Mkt-RF,SMB,HML,RF":
        raise ValueError("Unexpected provider schema")
    rows = [line.strip() for line in lines if re.match(r"^\s*\d{8},", line)]
    if not rows:
        raise ValueError("No daily observations found")
    frame = pd.read_csv(io.StringIO("date," + ",".join(COLUMNS) + "\n" + "\n".join(rows)))
    dates = pd.to_datetime(frame.pop("date").astype(str), format="%Y%m%d", errors="raise")
    frame.index = pd.DatetimeIndex(dates, name="date")
    frame = frame.apply(pd.to_numeric, errors="raise").astype(float)
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("Provider dates must be strictly increasing and unique")
    if not np.isfinite(frame.to_numpy()).all() or frame.isin([-99.99, -999.0]).any().any():
        raise ValueError("Missing or non-finite provider returns detected")
    if (frame.abs() > 100).any().any():
        raise ValueError("Implausible percent return; check units/schema")
    # Provider units are percent; all internal calculations use decimal returns.
    frame /= 100.0
    return frame, "\n".join(lines[:4]).strip()


def load_data(path: Path, expected_sha256: str | None = None) -> tuple[pd.DataFrame, dict]:
    digest = sha256_file(path)
    if expected_sha256 and digest != expected_sha256:
        raise ValueError(f"Snapshot SHA-256 mismatch: expected {expected_sha256}, got {digest}")
    frame, header = parse_archive(path.read_bytes())
    return frame, {
        "source_url": SOURCE_URL, "source_page": SOURCE_PAGE,
        "archive_sha256": digest, "source_header": header,
        "source_rows": len(frame), "source_start": str(frame.index.min().date()),
        "source_end": str(frame.index.max().date()), "units": "decimal returns",
        "availability": "Revised research data, not point-in-time publication timestamps",
    }


def download(path: Path, overwrite: bool = False) -> dict:
    if path.exists() and not overwrite:
        _, metadata = load_data(path)
        return metadata
    request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "equity-direction-benchmark/0.1 research"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read(5_000_001)
    if len(payload) > 5_000_000:
        raise ValueError("Unexpected archive size")
    parse_archive(payload)  # Validate before atomically replacing a local cache.
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
    _, metadata = load_data(path)
    metadata["retrieved_at_utc"] = datetime.now(timezone.utc).isoformat()
    path.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
