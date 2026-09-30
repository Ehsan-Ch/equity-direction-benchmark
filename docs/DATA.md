# Data provenance and permitted interpretation

The experiment downloads **Fama/French 3 Factors [Daily]** from the [Kenneth R. French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html). See the [provider's factor definitions](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/f-f_factors.html).

`Mkt-RF` is the US equity market return in excess of the risk-free rate. SMB and HML are research portfolio factors related to size and value. The target is the sign of market excess return, not the sign of a specific stock, index price, or total market return. Raw percent fields are converted to decimal returns.

The source carries copyright attribution to Eugene F. Fama and Kenneth R. French. The MIT license in this repository applies to project code, not the underlying data. The source archive and parsed factor history are not redistributed here. Download directly from the provider and comply with its applicable terms. Committed predictions contain derived binary outcomes and model probabilities, not the full factor-return dataset.

## Snapshot used for published results

- Provider header: **202608 CRSP database**.
- Source coverage: 1926-07-01 through 2026-08-31.
- Model target range: 1990-01-01 through 2025-12-31.
- Annual test targets: 2020-2025.
- SHA-256 and exact header: [data_manifest.json](../reports/data_manifest.json).
- Experiment environment and execution time: [metrics.json](../reports/metrics.json).

This archive changes as the provider updates and revises history. A fixed end date alone does not freeze a vintage. Preserve the downloaded ZIP locally and use the hash guard to reproduce the published results. The repository intentionally cannot guarantee that a future fresh download will reproduce identical scores.

## Three separate timing questions

1. **Feature computation:** each feature at t uses rows strictly before t; tests mutate present/future rows to verify this.
2. **Model fitting:** training, tuning, calibration and test target dates are ordered and separated; no current test labels select the model.
3. **Real availability:** the data archive is revised and has no point-in-time publication schedule. The first two controls do not solve this third issue.

In January 2025 the provider switched its US research-return production from CRSP's legacy FIZ files to CIZ files; its source page explains the change. Risk-free-rate sourcing also changes beginning June 2024. These are source-methodology considerations, not evidence of model improvement or deterioration by themselves.

## Before a live experiment

A genuine deployment would require a licensed, timestamped feed; point-in-time data storage; an exchange calendar; decisions about latency and revised records; monitoring for schema and feed failures; and prospective paper evaluation. A tradable instrument, execution rules, transaction costs and risk controls would need a separate study. This repository sends no orders and connects to no broker.
