# Research protocol

This protocol was written before the first reported experiment. Its grids, years, selection criterion and calibration rule were not changed after seeing test results.

## Question and data contract

Can lagged equity factor returns improve the probability forecast that the next observation's US market excess return is positive, compared with a constant historical positive rate?

The actual target is `1[Mkt-RF(t) > 0]`. All features for row t use rows t-1 or earlier. A return of zero is class 0. One step means the next observed trading date, not a 24-hour calendar interval. Inputs are the Kenneth French daily three-factor research archive: market excess return, SMB, HML and RF. Percent inputs are divided by 100. The code rejects duplicate/unsorted dates, missing sentinels, unexpected schemas and non-finite values. It never forward-fills returns.

**Crucial availability distinction:** date order is not publication order. This is a retrospective benchmark on a revised research vintage. The provider's files do not encode historical release timestamps. Features are lagged computationally, but may not have been published the next day. Therefore no deployable next-day trading claim follows from this study.

Training targets start in 1990; earlier rows supply rolling warm-up history. The fixed reported sample ends on 31 December 2025. Later observations in the downloaded archive are excluded from modelling and scoring.

## Annual forward design

For test year Y:

| Stage | Target dates | Purpose |
| --- | --- | --- |
| Train | 1990 through Y-3 | Fit each candidate |
| Tune | Y-2 | Choose each family's hyperparameters and the overall family by Brier score |
| Calibration | Y-1 | Fit a sigmoid transformation on recent probabilities |
| Test | Y | Evaluate frozen estimators and calibrators |

The last five observations of each pre-test stage are removed. No labels or sample membership overlap between stages. After tuning, the chosen estimator in each family is refitted on train plus tune, retaining those embargo exclusions. Scalers are fitted inside each training pipeline. Calibration data are never used for fitting a base estimator. This is an expanding window, not random cross-validation. Previous test years can enter subsequent folds as then-historical data.

Test years: **2020, 2021, 2022, 2023, 2024, 2025**. Each observation receives exactly one out-of-sample prediction. These are historical forward tests, not a prospective preregistration or a private competition holdout.

## Fixed features and candidates

31 features: market, SMB and HML lags of 1, 2, 5 and 10 observations; trailing means and sample standard deviations over 5, 21 and 63 observations for each; RF lag 1. Rolling features first shift the source by one row. There are no current-day returns, target encodings, news or future fundamentals.

- Logistic regression: standardized numeric features, C in {0.01, 0.1, 1}, maximum 2,000 solver iterations.
- Histogram gradient boosting: leaves in {3, 7}; 100 iterations, learning rate 0.05, minimum leaf observations 100, L2 penalty 10. Random internal early stopping is disabled.
- Historical prior: training positive rate for validation selection; all known train, tune and calibration positive labels for the final test forecast.
- 50% constant probability: reference benchmark, not a selectable model.

The lowest validation Brier score chooses the candidate/family. Ties break by candidate name. A selected ML model **always** receives sigmoid calibration, fitted to clipped log-odds on the calibration year with logistic regression C=1000. This policy is fixed, even if calibration later worsens test performance. Both raw and calibrated family predictions are exported. Calibration is not selected on test results. The reported `selected` series is the workflow chosen using validation information, not the best test series.

All estimators use seed 42 where relevant and one computational thread. Hyperparameter grids are intentionally small. The experiment does not search for a profitable strategy.

## Scores and uncertainty

Brier is primary. Log loss, ROC AUC, accuracy, balanced accuracy and positive-class prevalence are reported. Accuracy alone is misleading when positive returns are more frequent. Fixed probability bins show reliability and counts. Rolling 126-observation Brier scores illustrate degradation after labels become known.

Uncertainty: paired circular blocks of 20 observations, independently sampled within each annual test fold, 2,000 replicates with seed 42. Report a percentile 95% interval for selected minus prior Brier loss. This interval is conditional on the fitted models and this data vintage. It does not account for refitting, feature search, future regimes or publication delays, and is not a trading significance claim.

Feature PSI uses refit-data quantile bin edges and 0.5 histogram pseudocounts. It is a descriptive distribution-shift diagnostic, not a probability or automatic retraining trigger.

## Reproducibility and changes

Each run exports exact predictions, stage dates/row counts, candidate scores, selected candidates, feature names, dependency versions and the archive SHA-256. CSV probabilities retain 12 significant digits. Runtime and generation timestamps are intentionally not bit-stable. Future source downloads may differ because the provider revises history: identical reproduction requires the same locally retained ZIP and pinned environment. The `--expected-sha256` guard fails clearly if the vintage changes.

Any future feature, grid, split or threshold change should be documented as a new experiment rather than overwriting this result with a more favorable score. None of these results establish trading profitability.
