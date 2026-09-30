# Understanding and extending the project

This project was created with OpenAI Codex assistance for Ehsan Cheraghi's research portfolio. The code was executed and tested on real historical data. Before presenting it in an interview, run it yourself and understand the decisions below. Describe your own subsequent work accurately.

## Five questions to explain

**What is being predicted?** Whether the next observed US market excess return is positive. It is a binary research target, not the future price of an individual stock.

**What does the baseline do?** It estimates the positive fraction of previously available labels. Because positive market returns are more frequent, always predicting up can have accuracy above 50% without useful discrimination. A probability score and balanced accuracy reveal this.

**Where can leakage happen?** In an unshifted rolling feature, a scaler fitted before splitting, calibration using training predictions, hyperparameter selection on test results, or assuming revised data were available in real time. The code addresses the first four. The dataset cannot resolve the fifth.

**Why did calibration sometimes hurt?** It is fitted on a recent year with relatively few observations. Relationships can change in the next year, and an estimated sigmoid can add variance. A calibrated model is not automatically better; raw and transformed predictions are reported side by side.

**What did the run establish?** The validation-selected workflow has slightly lower aggregate Brier loss than the prior, but the paired uncertainty interval crosses zero. This result does not establish a reliable improvement, and certainly not trading profitability.

## Read the code in this order

1. `data.py`: schema, units, missing values and snapshot hash.
2. `features.py`: reproduce one lag and one rolling mean by hand.
3. `splits.py`: draw the four windows for test year 2025.
4. `experiment.py`: follow selection, refitting and calibration before test scoring.
5. `metrics.py`: compute Brier as the mean squared probability error.
6. `tests/`: identify which failure each test catches.

## Useful next experiments

- Obtain point-in-time instrument data before testing executable signals.
- Compare this fixed calibration policy with a nested, time-ordered calibration-policy selection rule in a new documented experiment.
- Study coefficient/feature stability using only each fold's permitted training data.
- Add a recent rolling training-window challenger, chosen without inspecting a new final holdout.
- Once a genuinely fresh period has accumulated, evaluate the frozen protocol prospectively.

Keep the original results. Record why a change was made, which data informed it, and which period remains untouched. Do not tune until an attractive historical score appears.
