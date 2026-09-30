"""Generate figures and a readable research result from actual predictions."""

from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABELS = {"coin_flip": "50% probability", "historical_prior": "Historical prior",
          "logistic_raw": "Logistic raw", "logistic_sigmoid": "Logistic calibrated",
          "histgb_raw": "Boosting raw", "histgb_sigmoid": "Boosting calibrated",
          "selected": "Validation-selected"}
COLORS = {"coin_flip": "#a3aab4", "historical_prior": "#64748b", "logistic_raw": "#60a5fa",
          "logistic_sigmoid": "#2563eb", "histgb_raw": "#fbbf24", "histgb_sigmoid": "#d97706", "selected": "#0d9488"}


def save_figure(fig, output: Path, name: str) -> None:
    fig.savefig(output/f"{name}.svg", bbox_inches="tight", metadata={"Date": None})
    fig.savefig(output/f"{name}.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


def write_report(report: dict, predictions: pd.DataFrame, output: Path) -> None:
    figures = output/"figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "svg.hashsalt": "equity-benchmark-v1"})
    scores = report["aggregate_metrics"]
    names = list(scores)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    bars = ax.barh([LABELS[n] for n in names], [scores[n]["brier"] for n in names],
                   color=[COLORS[n] for n in names])
    ax.bar_label(bars, fmt="%.4f", padding=5)
    ax.set_xlim(0, max(scores[n]["brier"] for n in names)*1.18)
    ax.set_xlabel("Brier score (lower is better; zero-based axis)")
    ax.set_title("US equity market direction | historical forward tests")
    fig.tight_layout(); save_figure(fig, figures, "benchmark")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for name in ["historical_prior", "logistic_sigmoid", "histgb_sigmoid", "selected"]:
        bins = report["reliability"][name]
        axes[0].plot([b["mean_probability"] for b in bins], [b["observed_frequency"] for b in bins],
                     "o-", color=COLORS[name], label=LABELS[name])
        axes[1].hist(predictions[name], bins=np.linspace(0, 1, 21), alpha=.45,
                     color=COLORS[name], label=LABELS[name])
    axes[0].plot([0, 1], [0, 1], "--", color="#94a3b8")
    axes[0].set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability", ylabel="Observed positive rate",
                title="Reliability in fixed probability bins")
    axes[1].set(xlim=(0, 1), xlabel="Predicted probability", ylabel="Observations", title="Probability distribution")
    axes[1].legend(fontsize=8)
    fig.tight_layout(); save_figure(fig, figures, "calibration")

    fig, ax = plt.subplots(figsize=(10, 4.2))
    indexed = predictions.set_index(pd.to_datetime(predictions.date))
    for name in ["historical_prior", "logistic_sigmoid", "histgb_sigmoid", "selected"]:
        loss = (indexed[name]-indexed.actual_up)**2
        ax.plot(indexed.index, loss.rolling(126, min_periods=126).mean(), label=LABELS[name], color=COLORS[name], lw=1.3)
    ax.set(ylabel="126-observation trailing Brier score", title="Observed-label monitoring across regimes")
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout(); save_figure(fig, figures, "rolling_loss")

    difference = report["paired_uncertainty"]
    if difference["ci95_high"] < 0:
        finding = "The preselected workflow has a lower Brier score than the historical prior in this retrospective sample; the paired block interval is below zero. This does not establish a tradable edge."
    elif difference["ci95_low"] > 0:
        finding = "The preselected workflow has a higher Brier score than the historical prior in this retrospective sample; the paired block interval is above zero. The more complex workflow does not improve this benchmark."
    else:
        finding = "The paired block interval includes zero. This experiment does not establish a reliable Brier-score improvement over the historical-prior baseline."
    lines = ["# Executed research results", "", finding, "",
             f"Generated from {len(predictions):,} real-data target observations across {len(report['folds'])} annual forward tests.", "",
             "## Aggregate probability scores", "",
             "| Model | Brier ↓ | Log loss ↓ | Accuracy | Balanced accuracy | ROC AUC |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name in names:
        m = scores[name]
        lines.append(f"| {LABELS[name]} | {m['brier']:.6f} | {m['log_loss']:.6f} | {m['accuracy']:.2%} | {m['balanced_accuracy']:.2%} | {m['roc_auc']:.4f} |")
    lines += ["", "The historical prior predicts the positive fraction in the known training, tuning and calibration labels. A high directional accuracy can reflect this positive-class imbalance; Brier score is the primary selection metric.",
              "", "![Probability benchmark](figures/benchmark.svg)", "", "## Annual decisions and results", "",
              "| Test year | Training rows | Tuning choice | Issued output | Selected Brier | Prior Brier |",
              "| --- | ---: | --- | --- | ---: | ---: |"]
    for f in report["folds"]:
        lines.append(f"| {f['test_year']} | {f['windows']['train']['rows']} | {f['selected_candidate']} | {f['selected_output']} | {f['test_metrics']['selected']['brier']:.6f} | {f['test_metrics']['historical_prior']['brier']:.6f} |")
    lines += ["", "Hyperparameters and model family are chosen on the tuning year. Sigmoid calibration is always applied to a selected ML family; its inclusion is fixed in advance, not selected on test results. Both raw and calibrated scores are shown for audit. Prior years' test observations may become later years' training data once in the past.",
              "", "## Uncertainty", "",
              f"Selected minus prior mean Brier difference: **{difference['mean_brier_difference']:+.6f}**. Paired 95% block-bootstrap interval: **[{difference['ci95_low']:+.6f}, {difference['ci95_high']:+.6f}]**.",
              "", f"{difference['replicates']} paired circular resamples, blocks of {difference['block_observations']} observations, resampling within each test year. The interval is conditional on fitted models, the fixed protocol and the source vintage; it does not include model refitting uncertainty, historical revisions or publication latency.",
              "", "## Calibration and monitoring", "", "![Reliability and probability distribution](figures/calibration.svg)", "",
              "Bin counts and exact values are in metrics.json; sparse reliability bins can be noisy. Calibration can worsen results under regime changes.",
              "", "![Trailing Brier monitoring](figures/rolling_loss.svg)", "",
              "Rolling scores use already observed outcomes and are descriptive. Feature PSI values are computed using reference-only quantile bins and stored per fold; they are not causal explanations or an automatic retraining policy.",
              "", "## Provenance and scope", "",
              f"Source archive SHA-256: `{report['data']['archive_sha256']}`.",
              "", "The provider revises historical factors and does not provide point-in-time daily publication timestamps in this archive. The split prevents computational look-ahead within this downloaded vintage, but not vintage/revision or publication-delay bias. Daily factors are research portfolio returns, not an executable instrument. No P&L, fills, transaction costs, live monitoring or production trading performance are claimed.",
              "", "Files: [metrics.json](metrics.json), [predictions.csv](predictions.csv), [data_manifest.json](data_manifest.json).", ""]
    (output/"RESULTS.md").write_text("\n".join(lines))
