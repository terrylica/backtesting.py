"""Probability calibration diagnostics for triple-barrier classification.

ADR: 2025-12-20-clickhouse-triple-barrier-backtest

Implements calibration metrics and visualization for probabilistic classifiers:
- Reliability diagrams (predicted vs actual probability)
- Brier score and decomposition
- Expected Calibration Error (ECE)

These diagnostics ensure that predicted probabilities are meaningful:
when we predict P(upper) = 0.7, the upper barrier should hit ~70% of the time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CalibrationMetrics:
    """Calibration metrics for a probabilistic classifier.

    Attributes:
        brier_score: Mean squared error of probability predictions
        ece: Expected Calibration Error
        mce: Maximum Calibration Error
        reliability_data: DataFrame with binned calibration data
        class_brier_scores: Brier score per class
    """

    brier_score: float
    ece: float
    mce: float
    reliability_data: pd.DataFrame
    class_brier_scores: dict[int, float]


def compute_brier_score(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    classes: np.ndarray | None = None,
) -> float:
    """Compute multi-class Brier score.

    The Brier score is a proper scoring rule that measures the mean squared
    difference between predicted probabilities and actual outcomes.

    Args:
        y_true: True class labels
        y_proba: Predicted probabilities (n_samples, n_classes)
        classes: Class labels in order matching y_proba columns

    Returns:
        Brier score (lower is better, 0 = perfect)
    """
    if classes is None:
        classes = np.array([-1, 0, 1])

    n_samples = len(y_true)
    n_classes = len(classes)

    # Convert y_true to one-hot
    y_true_onehot = np.zeros((n_samples, n_classes))
    for i, c in enumerate(classes):
        y_true_onehot[:, i] = (y_true == c).astype(float)

    # Compute mean squared error
    return np.mean(np.sum((y_proba - y_true_onehot) ** 2, axis=1))


def compute_brier_score_per_class(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    classes: np.ndarray | None = None,
) -> dict[int, float]:
    """Compute Brier score for each class separately.

    Args:
        y_true: True class labels
        y_proba: Predicted probabilities
        classes: Class labels

    Returns:
        Dictionary mapping class label to its Brier score
    """
    if classes is None:
        classes = np.array([-1, 0, 1])

    scores = {}
    for i, c in enumerate(classes):
        y_binary = (y_true == c).astype(float)
        p_class = y_proba[:, i]
        scores[int(c)] = np.mean((p_class - y_binary) ** 2)

    return scores


def compute_ece(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
    class_idx: int = 2,
) -> float:
    """Compute Expected Calibration Error.

    ECE measures the average gap between predicted probability and
    observed frequency, weighted by the number of samples in each bin.

    Args:
        y_true: True class labels
        y_proba: Predicted probabilities
        n_bins: Number of calibration bins
        class_idx: Which class to compute ECE for (default: 2 = upper barrier)

    Returns:
        ECE score (lower is better, 0 = perfectly calibrated)
    """
    # Get probabilities for target class
    probs = y_proba[:, class_idx]

    # Map class_idx back to actual label
    classes = np.array([-1, 0, 1])
    target_label = classes[class_idx]
    y_binary = (y_true == target_label).astype(float)

    # Bin probabilities
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(probs, bin_edges[1:-1])

    ece = 0.0
    n_total = len(probs)

    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        n_bin = mask.sum()

        if n_bin == 0:
            continue

        # Average predicted probability in bin
        avg_prob = probs[mask].mean()
        # Observed frequency in bin
        obs_freq = y_binary[mask].mean()

        # Weighted absolute difference
        ece += (n_bin / n_total) * np.abs(avg_prob - obs_freq)

    return ece


def compute_mce(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
    class_idx: int = 2,
) -> float:
    """Compute Maximum Calibration Error.

    MCE is the maximum gap between predicted and observed probability
    across all bins.

    Args:
        y_true: True class labels
        y_proba: Predicted probabilities
        n_bins: Number of calibration bins
        class_idx: Which class to compute MCE for

    Returns:
        MCE score (lower is better)
    """
    probs = y_proba[:, class_idx]
    classes = np.array([-1, 0, 1])
    target_label = classes[class_idx]
    y_binary = (y_true == target_label).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(probs, bin_edges[1:-1])

    max_error = 0.0

    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        if mask.sum() == 0:
            continue

        avg_prob = probs[mask].mean()
        obs_freq = y_binary[mask].mean()
        max_error = max(max_error, np.abs(avg_prob - obs_freq))

    return max_error


def compute_reliability_diagram_data(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
    class_idx: int = 2,
) -> pd.DataFrame:
    """Compute data for reliability diagram visualization.

    Args:
        y_true: True class labels
        y_proba: Predicted probabilities
        n_bins: Number of bins
        class_idx: Which class to analyze

    Returns:
        DataFrame with columns:
            - bin_center: Center of probability bin
            - predicted_prob: Average predicted probability in bin
            - observed_freq: Observed frequency of positive class
            - count: Number of samples in bin
            - gap: Difference between predicted and observed
    """
    probs = y_proba[:, class_idx]
    classes = np.array([-1, 0, 1])
    target_label = classes[class_idx]
    y_binary = (y_true == target_label).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_indices = np.digitize(probs, bin_edges[1:-1])

    data = []
    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        n_bin = mask.sum()

        if n_bin == 0:
            data.append({
                "bin_center": bin_centers[bin_idx],
                "predicted_prob": np.nan,
                "observed_freq": np.nan,
                "count": 0,
                "gap": np.nan,
            })
        else:
            avg_prob = probs[mask].mean()
            obs_freq = y_binary[mask].mean()
            data.append({
                "bin_center": bin_centers[bin_idx],
                "predicted_prob": avg_prob,
                "observed_freq": obs_freq,
                "count": n_bin,
                "gap": avg_prob - obs_freq,
            })

    return pd.DataFrame(data)


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> CalibrationMetrics:
    """Compute comprehensive calibration metrics.

    Args:
        y_true: True class labels
        y_proba: Predicted probabilities
        n_bins: Number of bins for ECE/reliability

    Returns:
        CalibrationMetrics with all diagnostic information
    """
    brier = compute_brier_score(y_true, y_proba)
    class_brier = compute_brier_score_per_class(y_true, y_proba)

    # Compute ECE for upper barrier class (most important for trading)
    ece = compute_ece(y_true, y_proba, n_bins=n_bins, class_idx=2)
    mce = compute_mce(y_true, y_proba, n_bins=n_bins, class_idx=2)

    reliability = compute_reliability_diagram_data(
        y_true, y_proba, n_bins=n_bins, class_idx=2
    )

    return CalibrationMetrics(
        brier_score=brier,
        ece=ece,
        mce=mce,
        reliability_data=reliability,
        class_brier_scores=class_brier,
    )


def print_calibration_report(metrics: CalibrationMetrics) -> None:
    """Print a formatted calibration report.

    Args:
        metrics: CalibrationMetrics from compute_calibration_metrics()
    """
    print("=" * 50)
    print("CALIBRATION REPORT")
    print("=" * 50)
    print(f"\nBrier Score (overall): {metrics.brier_score:.4f}")
    print("Brier Score (per class):")
    for label, score in metrics.class_brier_scores.items():
        label_name = {-1: "Lower", 0: "Timeout", 1: "Upper"}[label]
        print(f"  {label_name} (Y={label:+d}): {score:.4f}")

    print(f"\nExpected Calibration Error (ECE): {metrics.ece:.4f}")
    print(f"Maximum Calibration Error (MCE): {metrics.mce:.4f}")

    print("\nReliability Diagram Data (Upper Barrier):")
    rel_data = metrics.reliability_data.dropna()
    if len(rel_data) > 0:
        print(rel_data.to_string(index=False))
    else:
        print("  No data available")

    print("\n" + "=" * 50)


def is_well_calibrated(
    metrics: CalibrationMetrics,
    ece_threshold: float = 0.1,
    brier_threshold: float = 0.25,
) -> bool:
    """Check if classifier meets calibration quality thresholds.

    Args:
        metrics: Calibration metrics
        ece_threshold: Maximum acceptable ECE
        brier_threshold: Maximum acceptable Brier score

    Returns:
        True if classifier is well-calibrated
    """
    return metrics.ece <= ece_threshold and metrics.brier_score <= brier_threshold
