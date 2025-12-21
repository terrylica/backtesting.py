"""Hyperparameter optimization for triple-barrier classification.

ADR: 2025-12-20-clickhouse-triple-barrier-backtest

Implements grid search over barrier parameters (b, H) using purged
cross-validation. Optimization target is log-likelihood (cross-entropy)
to ensure calibrated probability estimates.

The optimization finds the best (barrier_pct, horizon_bars) configuration
that maximizes out-of-sample probability calibration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

if TYPE_CHECKING:
    from user_strategies.configs.barrier_config import BarrierConfig
    from user_strategies.strategies.classifier import TripleBarrierClassifier

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of hyperparameter optimization.

    Attributes:
        best_barrier_pct: Optimal barrier percentage
        best_horizon_bars: Optimal horizon in bars
        best_score: Best cross-validation score (negative log-loss)
        cv_results: DataFrame with all evaluated configurations
        best_config: BarrierConfig with optimal parameters
    """

    best_barrier_pct: float
    best_horizon_bars: int
    best_score: float
    cv_results: pd.DataFrame
    best_config: BarrierConfig | None = None


def optimize_barrier_params(
    prices: pd.Series,
    features: pd.DataFrame,
    barrier_pct_grid: list[float] | tuple[float, ...],
    horizon_bars_grid: list[int] | tuple[int, ...],
    classifier: TripleBarrierClassifier | None = None,
    n_splits: int = 5,
    embargo: int = 0,
    scoring: str = "neg_log_loss",
    verbose: bool = True,
) -> OptimizationResult:
    """Optimize barrier parameters using purged cross-validation.

    Performs grid search over (barrier_pct, horizon_bars) combinations,
    using purged k-fold CV to prevent look-ahead bias.

    Args:
        prices: Price series for label generation
        features: Feature DataFrame aligned with prices
        barrier_pct_grid: List of barrier percentages to try
        horizon_bars_grid: List of horizon values to try
        classifier: Classifier to use (default: logistic regression)
        n_splits: Number of CV folds
        embargo: Additional embargo period for CV
        scoring: Scoring metric ('neg_log_loss' or 'accuracy')
        verbose: Print progress

    Returns:
        OptimizationResult with best parameters and full results

    Example:
        >>> result = optimize_barrier_params(
        ...     prices=df['Close'],
        ...     features=df[feature_cols],
        ...     barrier_pct_grid=[0.005, 0.01, 0.015],
        ...     horizon_bars_grid=[12, 24, 48],
        ... )
        >>> print(f"Best: b={result.best_barrier_pct}, H={result.best_horizon_bars}")
    """
    from user_strategies.configs.barrier_config import BarrierConfig
    from user_strategies.strategies.classifier import TripleBarrierClassifier
    from user_strategies.strategies.purged_cv import PurgedKFold
    from user_strategies.strategies.triple_barrier import compute_triple_barrier_labels

    # Create default classifier if not provided
    if classifier is None:
        classifier = TripleBarrierClassifier(model_type="logistic", calibrated=True)

    results = []
    total_configs = len(barrier_pct_grid) * len(horizon_bars_grid)
    config_idx = 0

    for barrier_pct in barrier_pct_grid:
        for horizon_bars in horizon_bars_grid:
            config_idx += 1

            if verbose:
                logger.info(
                    "Evaluating config %d/%d: barrier_pct=%.3f, horizon_bars=%d",
                    config_idx,
                    total_configs,
                    barrier_pct,
                    horizon_bars,
                )

            # Generate labels for this configuration
            labels = compute_triple_barrier_labels(
                prices=prices,
                barrier_pct=barrier_pct,
                horizon_bars=horizon_bars,
            )

            # Align features and labels
            valid_mask = ~labels.isna()
            X = features.loc[valid_mask].values
            y = labels.loc[valid_mask].values

            if len(y) < n_splits * 10:
                logger.warning(
                    "Insufficient samples (%d) for config b=%.3f, H=%d",
                    len(y),
                    barrier_pct,
                    horizon_bars,
                )
                continue

            # Perform purged cross-validation
            cv = PurgedKFold(
                n_splits=n_splits,
                horizon=horizon_bars,
                embargo=embargo,
            )

            fold_scores = []
            for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # Skip if any class missing in train
                if len(np.unique(y_train)) < 3:
                    continue

                # Clone and fit classifier
                clf = _clone_classifier(classifier)
                clf.fit(X_train, y_train)

                # Score on test set
                if scoring == "neg_log_loss":
                    proba = clf.predict_proba(X_test)
                    score = -log_loss(y_test, proba, labels=clf.classes_)
                else:  # accuracy
                    score = clf.score(X_test, y_test)

                fold_scores.append(score)

            if len(fold_scores) == 0:
                continue

            mean_score = np.mean(fold_scores)
            std_score = np.std(fold_scores)

            results.append({
                "barrier_pct": barrier_pct,
                "horizon_bars": horizon_bars,
                "mean_score": mean_score,
                "std_score": std_score,
                "n_folds": len(fold_scores),
                "n_samples": len(y),
            })

            if verbose:
                logger.info(
                    "  Score: %.4f (+/- %.4f) from %d folds",
                    mean_score,
                    std_score,
                    len(fold_scores),
                )

    # Convert to DataFrame and find best
    cv_results = pd.DataFrame(results)

    if len(cv_results) == 0:
        msg = "No valid configurations found"
        raise ValueError(msg)

    best_idx = cv_results["mean_score"].idxmax()
    best_row = cv_results.loc[best_idx]

    best_config = BarrierConfig(
        barrier_pct=best_row["barrier_pct"],
        horizon_bars=int(best_row["horizon_bars"]),
    )

    return OptimizationResult(
        best_barrier_pct=best_row["barrier_pct"],
        best_horizon_bars=int(best_row["horizon_bars"]),
        best_score=best_row["mean_score"],
        cv_results=cv_results.sort_values("mean_score", ascending=False),
        best_config=best_config,
    )


def optimize_classifier_params(
    X: np.ndarray,
    y: np.ndarray,
    horizon: int,
    param_grid: dict[str, list[Any]],
    model_type: str = "logistic",
    n_splits: int = 5,
    embargo: int = 0,
) -> dict[str, Any]:
    """Optimize classifier hyperparameters with purged CV.

    Args:
        X: Feature matrix
        y: Labels
        horizon: Label horizon for purging
        param_grid: Dictionary of parameter grids
        model_type: Classifier type
        n_splits: Number of CV folds
        embargo: Embargo period

    Returns:
        Dictionary with best parameters and scores
    """
    from sklearn.model_selection import ParameterGrid

    from user_strategies.strategies.classifier import TripleBarrierClassifier
    from user_strategies.strategies.purged_cv import PurgedKFold

    cv = PurgedKFold(n_splits=n_splits, horizon=horizon, embargo=embargo)

    best_score = -np.inf
    best_params = None
    results = []

    for params in ParameterGrid(param_grid):
        clf = TripleBarrierClassifier(model_type=model_type, **params)

        fold_scores = []
        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            if len(np.unique(y_train)) < 3:
                continue

            clf.fit(X_train, y_train)
            proba = clf.predict_proba(X_test)
            score = -log_loss(y_test, proba, labels=clf.classes_)
            fold_scores.append(score)

        if len(fold_scores) == 0:
            continue

        mean_score = np.mean(fold_scores)
        results.append({"params": params, "mean_score": mean_score})

        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    return {
        "best_params": best_params,
        "best_score": best_score,
        "all_results": results,
    }


def _clone_classifier(classifier: TripleBarrierClassifier) -> TripleBarrierClassifier:
    """Create a fresh copy of a classifier with same parameters.

    Args:
        classifier: Classifier to clone

    Returns:
        New classifier instance with same configuration
    """
    from user_strategies.strategies.classifier import TripleBarrierClassifier

    return TripleBarrierClassifier(
        model_type=classifier.model_type,
        calibrated=classifier.calibrated,
        random_state=classifier.random_state,
        **classifier.model_kwargs,
    )
