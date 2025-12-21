"""Purged and embargoed cross-validation for time series classification.

ADR: 2025-12-20-clickhouse-triple-barrier-backtest

Implements purged k-fold cross-validation to prevent look-ahead bias in
time series ML models. This is critical for triple-barrier classification
where labels depend on future price movements.

Key concepts:
- **Purge**: Remove training samples within `horizon` bars of test start
  to prevent label leakage (labels computed from overlapping future data)
- **Embargo**: Add additional buffer between train and test to account
  for serial correlation in features

Reference: Lopez de Prado, "Advances in Financial Machine Learning", Ch. 7
"""

from __future__ import annotations

from collections.abc import Generator
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


class PurgedKFold:
    """Purged K-Fold cross-validator for time series data.

    This cross-validator is designed for financial time series where:
    1. Data must be split chronologically (no shuffling)
    2. Labels may overlap with future observations (triple-barrier)
    3. Features may be serially correlated

    The purge removes training samples that would leak information into
    the test set. The embargo adds additional separation to account for
    feature autocorrelation.

    Attributes:
        n_splits: Number of folds
        horizon: Label horizon (bars) - samples within this distance of
                 test start are purged from training
        embargo: Additional bars to remove after purge (default: 0)
    """

    def __init__(
        self,
        n_splits: int = 5,
        horizon: int = 24,
        embargo: int = 0,
    ) -> None:
        """Initialize purged k-fold cross-validator.

        Args:
            n_splits: Number of folds (must be >= 2)
            horizon: Label horizon in bars (for purging)
            embargo: Additional embargo period in bars

        Raises:
            ValueError: If n_splits < 2 or horizon < 0
        """
        if n_splits < 2:
            msg = f"n_splits must be >= 2, got {n_splits}"
            raise ValueError(msg)
        if horizon < 0:
            msg = f"horizon must be >= 0, got {horizon}"
            raise ValueError(msg)
        if embargo < 0:
            msg = f"embargo must be >= 0, got {embargo}"
            raise ValueError(msg)

        self.n_splits = n_splits
        self.horizon = horizon
        self.embargo = embargo

    def split(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series | None = None,
        groups: np.ndarray | None = None,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Generate train/test indices with purging and embargo.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (optional, not used for splitting)
            groups: Group labels (optional, not used)

        Yields:
            Tuple of (train_indices, test_indices) for each fold
        """
        n_samples = len(X)
        fold_size = n_samples // self.n_splits

        for fold_idx in range(self.n_splits):
            # Define test set boundaries
            test_start = fold_idx * fold_size
            test_end = (fold_idx + 1) * fold_size if fold_idx < self.n_splits - 1 else n_samples

            # Define train set with purging
            # Purge: Remove samples within `horizon` bars before test_start
            # Embargo: Remove additional `embargo` bars
            purge_start = max(0, test_start - self.horizon - self.embargo)

            # Train indices: everything before purge_start, plus everything after test_end
            train_before = np.arange(0, purge_start)
            train_after = np.arange(test_end, n_samples)
            train_indices = np.concatenate([train_before, train_after])

            # Test indices
            test_indices = np.arange(test_start, test_end)

            # Skip empty folds
            if len(train_indices) == 0 or len(test_indices) == 0:
                continue

            yield train_indices, test_indices

    def get_n_splits(
        self,
        X: np.ndarray | pd.DataFrame | None = None,
        y: np.ndarray | pd.Series | None = None,
        groups: np.ndarray | None = None,
    ) -> int:
        """Return the number of splits.

        Args:
            X: Ignored
            y: Ignored
            groups: Ignored

        Returns:
            Number of folds
        """
        return self.n_splits


class PurgedTimeSeriesSplit:
    """Purged time series split with expanding training window.

    Similar to sklearn's TimeSeriesSplit but with purging and embargo.
    Training set expands with each fold while test set moves forward.

    This is often preferred over k-fold for walk-forward validation
    as it better mimics real-world model deployment.
    """

    def __init__(
        self,
        n_splits: int = 5,
        horizon: int = 24,
        embargo: int = 0,
        test_size: int | None = None,
        gap: int = 0,
    ) -> None:
        """Initialize purged time series split.

        Args:
            n_splits: Number of splits
            horizon: Label horizon for purging
            embargo: Additional embargo period
            test_size: Fixed test set size (if None, auto-computed)
            gap: Gap between train and test (in addition to purge+embargo)
        """
        if n_splits < 2:
            msg = f"n_splits must be >= 2, got {n_splits}"
            raise ValueError(msg)

        self.n_splits = n_splits
        self.horizon = horizon
        self.embargo = embargo
        self.test_size = test_size
        self.gap = gap

    def split(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series | None = None,
        groups: np.ndarray | None = None,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Generate train/test indices with expanding window.

        Args:
            X: Feature matrix
            y: Target labels (optional)
            groups: Group labels (optional)

        Yields:
            Tuple of (train_indices, test_indices) for each fold
        """
        n_samples = len(X)

        # Compute test size if not specified
        test_size = self.test_size
        if test_size is None:
            test_size = n_samples // (self.n_splits + 1)

        # Total gap = purge horizon + embargo + explicit gap
        total_gap = self.horizon + self.embargo + self.gap

        for fold_idx in range(self.n_splits):
            # Test set moves forward with each fold
            test_end = n_samples - (self.n_splits - fold_idx - 1) * test_size
            test_start = test_end - test_size

            # Train set: everything before test_start minus gap
            train_end = max(0, test_start - total_gap)
            train_indices = np.arange(0, train_end)

            # Test indices
            test_indices = np.arange(test_start, test_end)

            # Skip if insufficient data
            if len(train_indices) < test_size or len(test_indices) == 0:
                continue

            yield train_indices, test_indices

    def get_n_splits(
        self,
        X: np.ndarray | pd.DataFrame | None = None,
        y: np.ndarray | pd.Series | None = None,
        groups: np.ndarray | None = None,
    ) -> int:
        """Return the number of splits."""
        return self.n_splits


def purged_train_test_split(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    test_size: float = 0.2,
    horizon: int = 24,
    embargo: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Single train/test split with purging.

    Convenience function for simple train/test split with purge and embargo.

    Args:
        X: Feature matrix
        y: Target labels
        test_size: Fraction of data for test set (0 < test_size < 1)
        horizon: Label horizon for purging
        embargo: Additional embargo period

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    n_train_max = n_samples - n_test

    # Apply purge and embargo
    total_gap = horizon + embargo
    n_train = max(0, n_train_max - total_gap)

    # Convert to numpy if needed
    X_arr = X.values if hasattr(X, "values") else np.asarray(X)
    y_arr = y.values if hasattr(y, "values") else np.asarray(y)

    X_train = X_arr[:n_train]
    y_train = y_arr[:n_train]
    X_test = X_arr[n_train_max:]
    y_test = y_arr[n_train_max:]

    return X_train, X_test, y_train, y_test


def compute_purge_statistics(
    n_samples: int,
    n_splits: int,
    horizon: int,
    embargo: int = 0,
) -> dict[str, int | float]:
    """Compute statistics about purged cross-validation.

    Useful for understanding how much data is lost to purging.

    Args:
        n_samples: Total number of samples
        n_splits: Number of CV folds
        horizon: Label horizon
        embargo: Embargo period

    Returns:
        Dictionary with purge statistics
    """
    fold_size = n_samples // n_splits
    total_purge = horizon + embargo

    # Average training size per fold (after purging)
    avg_train_size = n_samples - fold_size - total_purge

    # Percentage of data purged per fold
    pct_purged = total_purge / n_samples * 100

    return {
        "n_samples": n_samples,
        "n_splits": n_splits,
        "fold_size": fold_size,
        "purge_size": total_purge,
        "avg_train_size": max(0, avg_train_size),
        "pct_purged_per_fold": pct_purged,
        "effective_train_ratio": max(0, avg_train_size) / n_samples,
    }
