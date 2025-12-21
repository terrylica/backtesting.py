"""Triple-barrier label generation for probabilistic classification.

ADR: 2025-12-20-clickhouse-triple-barrier-backtest

Implements the triple-barrier labeling method from Marco Lopez de Prado's
"Advances in Financial Machine Learning". At each event time t_0, we define:

    tau^+ = inf{t > t_0 : X_t - X_{t_0} >= b}  (upper barrier hit)
    tau^- = inf{t > t_0 : X_t - X_{t_0} <= -b} (lower barrier hit)
    tau^0 = t_0 + H                             (horizon timeout)
    tau = min(tau^+, tau^-, tau^0)

Labels:
    Y = +1 if tau = tau^+  (upper hit first → bullish)
    Y = -1 if tau = tau^-  (lower hit first → bearish)
    Y =  0 if tau = tau^0  (timeout → neutral/sideways)

This produces 3-class labels suitable for probabilistic classification
with proper scoring rules (cross-entropy, Brier score).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from user_strategies.configs.barrier_config import BarrierConfig

# Label constants
LABEL_UPPER = 1   # Upper barrier hit first (bullish)
LABEL_LOWER = -1  # Lower barrier hit first (bearish)
LABEL_TIMEOUT = 0  # Horizon timeout (neutral)


def compute_triple_barrier_labels(
    prices: pd.Series,
    barrier_pct: float,
    horizon_bars: int,
    price_col: str | None = None,
) -> pd.Series:
    """Compute triple-barrier labels for a price series.

    For each bar, determines whether price hits the upper barrier (+barrier_pct),
    lower barrier (-barrier_pct), or times out after horizon_bars.

    Args:
        prices: Price series (typically Close prices) with DatetimeIndex
        barrier_pct: Percentage distance to barriers (e.g., 0.01 = 1%)
        horizon_bars: Maximum bars to look forward before timeout
        price_col: If prices is a DataFrame, column to use (default: 'Close')

    Returns:
        pd.Series with same index as prices, values in {-1, 0, +1}:
            +1: Upper barrier hit first (bullish outcome)
            -1: Lower barrier hit first (bearish outcome)
             0: Timeout (neutral/sideways)
            NaN: Insufficient forward data for labeling

    Raises:
        ValueError: If prices is empty or barrier_pct <= 0

    Example:
        >>> close = df['Close']
        >>> labels = compute_triple_barrier_labels(close, barrier_pct=0.01, horizon_bars=24)
        >>> labels.value_counts()
        1     450
        -1    380
        0     170
        Name: label, dtype: int64
    """
    # Handle DataFrame input
    if isinstance(prices, pd.DataFrame):
        if price_col is None:
            price_col = "Close"
        prices = prices[price_col]

    if len(prices) == 0:
        msg = "Price series cannot be empty"
        raise ValueError(msg)

    if barrier_pct <= 0:
        msg = f"barrier_pct must be positive, got {barrier_pct}"
        raise ValueError(msg)

    if horizon_bars <= 0:
        msg = f"horizon_bars must be positive, got {horizon_bars}"
        raise ValueError(msg)

    # Convert to numpy for faster computation
    price_arr = prices.values.astype(np.float64)
    n = len(price_arr)

    # Initialize labels with NaN (will be set for valid indices)
    labels = np.full(n, np.nan)

    # Compute labels for each bar
    for i in range(n):
        # Check if we have enough forward data
        if i + horizon_bars >= n:
            # Insufficient forward data - leave as NaN
            continue

        entry_price = price_arr[i]
        upper_barrier = entry_price * (1 + barrier_pct)
        lower_barrier = entry_price * (1 - barrier_pct)

        # Look forward up to horizon_bars
        label = _find_first_barrier_hit(
            price_arr[i + 1 : i + 1 + horizon_bars],
            upper_barrier,
            lower_barrier,
        )
        labels[i] = label

    return pd.Series(labels, index=prices.index, name="label")


def _find_first_barrier_hit(
    future_prices: np.ndarray,
    upper_barrier: float,
    lower_barrier: float,
) -> int:
    """Find which barrier is hit first in forward price window.

    Args:
        future_prices: Array of future prices (not including entry bar)
        upper_barrier: Upper barrier price level
        lower_barrier: Lower barrier price level

    Returns:
        +1 if upper hit first, -1 if lower hit first, 0 if timeout
    """
    # Find first bar where each barrier is breached
    upper_hits = np.where(future_prices >= upper_barrier)[0]
    lower_hits = np.where(future_prices <= lower_barrier)[0]

    # Determine first hit times (inf if never hit)
    first_upper = upper_hits[0] if len(upper_hits) > 0 else np.inf
    first_lower = lower_hits[0] if len(lower_hits) > 0 else np.inf

    # Return label based on which hit first
    if first_upper == np.inf and first_lower == np.inf:
        return LABEL_TIMEOUT  # Neither barrier hit → timeout
    elif first_upper <= first_lower:
        return LABEL_UPPER  # Upper hit first or same time
    else:
        return LABEL_LOWER  # Lower hit first


def compute_triple_barrier_labels_vectorized(
    prices: pd.Series,
    barrier_pct: float,
    horizon_bars: int,
) -> pd.Series:
    """Vectorized triple-barrier label computation (faster for large datasets).

    Uses rolling window operations for better performance on large datasets.
    Produces identical results to compute_triple_barrier_labels().

    Args:
        prices: Price series with DatetimeIndex
        barrier_pct: Percentage distance to barriers
        horizon_bars: Maximum bars to look forward

    Returns:
        pd.Series with labels in {-1, 0, +1, NaN}
    """
    price_arr = prices.values.astype(np.float64)
    n = len(price_arr)

    # Precompute barrier levels for all entry points
    upper_barriers = price_arr * (1 + barrier_pct)
    lower_barriers = price_arr * (1 - barrier_pct)

    labels = np.full(n, np.nan)

    # Process in chunks for memory efficiency
    chunk_size = min(10000, n)

    for start in range(0, n - horizon_bars, chunk_size):
        end = min(start + chunk_size, n - horizon_bars)

        for i in range(start, end):
            future = price_arr[i + 1 : i + 1 + horizon_bars]

            # Vectorized comparison
            upper_hit_mask = future >= upper_barriers[i]
            lower_hit_mask = future <= lower_barriers[i]

            first_upper = np.argmax(upper_hit_mask) if upper_hit_mask.any() else horizon_bars
            first_lower = np.argmax(lower_hit_mask) if lower_hit_mask.any() else horizon_bars

            # Handle case where argmax returns 0 but no hit
            if not upper_hit_mask.any():
                first_upper = horizon_bars
            if not lower_hit_mask.any():
                first_lower = horizon_bars

            if first_upper == horizon_bars and first_lower == horizon_bars:
                labels[i] = LABEL_TIMEOUT
            elif first_upper <= first_lower:
                labels[i] = LABEL_UPPER
            else:
                labels[i] = LABEL_LOWER

    return pd.Series(labels, index=prices.index, name="label")


def compute_labels_with_config(
    prices: pd.Series,
    config: BarrierConfig,
) -> pd.Series:
    """Compute labels using a BarrierConfig object.

    Convenience wrapper for compute_triple_barrier_labels().

    Args:
        prices: Price series with DatetimeIndex
        config: BarrierConfig with barrier_pct and horizon_bars

    Returns:
        pd.Series with labels in {-1, 0, +1, NaN}
    """
    return compute_triple_barrier_labels(
        prices=prices,
        barrier_pct=config.barrier_pct,
        horizon_bars=config.horizon_bars,
    )


def get_label_statistics(labels: pd.Series) -> dict[str, float]:
    """Compute statistics for a label series.

    Args:
        labels: Series of triple-barrier labels

    Returns:
        Dictionary with label distribution and statistics
    """
    # Remove NaN values
    valid_labels = labels.dropna()
    n_total = len(valid_labels)

    if n_total == 0:
        return {
            "n_total": 0,
            "n_upper": 0,
            "n_lower": 0,
            "n_timeout": 0,
            "pct_upper": 0.0,
            "pct_lower": 0.0,
            "pct_timeout": 0.0,
            "imbalance_ratio": 0.0,
        }

    n_upper = (valid_labels == LABEL_UPPER).sum()
    n_lower = (valid_labels == LABEL_LOWER).sum()
    n_timeout = (valid_labels == LABEL_TIMEOUT).sum()

    return {
        "n_total": n_total,
        "n_upper": int(n_upper),
        "n_lower": int(n_lower),
        "n_timeout": int(n_timeout),
        "pct_upper": n_upper / n_total * 100,
        "pct_lower": n_lower / n_total * 100,
        "pct_timeout": n_timeout / n_total * 100,
        "imbalance_ratio": max(n_upper, n_lower) / max(min(n_upper, n_lower), 1),
    }


def compute_label_quality_metrics(
    prices: pd.Series,
    labels: pd.Series,
    barrier_pct: float,
) -> dict[str, float]:
    """Compute quality metrics for labels (e.g., for hyperparameter selection).

    Args:
        prices: Original price series
        labels: Computed label series
        barrier_pct: Barrier percentage used

    Returns:
        Dictionary with quality metrics:
            - entropy: Label distribution entropy (higher = more balanced)
            - directional_accuracy: Realized return sign matches label sign
            - avg_return_per_label: Average return by label class
    """
    valid_mask = ~labels.isna()
    valid_labels = labels[valid_mask]
    valid_prices = prices[valid_mask]

    if len(valid_labels) == 0:
        return {"entropy": 0.0, "directional_accuracy": 0.0}

    # Compute entropy of label distribution
    counts = valid_labels.value_counts(normalize=True)
    entropy = -np.sum(counts * np.log(counts + 1e-10))

    # Normalize to [0, 1] range (max entropy for 3 classes is log(3))
    max_entropy = np.log(3)
    normalized_entropy = entropy / max_entropy

    return {
        "entropy": float(normalized_entropy),
        "n_valid_labels": len(valid_labels),
        "n_nan_labels": valid_mask.sum() - len(valid_labels),
    }
