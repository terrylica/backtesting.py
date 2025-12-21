"""Microstructure feature computation via ClickHouse SQL.

ADR: 2025-12-20-clickhouse-triple-barrier-backtest

This module provides Python wrappers for executing microstructure feature
SQL queries against ClickHouse. Features are computed from second-level
data and aggregated to the event interval for classification.

Features computed:
- Taker buy ratio (order flow proxy)
- Order flow imbalance (signed volume delta)
- Trade intensity (activity metrics)
- VWAP deviation (price location)
- Range volatility (intrabar dispersion)
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from user_strategies.configs.clickhouse_config import ClickHouseConfig

logger = logging.getLogger(__name__)

# Path to SQL templates
SQL_DIR = Path(__file__).parent.parent / "sql"
COMBINED_QUERY_FILE = SQL_DIR / "microstructure_features.sql"

# Default table name for gapless-crypto-clickhouse
DEFAULT_TABLE = "binance_klines"

# Feature columns produced by combined query
FEATURE_COLUMNS = [
    "taker_buy_ratio",
    "order_flow_imbalance",
    "total_trades",
    "avg_trades_per_bar",
    "vwap",
    "vwap_deviation_pct",
    "range_volatility",
]


def compute_microstructure_features(
    symbol: str,
    start: str | datetime,
    end: str | datetime,
    interval_seconds: int = 3600,
    config: ClickHouseConfig | None = None,
    table_name: str = DEFAULT_TABLE,
) -> pd.DataFrame:
    """Compute microstructure features from ClickHouse second-level data.

    Executes the combined feature query against ClickHouse to compute
    all microstructure features in a single pass. Features are aggregated
    from source bars (e.g., 1-second) to the specified interval.

    Args:
        symbol: Trading pair symbol (e.g., "BTCUSDT")
        start: Start datetime or string (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)
        end: End datetime or string
        interval_seconds: Aggregation interval in seconds (default: 3600 = 1 hour)
        config: Optional ClickHouseConfig; uses environment if None
        table_name: ClickHouse table name (default: binance_klines)

    Returns:
        pd.DataFrame with DatetimeIndex and columns:
            - OHLCV: open, high, low, close, volume
            - Features: taker_buy_ratio, order_flow_imbalance, total_trades,
                       avg_trades_per_bar, vwap, vwap_deviation_pct, range_volatility
            - Metadata: source_bar_count, quote_volume

    Raises:
        ImportError: If clickhouse-driver not installed
        ConnectionError: If ClickHouse connection fails
        ValueError: If no data found

    Example:
        >>> df = compute_microstructure_features(
        ...     symbol="BTCUSDT",
        ...     start="2024-01-01",
        ...     end="2024-01-31",
        ...     interval_seconds=3600,  # hourly
        ... )
        >>> df[FEATURE_COLUMNS].describe()
    """
    try:
        from clickhouse_driver import Client
    except ImportError as e:
        msg = (
            "clickhouse-driver is required for direct SQL queries. "
            "Install with: uv add clickhouse-driver"
        )
        raise ImportError(msg) from e

    # Load config
    if config is None:
        from user_strategies.configs.clickhouse_config import DEFAULT_CONFIG

        config = DEFAULT_CONFIG

    # Parse dates
    start_str = _format_datetime(start)
    end_str = _format_datetime(end)

    logger.info(
        "Computing microstructure features: symbol=%s, interval=%ds, %s to %s",
        symbol,
        interval_seconds,
        start_str,
        end_str,
    )

    # Build query from template
    query = _build_combined_query(
        symbol=symbol,
        start_time=start_str,
        end_time=end_str,
        interval_seconds=interval_seconds,
        table_name=table_name,
    )

    # Execute query
    client = Client(**config.to_dict())

    try:
        result = client.execute(query, with_column_types=True)
        data, columns = result
        column_names = [col[0] for col in columns]
    finally:
        client.disconnect()

    if not data:
        msg = f"No data found for {symbol} from {start_str} to {end_str}"
        raise ValueError(msg)

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=column_names)

    # Set DatetimeIndex
    df["interval_start"] = pd.to_datetime(df["interval_start"])
    df = df.set_index("interval_start")
    df = df.sort_index()

    logger.info(
        "Computed features for %d intervals from %s to %s",
        len(df),
        df.index[0],
        df.index[-1],
    )

    return df


def _build_combined_query(
    symbol: str,
    start_time: str,
    end_time: str,
    interval_seconds: int,
    table_name: str,
) -> str:
    """Build the combined feature query with parameters.

    Args:
        symbol: Trading pair symbol
        start_time: Start timestamp string
        end_time: End timestamp string
        interval_seconds: Aggregation interval
        table_name: ClickHouse table name

    Returns:
        Parameterized SQL query string
    """
    # Extract combined query from SQL file
    query_template = _extract_combined_query()

    # Substitute parameters
    query = query_template.format(
        symbol=f"'{symbol}'",
        start_time=f"'{start_time}'",
        end_time=f"'{end_time}'",
        interval_seconds=interval_seconds,
        table_name=table_name,
    )

    return query


def _extract_combined_query() -> str:
    """Extract the combined feature query from the SQL file.

    Returns:
        SQL query template string
    """
    if not COMBINED_QUERY_FILE.exists():
        # Fallback inline query if file not found
        return _get_inline_combined_query()

    content = COMBINED_QUERY_FILE.read_text()

    # Find the COMBINED FEATURE QUERY section
    marker = "-- COMBINED FEATURE QUERY"
    start_idx = content.find(marker)

    if start_idx == -1:
        return _get_inline_combined_query()

    # Extract from marker to end of file
    query_section = content[start_idx:]

    # Find the actual SELECT statement
    select_idx = query_section.find("SELECT")
    if select_idx == -1:
        return _get_inline_combined_query()

    return query_section[select_idx:]


def _get_inline_combined_query() -> str:
    """Return inline combined query as fallback.

    Returns:
        SQL query template string
    """
    return """
SELECT
    toStartOfInterval(timestamp, INTERVAL {interval_seconds} SECOND) AS interval_start,

    -- OHLCV aggregates
    argMin(open, timestamp) AS open,
    max(high) AS high,
    min(low) AS low,
    argMax(close, timestamp) AS close,
    sum(volume) AS volume,

    -- Feature 1: Taker Buy Ratio
    sum(taker_buy_base_asset_volume) / nullIf(sum(volume), 0) AS taker_buy_ratio,

    -- Feature 2: Order Flow Imbalance
    (2 * sum(taker_buy_base_asset_volume) - sum(volume)) / nullIf(sum(volume), 0) AS order_flow_imbalance,

    -- Feature 3: Trade Intensity
    sum(number_of_trades) AS total_trades,
    avg(number_of_trades) AS avg_trades_per_bar,

    -- Feature 4: VWAP and Deviation
    sum(close * volume) / nullIf(sum(volume), 0) AS vwap,
    (argMax(close, timestamp) - sum(close * volume) / nullIf(sum(volume), 0))
        / nullIf(sum(close * volume) / nullIf(sum(volume), 0), 0) AS vwap_deviation_pct,

    -- Feature 5: Range Volatility
    (max(high) - min(low)) / nullIf(avg(close), 0) AS range_volatility,

    -- Metadata
    count(*) AS source_bar_count,
    sum(quote_asset_volume) AS quote_volume

FROM {table_name}
WHERE symbol = {symbol}
  AND timestamp >= {start_time}
  AND timestamp < {end_time}
GROUP BY interval_start
ORDER BY interval_start
"""


def _format_datetime(dt: str | datetime) -> str:
    """Format datetime to ClickHouse-compatible string.

    Args:
        dt: Datetime object or string

    Returns:
        Formatted string (YYYY-MM-DD HH:MM:SS)
    """
    if isinstance(dt, datetime):
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    # Assume string is already properly formatted
    return dt


def normalize_features(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    method: str = "zscore",
    window: int | None = None,
) -> pd.DataFrame:
    """Normalize features for ML model input.

    Args:
        df: DataFrame with feature columns
        feature_cols: Columns to normalize (default: FEATURE_COLUMNS)
        method: Normalization method ('zscore', 'minmax', 'robust')
        window: Rolling window for normalization (None = global)

    Returns:
        DataFrame with normalized features (suffixed with '_norm')
    """
    if feature_cols is None:
        feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]

    result = df.copy()

    for col in feature_cols:
        if col not in df.columns:
            continue

        norm_col = f"{col}_norm"

        if window is not None:
            # Rolling normalization
            if method == "zscore":
                roll_mean = df[col].rolling(window).mean()
                roll_std = df[col].rolling(window).std()
                result[norm_col] = (df[col] - roll_mean) / roll_std.replace(0, 1)
            elif method == "minmax":
                roll_min = df[col].rolling(window).min()
                roll_max = df[col].rolling(window).max()
                result[norm_col] = (df[col] - roll_min) / (roll_max - roll_min).replace(
                    0, 1
                )
            elif method == "robust":
                roll_median = df[col].rolling(window).median()
                roll_iqr = df[col].rolling(window).quantile(0.75) - df[col].rolling(
                    window
                ).quantile(0.25)
                result[norm_col] = (df[col] - roll_median) / roll_iqr.replace(0, 1)
        else:
            # Global normalization
            if method == "zscore":
                result[norm_col] = (df[col] - df[col].mean()) / df[col].std()
            elif method == "minmax":
                result[norm_col] = (df[col] - df[col].min()) / (
                    df[col].max() - df[col].min()
                )
            elif method == "robust":
                median = df[col].median()
                iqr = df[col].quantile(0.75) - df[col].quantile(0.25)
                result[norm_col] = (df[col] - median) / iqr

    return result
