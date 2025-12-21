"""ClickHouse data adapter for backtesting.py integration.

ADR: 2025-12-20-clickhouse-triple-barrier-backtest

This module provides a data adapter that fetches OHLCV data from ClickHouse
via gapless-crypto-clickhouse and maps it to backtesting.py's expected format.

The adapter supports:
- Auto-ingestion for missing data periods
- Column mapping to backtesting.py format (Open, High, Low, Close, Volume)
- Preservation of microstructure columns for feature engineering
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from user_strategies.configs.clickhouse_config import ClickHouseConfig

logger = logging.getLogger(__name__)

# backtesting.py required columns (capitalized)
BACKTESTING_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

# gapless-crypto-clickhouse column mapping
COLUMN_MAPPING = {
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "volume": "Volume",
}

# Microstructure columns to preserve for feature engineering
MICROSTRUCTURE_COLUMNS = [
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
]


def get_data_from_clickhouse(
    symbol: str,
    timeframe: str = "1h",
    start: str | datetime | None = None,
    end: str | datetime | None = None,
    config: ClickHouseConfig | None = None,
    include_microstructure: bool = True,
) -> pd.DataFrame:
    """Fetch OHLCV data from ClickHouse via gapless-crypto-clickhouse.

    This function retrieves cryptocurrency OHLCV data with zero-gap guarantees
    and maps it to backtesting.py's expected format. Optionally includes
    microstructure columns for advanced feature engineering.

    Args:
        symbol: Trading pair symbol (e.g., "BTCUSDT", "ETHUSDT")
        timeframe: CCXT-compatible timeframe string (e.g., "1s", "1m", "1h", "1d")
        start: Start date as string (YYYY-MM-DD) or datetime
        end: End date as string (YYYY-MM-DD) or datetime
        config: Optional ClickHouseConfig; uses environment variables if None
        include_microstructure: If True, include taker/trade columns for features

    Returns:
        pd.DataFrame with DatetimeIndex and columns:
            - Open, High, Low, Close, Volume (always present)
            - quote_asset_volume, number_of_trades, taker_buy_base_asset_volume,
              taker_buy_quote_asset_volume (if include_microstructure=True)

    Raises:
        ImportError: If gapless-crypto-clickhouse is not installed
        ConnectionError: If ClickHouse connection fails
        ValueError: If no data found for the specified parameters

    Example:
        >>> df = get_data_from_clickhouse(
        ...     symbol="BTCUSDT",
        ...     timeframe="1h",
        ...     start="2024-01-01",
        ...     end="2024-06-30",
        ... )
        >>> df.columns.tolist()
        ['Open', 'High', 'Low', 'Close', 'Volume', 'quote_asset_volume', ...]
    """
    try:
        import gapless_crypto_clickhouse as gcch
    except ImportError as e:
        msg = (
            "gapless-crypto-clickhouse is required for ClickHouse data access. "
            "Install with: uv add gapless-crypto-clickhouse"
        )
        raise ImportError(msg) from e

    # Convert datetime to string if needed
    start_str = start.strftime("%Y-%m-%d") if isinstance(start, datetime) else start
    end_str = end.strftime("%Y-%m-%d") if isinstance(end, datetime) else end

    logger.info(
        "Fetching %s data: symbol=%s, timeframe=%s, start=%s, end=%s",
        "with microstructure" if include_microstructure else "OHLCV",
        symbol,
        timeframe,
        start_str,
        end_str,
    )

    # Fetch data with auto-ingestion for missing periods
    # query_ohlcv downloads data if missing from ClickHouse
    df = gcch.query_ohlcv(
        symbol=symbol,
        timeframe=timeframe,
        start=start_str,
        end=end_str,
    )

    if df is None or df.empty:
        msg = f"No data found for {symbol} from {start_str} to {end_str}"
        raise ValueError(msg)

    # Map columns to backtesting.py format
    df = _map_columns_to_backtesting(df, include_microstructure)

    # Ensure DatetimeIndex
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Sort by timestamp (ascending)
    df = df.sort_index()

    logger.info(
        "Loaded %d bars from %s to %s",
        len(df),
        df.index[0],
        df.index[-1],
    )

    return df


def _map_columns_to_backtesting(
    df: pd.DataFrame,
    include_microstructure: bool = True,
) -> pd.DataFrame:
    """Map gapless-crypto-clickhouse columns to backtesting.py format.

    Args:
        df: DataFrame from gapless-crypto-clickhouse
        include_microstructure: If True, preserve microstructure columns

    Returns:
        DataFrame with renamed columns matching backtesting.py expectations
    """
    # Rename OHLCV columns
    df = df.rename(columns=COLUMN_MAPPING)

    # Determine which columns to keep
    columns_to_keep = list(BACKTESTING_COLUMNS)

    if include_microstructure:
        # Add microstructure columns that exist in the data
        for col in MICROSTRUCTURE_COLUMNS:
            if col in df.columns:
                columns_to_keep.append(col)

    # Keep timestamp/index column if present
    if "timestamp" in df.columns:
        columns_to_keep.insert(0, "timestamp")

    # Filter to only columns that exist
    available_columns = [c for c in columns_to_keep if c in df.columns]

    return df[available_columns]


def validate_data_for_backtesting(df: pd.DataFrame) -> bool:
    """Validate that a DataFrame meets backtesting.py requirements.

    Args:
        df: DataFrame to validate

    Returns:
        True if valid, raises ValueError otherwise

    Raises:
        ValueError: If required columns missing or data invalid
    """
    # Check required columns
    missing = set(BACKTESTING_COLUMNS) - set(df.columns)
    if missing:
        msg = f"Missing required columns: {missing}"
        raise ValueError(msg)

    # Check for DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        msg = "DataFrame must have DatetimeIndex"
        raise ValueError(msg)

    # Check for NaN values in OHLCV
    nan_counts = df[BACKTESTING_COLUMNS].isna().sum()
    if nan_counts.any():
        msg = f"NaN values found in OHLCV columns: {nan_counts[nan_counts > 0].to_dict()}"
        raise ValueError(msg)

    # Check for non-positive prices
    price_cols = ["Open", "High", "Low", "Close"]
    for col in price_cols:
        if (df[col] <= 0).any():
            msg = f"Non-positive values found in {col}"
            raise ValueError(msg)

    # Check OHLC relationship: High >= max(Open, Close), Low <= min(Open, Close)
    invalid_high = df["High"] < df[["Open", "Close"]].max(axis=1)
    invalid_low = df["Low"] > df[["Open", "Close"]].min(axis=1)

    if invalid_high.any():
        msg = f"Invalid High values (< Open or Close) at {invalid_high.sum()} rows"
        raise ValueError(msg)

    if invalid_low.any():
        msg = f"Invalid Low values (> Open or Close) at {invalid_low.sum()} rows"
        raise ValueError(msg)

    return True
