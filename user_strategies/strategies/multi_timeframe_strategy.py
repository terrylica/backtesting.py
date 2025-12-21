"""
Multi-Timeframe Strategy with Second-Granularity Stop-Loss.

ADR: 2025-12-20-multi-timeframe-second-granularity

This module implements a multi-timeframe backtesting strategy that:
1. Runs on 1-second OHLCV data (86,400 bars/day)
2. Derives entry/exit signals from hourly indicators via resample_apply
3. Uses TrailingStrategy for dynamic stop-loss protection checked every second
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from backtesting import Backtest
from backtesting.lib import TrailingStrategy, resample_apply

if TYPE_CHECKING:
    from backtesting._typing import Array


def SMA(series: Array, n: int) -> pd.Series:
    """Simple Moving Average indicator."""
    return pd.Series(series).rolling(n).mean()


class MultiTimeframeTrailingStrategy(TrailingStrategy):
    """
    Multi-timeframe strategy with second-granularity stop-loss.

    - Entry signals: Hourly SMA crossover (via resample_apply)
    - Stop-loss: ATR-based trailing stop checked every bar (1-second with 1s data)

    The key insight is that while entry/exit DECISIONS are made on hourly
    indicators, the trailing stop-loss is CHECKED on every bar. With 1-second
    data, this means 86,400 stop-loss checks per day instead of just 24.

    ADR: 2025-12-20-multi-timeframe-second-granularity
    """

    # Strategy parameters (can be optimized)
    hourly_sma_fast: int = 10
    hourly_sma_slow: int = 30
    atr_periods: int = 100
    trailing_atr_multiplier: float = 2.0

    def init(self):
        """Initialize strategy with hourly indicators resampled from second data."""
        super().init()

        # Set ATR periods for trailing stop calculation
        self.set_atr_periods(self.atr_periods)

        # Build HOURLY indicators from SECOND (or any) data
        # resample_apply forward-fills the hourly values to match bar frequency
        self.hourly_sma_fast_line = resample_apply(
            '1h', SMA, self.data.Close, self.hourly_sma_fast
        )
        self.hourly_sma_slow_line = resample_apply(
            '1h', SMA, self.data.Close, self.hourly_sma_slow
        )

    def next(self):
        """
        Execute on every bar (every second with 1s data).

        TrailingStrategy.next() updates stop-loss levels on EVERY bar,
        providing maximum precision for profit protection.
        """
        super().next()  # TrailingStrategy handles stop-loss updates

        # Skip if not enough data for indicators
        if len(self.data) < 2:
            return

        # Check for NaN in indicators (warmup period)
        if np.isnan(self.hourly_sma_fast_line[-1]) or np.isnan(self.hourly_sma_slow_line[-1]):
            return

        # Get hourly signals (forward-filled to second level)
        fast = self.hourly_sma_fast_line[-1]
        slow = self.hourly_sma_slow_line[-1]
        fast_prev = self.hourly_sma_fast_line[-2]
        slow_prev = self.hourly_sma_slow_line[-2]

        # Entry logic: SMA crossover (only when not in position)
        if not self.position:
            # Bullish crossover: fast crosses above slow
            if fast_prev <= slow_prev and fast > slow:
                self.buy()
                self.set_trailing_sl(self.trailing_atr_multiplier)

            # Bearish crossover: fast crosses below slow (for short positions)
            elif fast_prev >= slow_prev and fast < slow:
                self.sell()
                self.set_trailing_sl(self.trailing_atr_multiplier)


def load_second_data(
    symbol: str = 'BTCUSDT',
    start: str = '2024-01-01',
    end: str = '2024-01-02',
) -> pd.DataFrame:
    """
    Load 1-second OHLCV data from gapless-crypto-clickhouse.

    Args:
        symbol: Trading pair (default: BTCUSDT)
        start: Start date (YYYY-MM-DD or datetime string)
        end: End date (YYYY-MM-DD or datetime string)

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume
        Index is DatetimeIndex at second frequency

    Note:
        Requires gapless-crypto-clickhouse package and ClickHouse connection.
        ~86,400 bars per day, ~16MB memory per day, ~45s download per day.
    """
    try:
        import gapless_crypto_clickhouse as gcch
    except ImportError as e:
        raise ImportError(
            "gapless-crypto-clickhouse required for 1s data. "
            "Install with: uv add gapless-crypto-clickhouse"
        ) from e

    df = gcch.download(
        symbol=symbol,
        timeframe='1s',
        start=start,
        end=end,
        index_type='datetime'
    )

    # Rename to backtesting.py format
    df = df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })

    return df[['Open', 'High', 'Low', 'Close', 'Volume']]


def run_multi_timeframe_demo(
    symbol: str = 'BTCUSDT',
    start: str = '2024-01-01',
    end: str = '2024-01-02',
    cash: float = 10_000,
    commission: float = 0.001,
    output_dir: str | Path | None = None,
) -> dict:
    """
    Run demo backtest with 1-second data and hourly signals.

    Args:
        symbol: Trading pair
        start: Start date
        end: End date
        cash: Starting capital
        commission: Trading commission rate
        output_dir: Directory for output files (default: user_strategies/data/backtests)

    Returns:
        Backtest statistics dictionary
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / 'data' / 'backtests'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading 1s data for {symbol} from {start} to {end}...")
    data = load_second_data(symbol, start, end)
    print(f"Loaded {len(data):,} bars ({len(data)/86400:.1f} days)")

    bt = Backtest(
        data,
        MultiTimeframeTrailingStrategy,
        cash=cash,
        commission=commission,
        exclusive_orders=True
    )

    print("Running backtest with second-granularity stop-loss...")
    stats = bt.run()

    print(f"\n{'='*60}")
    print("MULTI-TIMEFRAME BACKTEST RESULTS")
    print(f"{'='*60}")
    print(f"  Symbol:        {symbol}")
    print(f"  Period:        {start} to {end}")
    print(f"  Bars:          {len(data):,} (1-second)")
    print(f"  Stop checks:   {len(data):,}/period (vs {len(data)//3600:,} with 1H data)")
    print(f"{'='*60}")
    print(f"  Return:        {stats['Return [%]']:.2f}%")
    print(f"  Sharpe:        {stats['Sharpe Ratio']:.2f}")
    print(f"  Max Drawdown:  {stats['Max. Drawdown [%]']:.2f}%")
    print(f"  Win Rate:      {stats['Win Rate [%]']:.1f}%")
    print(f"  Trades:        {stats['# Trades']}")
    print(f"{'='*60}")

    # Save HTML visualization
    html_path = output_dir / f"multi_timeframe_{symbol}_{start}_{end}.html"
    bt.plot(filename=str(html_path), open_browser=False)
    print(f"\nVisualization saved: {html_path}")

    return stats


if __name__ == '__main__':
    # Demo with 1 day of 1-second data
    run_multi_timeframe_demo(
        symbol='BTCUSDT',
        start='2024-01-01',
        end='2024-01-02',
    )
