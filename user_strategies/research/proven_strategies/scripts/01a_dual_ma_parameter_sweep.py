#!/usr/bin/env python3
"""
Phase 14A: Dual MA Crossover - Parameter Sweep

Test alternative MA period combinations to determine if trend following
can work on crypto 5-minute data with parameter adjustment.

TEST MATRIX:
- Fast MA: [20, 50, 100]
- Slow MA: [100, 200, 300]
- Constraint: fast < slow

HYPOTHESIS: If all combinations fail (<35% win rate), crypto 5-min is
fundamentally incompatible with trend following.

VERSION: 1.0.0
DATE: 2025-10-05
"""

import pandas as pd
import numpy as np
from pathlib import Path
from backtesting import Strategy, Backtest
import warnings
warnings.filterwarnings('ignore')


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range (reused from Phase 10D)."""
    high = df['High']
    low = df['Low']
    close = df['Close']
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    return atr


class DualMACrossoverStrategy(Strategy):
    """Dual MA Crossover with parameterized periods."""

    # Parameterized periods
    ma_fast_period = 50
    ma_slow_period = 200
    atr_period = 14
    stop_atr_multiple = 2.0
    trailing_atr_multiple = 3.0
    max_hold_bars = 500

    def init(self):
        """Initialize indicators."""

        if self.data.df.isnull().sum().sum() > 0:
            raise ValueError(f"NaN values in input data: {self.data.df.isnull().sum()}")

        # Require enough data for slow MA
        min_bars = max(self.ma_slow_period, self.ma_fast_period) + 50
        if len(self.data.df) < min_bars:
            raise ValueError(f"Insufficient data: {len(self.data.df)} bars (need ≥{min_bars})")

        df = self.data.df.copy()

        # Moving averages
        self.ma_fast = self.I(
            lambda: df['Close'].rolling(self.ma_fast_period).mean(),
            name='MA_Fast'
        )
        self.ma_slow = self.I(
            lambda: df['Close'].rolling(self.ma_slow_period).mean(),
            name='MA_Slow'
        )

        # ATR for stops
        self.atr = self.I(lambda: calculate_atr(df, self.atr_period), name='ATR')

        # Position tracking
        self.entry_bar = None
        self.entry_price = None
        self.stop_loss = None
        self.highest_high = None
        self.lowest_low = None

    def next(self):
        """Execute strategy logic."""
        if self.position:
            self._manage_exits()
            return
        self._check_entries()

    def _check_entries(self):
        """Check for entry signals."""
        if len(self.data) < max(self.ma_fast_period, self.ma_slow_period) + 10:
            return

        if pd.isna(self.ma_fast[-1]) or pd.isna(self.ma_slow[-1]):
            raise RuntimeError(
                f"NaN MA at bar {len(self.data)}: "
                f"fast={self.ma_fast[-1]}, slow={self.ma_slow[-1]}"
            )

        if pd.isna(self.atr[-1]) or self.atr[-1] <= 0:
            raise RuntimeError(f"Invalid ATR at bar {len(self.data)}: {self.atr[-1]}")

        if len(self.data) < 2:
            return

        ma_fast_prev = self.ma_fast[-2]
        ma_slow_prev = self.ma_slow[-2]
        ma_fast_curr = self.ma_fast[-1]
        ma_slow_curr = self.ma_slow[-1]

        current_price = self.data.Close[-1]
        current_atr = self.atr[-1]

        # LONG: Fast crosses above slow
        if ma_fast_prev <= ma_slow_prev and ma_fast_curr > ma_slow_curr:
            self.buy(size=0.95)
            self.entry_bar = len(self.data)
            self.entry_price = current_price
            self.stop_loss = current_price - (self.stop_atr_multiple * current_atr)
            self.highest_high = current_price

        # SHORT: Fast crosses below slow
        elif ma_fast_prev >= ma_slow_prev and ma_fast_curr < ma_slow_curr:
            self.sell(size=0.95)
            self.entry_bar = len(self.data)
            self.entry_price = current_price
            self.stop_loss = current_price + (self.stop_atr_multiple * current_atr)
            self.lowest_low = current_price

    def _manage_exits(self):
        """Manage position exits."""
        bars_held = len(self.data) - self.entry_bar
        current_price = self.data.Close[-1]
        current_atr = self.atr[-1]

        if pd.isna(current_atr) or current_atr <= 0:
            raise RuntimeError(f"Invalid ATR during exit at bar {len(self.data)}: {current_atr}")

        # Exit: Max hold time
        if bars_held >= self.max_hold_bars:
            self.position.close()
            self._reset_position_tracking()
            return

        # Exit: Opposite crossover
        if len(self.data) >= 2:
            ma_fast_prev = self.ma_fast[-2]
            ma_slow_prev = self.ma_slow[-2]
            ma_fast_curr = self.ma_fast[-1]
            ma_slow_curr = self.ma_slow[-1]

            if self.position.is_long and ma_fast_prev >= ma_slow_prev and ma_fast_curr < ma_slow_curr:
                self.position.close()
                self._reset_position_tracking()
                return

            if self.position.is_short and ma_fast_prev <= ma_slow_prev and ma_fast_curr > ma_slow_curr:
                self.position.close()
                self._reset_position_tracking()
                return

        # Exit: Stop loss
        if self.position.is_long and self.data.Low[-1] <= self.stop_loss:
            self.position.close()
            self._reset_position_tracking()
            return

        if self.position.is_short and self.data.High[-1] >= self.stop_loss:
            self.position.close()
            self._reset_position_tracking()
            return

        # Exit: Trailing stop
        if self.position.is_long:
            if self.data.High[-1] > self.highest_high:
                self.highest_high = self.data.High[-1]

            trailing_stop = self.highest_high - (self.trailing_atr_multiple * current_atr)
            self.stop_loss = max(self.stop_loss, trailing_stop)

            if self.data.Low[-1] <= trailing_stop:
                self.position.close()
                self._reset_position_tracking()
                return

        if self.position.is_short:
            if self.data.Low[-1] < self.lowest_low:
                self.lowest_low = self.data.Low[-1]

            trailing_stop = self.lowest_low + (self.trailing_atr_multiple * current_atr)
            self.stop_loss = min(self.stop_loss, trailing_stop)

            if self.data.High[-1] >= trailing_stop:
                self.position.close()
                self._reset_position_tracking()
                return

    def _reset_position_tracking(self):
        """Reset position tracking variables."""
        self.entry_bar = None
        self.entry_price = None
        self.stop_loss = None
        self.highest_high = None
        self.lowest_low = None


def load_5m_data(csv_path: Path, n_bars: int = None) -> pd.DataFrame:
    """Load 5-minute crypto data."""
    df = pd.read_csv(csv_path, skiprows=10)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    df = df.rename(columns={
        'open': 'Open', 'high': 'High', 'low': 'Low',
        'close': 'Close', 'volume': 'Volume'
    })

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

    if n_bars:
        df = df.tail(n_bars)

    return df


# ============================================================================
# PARAMETER SWEEP
# ============================================================================

def run_parameter_sweep():
    """Run parameter sweep for dual MA crossover."""

    data_dir = Path('/Users/terryli/eon/backtesting.py/user_strategies/data/raw/crypto_5m')
    csv_path = data_dir / 'binance_spot_ETHUSDT-5m_20220101-20250930_v2.10.0.csv'

    print("="*70)
    print("PHASE 14A: DUAL MA CROSSOVER - PARAMETER SWEEP")
    print("="*70)

    print("\nLoading ETH data...")
    df = load_5m_data(csv_path, n_bars=None)
    print(f"Loaded {len(df):,} bars ({df.index[0].date()} to {df.index[-1].date()})")

    # Test matrix
    fast_periods = [20, 50, 100]
    slow_periods = [100, 200, 300]

    results = []

    print("\nRunning parameter sweep...")
    print(f"Test combinations: {len(fast_periods) * len(slow_periods)}")
    print()

    for fast in fast_periods:
        for slow in slow_periods:
            # Constraint: fast < slow
            if fast >= slow:
                continue

            print(f"Testing MA {fast}/{slow}...", end=' ')

            try:
                bt = Backtest(
                    df, DualMACrossoverStrategy,
                    cash=10_000_000,
                    commission=0.0002,
                    margin=0.05,
                    exclusive_orders=True
                )
                stats = bt.run(
                    ma_fast_period=fast,
                    ma_slow_period=slow
                )

                result = {
                    'ma_fast': fast,
                    'ma_slow': slow,
                    'return_pct': stats['Return [%]'],
                    'n_trades': stats['# Trades'],
                    'win_rate_pct': stats['Win Rate [%]'],
                    'sharpe': stats['Sharpe Ratio'],
                    'max_dd_pct': stats['Max. Drawdown [%]']
                }

                results.append(result)

                # Print concise result
                print(f"Win Rate: {stats['Win Rate [%]']:5.1f}%, "
                      f"Return: {stats['Return [%]']:+7.2f}%, "
                      f"Trades: {stats['# Trades']:3d}")

            except Exception as e:
                print(f"ERROR: {e}")

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Sort by win rate descending
    results_df = results_df.sort_values('win_rate_pct', ascending=False)

    print(f"\n{'='*70}")
    print("PARAMETER SWEEP RESULTS")
    print(f"{'='*70}\n")

    print(results_df.to_string(index=False))

    # Analysis
    print(f"\n{'='*70}")
    print("ANALYSIS")
    print(f"{'='*70}")

    best = results_df.iloc[0]
    worst = results_df.iloc[-1]

    print(f"\nBest configuration:")
    print(f"  MA: {int(best['ma_fast'])}/{int(best['ma_slow'])}")
    print(f"  Win Rate: {best['win_rate_pct']:.1f}%")
    print(f"  Return: {best['return_pct']:+.2f}%")
    print(f"  Sharpe: {best['sharpe']:+.2f}")
    print(f"  Trades: {int(best['n_trades'])}")

    print(f"\nWorst configuration:")
    print(f"  MA: {int(worst['ma_fast'])}/{int(worst['ma_slow'])}")
    print(f"  Win Rate: {worst['win_rate_pct']:.1f}%")
    print(f"  Return: {worst['return_pct']:+.2f}%")

    # Check if ANY configuration meets Gate 1 criteria
    gate1_pass = (
        (best['win_rate_pct'] >= 35) and
        (best['return_pct'] > 0) and
        (best['n_trades'] >= 10) and
        (best['sharpe'] > 0)
    )

    print(f"\n{'='*70}")
    print("GATE 1 CHECK (Best Configuration)")
    print(f"{'='*70}")

    criteria = [
        (best['win_rate_pct'] >= 35, f"Win Rate ≥ 35%:  {best['win_rate_pct']:.1f}%"),
        (best['return_pct'] > 0, f"Return > 0%:     {best['return_pct']:+.2f}%"),
        (best['n_trades'] >= 10, f"Trades ≥ 10:     {int(best['n_trades'])}"),
        (best['sharpe'] > 0, f"Sharpe > 0.0:    {best['sharpe']:+.2f}")
    ]

    for passed, message in criteria:
        status = '✅ PASS' if passed else '❌ FAIL'
        print(f"{message} - {status}")

    if gate1_pass:
        print(f"\n✅ GATE 1: PASS - Proceed to Phase 14B with MA {int(best['ma_fast'])}/{int(best['ma_slow'])}")
    else:
        print(f"\n❌ GATE 1: FAIL - All MA combinations fail on crypto 5-min data")
        print(f"\nCONCLUSION:")
        print(f"  - Tested {len(results_df)} MA combinations")
        print(f"  - Best win rate: {best['win_rate_pct']:.1f}% (need ≥35%)")
        print(f"  - Best return: {best['return_pct']:+.2f}% (need >0%)")
        print(f"  - Trend following does NOT work on crypto 5-minute data")
        print(f"\nRECOMMENDATION:")
        print(f"  - Skip to Phase 15: Mean Reversion from Extremes")
        print(f"  - Hypothesis: Crypto 5-min is mean-reverting, not trending")

        raise RuntimeError("GATE 1 FAIL: All MA combinations fail trend following baseline")

    # Save results
    results_dir = Path('/Users/terryli/eon/backtesting.py/user_strategies/research/proven_strategies/results/phase_14_trend_following')
    results_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / 'phase_14a_parameter_sweep.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Results saved: {results_path}")

    return results_df


if __name__ == '__main__':
    results = run_parameter_sweep()
