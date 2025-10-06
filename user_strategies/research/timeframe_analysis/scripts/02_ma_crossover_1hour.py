#!/usr/bin/env python3
"""
Phase 16B: Dual MA Crossover on 1-Hour Data

CONTEXT:
- Phase 16A (15-minute): FAILED with 38.7% win rate, -99.97% return
- 5-minute (Phase 14A): FAILED with 40.3% win rate, -100% return

HYPOTHESIS:
1-hour timeframe (12× aggregation from 5-min) provides maximum noise
reduction and trend persistence for crypto intraday trading.

STRATEGIES TESTED:
- MA 100/300: Best from 5-min (40.3% win rate)
- MA 50/200: Traditional baseline
- MA 20/50: Shorter for lower bar count (~32k bars)

SUCCESS CRITERIA:
- Win rate ≥ 45%
- Return > 0%
- Trades ≥ 10 (lower threshold due to fewer bars)
- Sharpe > 0.5

FINAL TEST: If this fails, crypto directional trading abandoned entirely.

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
    """Dual MA Crossover with parameterized periods (reused from Phase 14A)."""

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

        min_bars = max(self.ma_slow_period, self.ma_fast_period) + 50
        if len(self.data.df) < min_bars:
            raise ValueError(f"Insufficient data: {len(self.data.df)} bars (need ≥{min_bars})")

        df = self.data.df.copy()

        self.ma_fast = self.I(
            lambda: df['Close'].rolling(self.ma_fast_period).mean(),
            name='MA_Fast'
        )
        self.ma_slow = self.I(
            lambda: df['Close'].rolling(self.ma_slow_period).mean(),
            name='MA_Slow'
        )
        self.atr = self.I(lambda: calculate_atr(df, self.atr_period), name='ATR')

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


def load_and_resample_1hour(csv_path: Path) -> pd.DataFrame:
    """
    Load 5-minute data and resample to 1-hour.

    Resampling rules (out-of-the-box pandas):
    - Open: first value in 1-hour window
    - High: max value in 1-hour window
    - Low: min value in 1-hour window
    - Close: last value in 1-hour window
    - Volume: sum of volumes in 1-hour window
    """
    # Load 5-minute data
    df_5m = pd.read_csv(csv_path, skiprows=10)
    df_5m['date'] = pd.to_datetime(df_5m['date'])
    df_5m = df_5m.set_index('date')

    df_5m = df_5m.rename(columns={
        'open': 'Open', 'high': 'High', 'low': 'Low',
        'close': 'Close', 'volume': 'Volume'
    })

    df_5m = df_5m[['Open', 'High', 'Low', 'Close', 'Volume']]

    print(f"Loaded 5-minute data: {len(df_5m):,} bars ({df_5m.index[0].date()} to {df_5m.index[-1].date()})")

    # Resample to 1-hour (out-of-the-box pandas)
    df_1h = df_5m.resample('1h').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()

    print(f"Resampled to 1-hour: {len(df_1h):,} bars ({df_1h.index[0].date()} to {df_1h.index[-1].date()})")

    # Validate resampling
    expected_ratio = 12.0  # 1-hour / 5-min = 12
    actual_ratio = len(df_5m) / len(df_1h)
    ratio_error = abs(actual_ratio - expected_ratio) / expected_ratio

    if ratio_error > 0.10:
        raise RuntimeError(
            f"Resampling ratio incorrect: expected ~{expected_ratio}:1, "
            f"got {actual_ratio:.2f}:1 (error: {ratio_error:.1%})"
        )

    # Validate OHLC integrity
    if (df_1h['Low'] > df_1h['High']).any():
        raise RuntimeError("OHLC integrity violated: Low > High after resampling")

    if (df_1h['Open'] > df_1h['High']).any() or (df_1h['Open'] < df_1h['Low']).any():
        raise RuntimeError("OHLC integrity violated: Open outside High/Low range")

    if (df_1h['Close'] > df_1h['High']).any() or (df_1h['Close'] < df_1h['Low']).any():
        raise RuntimeError("OHLC integrity violated: Close outside High/Low range")

    if df_1h.isnull().sum().sum() > 0:
        raise ValueError(f"NaN values after resampling: {df_1h.isnull().sum()}")

    print(f"✓ Resampling validation passed (ratio: {actual_ratio:.2f}:1, OHLC integrity verified)")

    return df_1h


# ============================================================================
# PHASE 16B VALIDATION
# ============================================================================

def run_phase_16b_validation():
    """Run Phase 16B validation: MA crossover on 1-hour data."""

    data_dir = Path('/Users/terryli/eon/backtesting.py/user_strategies/data/raw/crypto_5m')
    csv_path = data_dir / 'binance_spot_ETHUSDT-5m_20220101-20250930_v2.10.0.csv'

    print("="*70)
    print("PHASE 16B: DUAL MA CROSSOVER - 1-HOUR VALIDATION")
    print("="*70)
    print("\n📊 TIMEFRAME CHANGE: 5-minute → 1-hour (12× aggregation)")
    print("   Context: Phase 16A (15-min) FAILED with 38.7% win rate")
    print("   Hypothesis: Maximum noise reduction enables profitable trading\n")

    # Load and resample data
    print("Loading and resampling data...")
    df_1h = load_and_resample_1hour(csv_path)

    # Test three MA configurations
    configs = [
        {'ma_fast': 100, 'ma_slow': 300, 'name': '100/300 (best from 5-min)'},
        {'ma_fast': 50, 'ma_slow': 200, 'name': '50/200 (traditional)'},
        {'ma_fast': 20, 'ma_slow': 50, 'name': '20/50 (shorter for lower bar count)'}
    ]

    results = []

    for config in configs:
        print(f"\n{'='*70}")
        print(f"Testing MA {config['ma_fast']}/{config['ma_slow']}")
        print(f"{'='*70}")

        bt = Backtest(
            df_1h, DualMACrossoverStrategy,
            cash=10_000_000,
            commission=0.0002,
            margin=0.05,
            exclusive_orders=True
        )
        stats = bt.run(
            ma_fast_period=config['ma_fast'],
            ma_slow_period=config['ma_slow']
        )

        print(f"\nResults:")
        print(f"  Return:   {stats['Return [%]']:+.2f}%")
        print(f"  Trades:   {stats['# Trades']}")
        print(f"  Win Rate: {stats['Win Rate [%]']:.1f}%")
        print(f"  Sharpe:   {stats['Sharpe Ratio']:+.2f}")
        print(f"  Max DD:   {stats['Max. Drawdown [%]']:.2f}%")

        results.append({
            'config': f"MA {config['ma_fast']}/{config['ma_slow']}",
            'timeframe': '1-hour',
            'ma_fast': config['ma_fast'],
            'ma_slow': config['ma_slow'],
            'return_pct': stats['Return [%]'],
            'n_trades': stats['# Trades'],
            'win_rate_pct': stats['Win Rate [%]'],
            'sharpe': stats['Sharpe Ratio'],
            'max_dd_pct': stats['Max. Drawdown [%]'],
        })

    # Gate 2 check
    print(f"\n{'='*70}")
    print("GATE 2 CHECK: 1-HOUR TIMEFRAME")
    print(f"{'='*70}\n")

    results_df = pd.DataFrame(results)
    best = results_df.loc[results_df['win_rate_pct'].idxmax()]

    print(f"Best configuration: MA {int(best['ma_fast'])}/{int(best['ma_slow'])}")
    print(f"  Win Rate: {best['win_rate_pct']:.1f}%")
    print(f"  Return:   {best['return_pct']:+.2f}%")
    print(f"  Sharpe:   {best['sharpe']:+.2f}")
    print(f"  Trades:   {int(best['n_trades'])}\n")

    gate2_criteria = [
        (best['win_rate_pct'] >= 45, f"Win Rate ≥ 45%:  {best['win_rate_pct']:.1f}%"),
        (best['return_pct'] > 0, f"Return > 0%:     {best['return_pct']:+.2f}%"),
        (best['n_trades'] >= 10, f"Trades ≥ 10:     {int(best['n_trades'])}"),
        (best['sharpe'] > 0.5, f"Sharpe > 0.5:    {best['sharpe']:+.2f}")
    ]

    gate2_pass = all(criterion[0] for criterion in gate2_criteria)

    for passed, message in gate2_criteria:
        status = '✅ PASS' if passed else '❌ FAIL'
        print(f"{message} - {status}")

    # Comprehensive comparison
    print(f"\n{'='*70}")
    print("COMPREHENSIVE TIMEFRAME COMPARISON")
    print(f"{'='*70}\n")

    print("MA 100/300 across all timeframes:")
    print(f"  5-minute:  40.3% win rate, -100.00% return (Phase 14A)")
    print(f"  15-minute: 38.7% win rate, -99.97% return (Phase 16A)")
    print(f"  1-hour:    {results_df[results_df['ma_fast']==100]['win_rate_pct'].values[0]:.1f}% win rate, "
          f"{results_df[results_df['ma_fast']==100]['return_pct'].values[0]:+.2f}% return (Phase 16B)\n")

    print("MA 50/200 across all timeframes:")
    print(f"  5-minute:  30.2% win rate, -100.00% return (Phase 14A)")
    print(f"  15-minute: 36.7% win rate, -100.00% return (Phase 16A)")
    print(f"  1-hour:    {results_df[results_df['ma_fast']==50]['win_rate_pct'].values[0]:.1f}% win rate, "
          f"{results_df[results_df['ma_fast']==50]['return_pct'].values[0]:+.2f}% return (Phase 16B)\n")

    # Final decision
    print(f"{'='*70}")
    print("FINAL DECISION")
    print(f"{'='*70}\n")

    if gate2_pass:
        print(f"✅ GATE 2: PASS - 1-hour timeframe enables profitable trading!")
        print(f"\n🎉 BREAKTHROUGH:")
        print(f"   - Win rate: {best['win_rate_pct']:.1f}% (≥45% threshold)")
        print(f"   - Return: {best['return_pct']:+.2f}% (profitable)")
        print(f"   - 12× noise reduction hypothesis VALIDATED")
        print(f"\n➡️  NEXT: Proceed to Phase 16D (cross-asset validation on 1-hour)")
    else:
        print(f"❌ GATE 2: FAIL - 1-hour timeframe insufficient")
        print(f"\n⚠️  CRITICAL FINDING:")
        print(f"   ALL intraday crypto timeframes fail:")
        print(f"   • 5-minute:  Best 40.3% win rate, -100% return")
        print(f"   • 15-minute: Best 38.7% win rate, -99.97% return")
        print(f"   • 1-hour:    Best {best['win_rate_pct']:.1f}% win rate, {best['return_pct']:+.2f}% return")
        print(f"\n   Failed criteria:")
        for passed, msg in gate2_criteria:
            if not passed:
                print(f"   • {msg}")
        print(f"\n📊 RESEARCH SUMMARY:")
        print(f"   - Total phases: 11 (Phases 8-15A on 5-min, 16A-16B on higher TF)")
        print(f"   - Strategies tested: 17 variations")
        print(f"   - Timeframes tested: 5-min, 15-min, 1-hour")
        print(f"   - Viable strategies: 0 / 17 (0%)")
        print(f"\n🚨 CONCLUSION:")
        print(f"   Crypto directional trading is FUNDAMENTALLY UNSUITABLE")
        print(f"   across ALL intraday timeframes (5-min to 1-hour).")
        print(f"\n➡️  NEXT: Option B (test traditional markets) or ABANDON directional trading")

    # Save results
    results_dir = Path('/Users/terryli/eon/backtesting.py/user_strategies/research/timeframe_analysis/results/phase_16_timeframe_analysis')
    results_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / 'phase_16b_1hour_ma.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Results saved: {results_path}")

    return results_df, gate2_pass


if __name__ == '__main__':
    results, gate2_pass = run_phase_16b_validation()
