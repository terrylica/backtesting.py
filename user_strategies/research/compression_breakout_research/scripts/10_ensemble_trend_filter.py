#!/usr/bin/env python3
"""
Phase 13A: Ensemble Strategy - Trend Filter

CHANGES FROM PHASE 10D:
- Add Filter 1: Trend Alignment (50-SMA slope)
- Entry logic: ORIGINAL Phase 10D (buy high, sell low)
- Additional filter: Only trade in direction of trend

HYPOTHESIS: Filtering counter-trend breakouts will improve win rate from 36.3% to >40%.

VERSION: 1.0.0
DATE: 2025-10-04
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


def calculate_percentile_rank(series: pd.Series, window: int = 150) -> pd.Series:
    """Calculate rolling percentile rank 0-1 (reused from Phase 10D)."""
    def percentile_rank(x):
        if len(x) < 2:
            return np.nan
        current_value = x.iloc[-1]
        return (x < current_value).sum() / len(x)

    return series.rolling(window).apply(percentile_rank, raw=False)


class EnsembleTrendFilterStrategy(Strategy):
    """
    Phase 13A: Breakout strategy with trend filter.

    FILTERS:
    1. Volatility Compression: ATR percentile < 10% (Phase 10D)
    2. Trend Alignment: Only trade in direction of 50-SMA slope (NEW)

    ENTRY:
    - Buy: Price > 20-period high AND 50-SMA slope > 0 (uptrend)
    - Sell: Price < 20-period low AND 50-SMA slope < 0 (downtrend)

    EXIT:
    - Stop: 2.0 × ATR
    - Target: 4.0 × ATR
    - Timeout: 100 bars
    """

    # Base strategy parameters (from Phase 10D)
    atr_period = 14
    percentile_window = 150
    volatility_threshold = 0.10
    breakout_period = 20
    stop_atr_multiple = 2.0
    target_atr_multiple = 4.0
    max_hold_bars = 100
    risk_per_trade = 0.02

    # Trend filter parameters (NEW)
    sma_period = 50

    def init(self):
        """Initialize indicators with trend filter."""

        # Validate data integrity
        if self.data.df.isnull().sum().sum() > 0:
            raise ValueError(f"NaN values in input data: {self.data.df.isnull().sum()}")

        if len(self.data.df) < 200:
            raise ValueError(f"Insufficient data: {len(self.data.df)} bars (need ≥200)")

        df_5m = self.data.df.copy()
        atr_5m = calculate_atr(df_5m, self.atr_period)

        # Multi-timeframe ATR (from Phase 10D)
        df_15m = df_5m.resample('15min').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min',
            'Close': 'last', 'Volume': 'sum'
        }).dropna()
        atr_15m = calculate_atr(df_15m, self.atr_period)

        df_30m = df_5m.resample('30min').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min',
            'Close': 'last', 'Volume': 'sum'
        }).dropna()
        atr_30m = calculate_atr(df_30m, self.atr_period)

        # Percentile ranks
        atr_5m_pct = calculate_percentile_rank(atr_5m, self.percentile_window)
        atr_15m_pct = calculate_percentile_rank(atr_15m, self.percentile_window)
        atr_30m_pct = calculate_percentile_rank(atr_30m, self.percentile_window)

        # Align to 5m bars
        atr_15m_aligned = atr_15m_pct.reindex(df_5m.index, method='ffill')
        atr_30m_aligned = atr_30m_pct.reindex(df_5m.index, method='ffill')

        # Low volatility filter (Phase 10D)
        self.low_vol_filter = self.I(
            lambda: (
                (atr_5m_pct < self.volatility_threshold) &
                (atr_15m_aligned < self.volatility_threshold) &
                (atr_30m_aligned < self.volatility_threshold)
            ).astype(int),
            name='LowVolFilter'
        )

        # Breakout bands (Phase 10D)
        self.breakout_high = self.I(
            lambda: df_5m['High'].rolling(self.breakout_period).max(),
            name='BreakoutHigh'
        )
        self.breakout_low = self.I(
            lambda: df_5m['Low'].rolling(self.breakout_period).min(),
            name='BreakoutLow'
        )

        self.atr = self.I(lambda: atr_5m, name='ATR')

        # TREND FILTER (NEW - Phase 13A)
        self.sma_50 = self.I(
            lambda: df_5m['Close'].rolling(self.sma_period).mean(),
            name='SMA50'
        )

        # Position tracking
        self.entry_bar = None
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None

    def next(self):
        """Execute strategy with trend filter."""

        # Exit management (unchanged from Phase 10D)
        if self.position:
            bars_held = len(self.data) - self.entry_bar
            current_price = self.data.Close[-1]

            should_exit = False
            if bars_held >= self.max_hold_bars:
                should_exit = True
            elif self.position.is_long and self.data.Low[-1] <= self.stop_loss:
                should_exit = True
            elif self.position.is_short and self.data.High[-1] >= self.stop_loss:
                should_exit = True
            elif self.position.is_long and self.data.High[-1] >= self.take_profit:
                should_exit = True
            elif self.position.is_short and self.data.Low[-1] <= self.take_profit:
                should_exit = True

            if should_exit:
                self.position.close()
                self.entry_bar = None
                self.entry_price = None
                self.stop_loss = None
                self.take_profit = None
                return

        # Entry logic with TREND FILTER
        if not self._check_entry_conditions():
            return

        current_atr = self.atr[-1]

        # Validate ATR
        if pd.isna(current_atr) or current_atr == 0:
            raise RuntimeError(f"Invalid ATR at bar {len(self.data)}: {current_atr}")

        # Validate SMA
        if pd.isna(self.sma_50[-1]):
            raise RuntimeError(f"NaN SMA at bar {len(self.data)}")

        # Calculate trend slope
        if len(self.data) < 2:
            return  # Need at least 2 bars for slope

        sma_slope = self.sma_50[-1] - self.sma_50[-2]

        current_price = self.data.Close[-1]
        prev_high = self.breakout_high[-2]
        prev_low = self.breakout_low[-2]

        # ORIGINAL BREAKOUT LOGIC (Phase 10D) with TREND FILTER
        if current_price > prev_high:
            # Upside breakout: Only enter if in uptrend
            if sma_slope <= 0:
                return  # FILTER: Block counter-trend long

            stop_distance = self.stop_atr_multiple * current_atr
            position_fraction = self.risk_per_trade * current_price / stop_distance
            position_fraction = max(0.01, min(0.95, position_fraction))

            if position_fraction <= 0:
                raise AssertionError(f"Invalid position size: {position_fraction}")

            self.buy(size=position_fraction)
            self.entry_bar = len(self.data)
            self.entry_price = current_price
            self.stop_loss = current_price - stop_distance
            self.take_profit = current_price + (self.target_atr_multiple * current_atr)

        elif current_price < prev_low:
            # Downside breakout: Only enter if in downtrend
            if sma_slope >= 0:
                return  # FILTER: Block counter-trend short

            stop_distance = self.stop_atr_multiple * current_atr
            position_fraction = self.risk_per_trade * current_price / stop_distance
            position_fraction = max(0.01, min(0.95, position_fraction))

            if position_fraction <= 0:
                raise AssertionError(f"Invalid position size: {position_fraction}")

            self.sell(size=position_fraction)
            self.entry_bar = len(self.data)
            self.entry_price = current_price
            self.stop_loss = current_price + stop_distance
            self.take_profit = current_price - (self.target_atr_multiple * current_atr)

    def _check_entry_conditions(self) -> bool:
        """Check if entry conditions are met (reused from Phase 10D)."""

        # Filter: Low volatility
        if self.low_vol_filter[-1] != 1:
            return False

        # Filter: Minimum data
        if len(self.data) < max(self.percentile_window, self.breakout_period, self.sma_period) + 50:
            return False

        # Filter: Valid ATR
        current_atr = self.atr[-1]
        if pd.isna(current_atr) or current_atr == 0:
            return False

        # Filter: Breakout signal
        current_price = self.data.Close[-1]
        prev_high = self.breakout_high[-2]
        prev_low = self.breakout_low[-2]

        if current_price > prev_high or current_price < prev_low:
            return True

        return False


def load_5m_data(csv_path: Path, n_bars: int = None) -> pd.DataFrame:
    """Load 5-minute crypto data (reused from Phase 10D)."""
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
# PHASE 13A VALIDATION
# ============================================================================

def run_phase_13a_validation():
    """Run Phase 13A validation with Gate 1 check."""

    data_dir = Path('/Users/terryli/eon/backtesting.py/user_strategies/data/raw/crypto_5m')
    csv_path = data_dir / 'binance_spot_ETHUSDT-5m_20220101-20250930_v2.10.0.csv'

    print("="*70)
    print("PHASE 13A: TREND FILTER VALIDATION")
    print("="*70)

    print("\nLoading ETH data...")
    df = load_5m_data(csv_path, n_bars=None)
    print(f"Loaded {len(df):,} bars ({df.index[0].date()} to {df.index[-1].date()})")

    # Run ensemble strategy
    print("\nRunning Phase 13A ensemble (compression + trend filter)...")
    bt = Backtest(df, EnsembleTrendFilterStrategy, cash=10_000_000, commission=0.0002, margin=0.05, exclusive_orders=True)
    stats = bt.run()

    print(f"\n{'='*70}")
    print("PHASE 13A RESULTS")
    print(f"{'='*70}")
    print(f"Return:   {stats['Return [%]']:+.2f}%")
    print(f"Trades:   {stats['# Trades']}")
    print(f"Win Rate: {stats['Win Rate [%]']:.1f}%")
    print(f"Sharpe:   {stats['Sharpe Ratio']:+.2f}")
    print(f"Max DD:   {stats['Max. Drawdown [%]']:.2f}%")

    # Gate 1 check
    print(f"\n{'='*70}")
    print("GATE 1 CHECK")
    print(f"{'='*70}")

    win_rate = stats['Win Rate [%]']
    n_trades = stats['# Trades']

    gate1_pass = (win_rate > 40) and (n_trades >= 100)

    print(f"Win Rate > 40%:  {win_rate:.1f}% - {'✅ PASS' if win_rate > 40 else '❌ FAIL'}")
    print(f"Trades ≥ 100:    {n_trades} - {'✅ PASS' if n_trades >= 100 else '❌ FAIL'}")

    if gate1_pass:
        print(f"\n✅ GATE 1: PASS - Proceed to Phase 13B (add volume filter)")
        print(f"   Improvement: {win_rate - 36.3:+.1f}pp vs Phase 10D baseline (36.3%)")
    else:
        print(f"\n❌ GATE 1: FAIL - Abort Phase 13, recommend proven strategies")
        if win_rate <= 40:
            print(f"   Win rate {win_rate:.1f}% did not improve sufficiently (need >40%)")
        if n_trades < 100:
            print(f"   Insufficient trades {n_trades} (need ≥100 for statistical validity)")
        raise RuntimeError("GATE 1 FAIL: Trend filter insufficient, abort Phase 13")

    # Save results
    results_dir = Path('/Users/terryli/eon/backtesting.py/user_strategies/research/compression_breakout_research/results/phase_13_ensemble')
    results_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame([{
        'phase': '13A',
        'filters': 'compression+trend',
        'return_pct': stats['Return [%]'],
        'n_trades': stats['# Trades'],
        'win_rate_pct': stats['Win Rate [%]'],
        'sharpe': stats['Sharpe Ratio'],
        'max_dd_pct': stats['Max. Drawdown [%]'],
    }])

    results_path = results_dir / 'phase_13a_trend_filter.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Results saved: {results_path}")

    return stats


if __name__ == '__main__':
    stats = run_phase_13a_validation()
