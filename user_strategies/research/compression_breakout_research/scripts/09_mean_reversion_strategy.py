#!/usr/bin/env python3
"""
Phase 12A: Mean Reversion Strategy - Core Logic Inversion

HYPOTHESIS: If 70% of breakouts fail (Phase 11), then fading breakouts should yield ~70% win rate.

CHANGES FROM PHASE 10D:
- Entry: INVERTED (sell on upside breakout, buy on downside breakout)
- Exit: INVERTED (target range midpoint, stop on breakout continuation)
- Regime filtering: DISABLED (test baseline first)

VERSION: 1.0.0
DATE: 2025-10-04
"""

import pandas as pd
import numpy as np
from pathlib import Path
from backtesting import Strategy
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


class MeanReversionRegimeStrategy(Strategy):
    """
    Mean reversion strategy: Fade volatility compression breakouts.

    ENTRY LOGIC (INVERTED from Phase 10D):
    - Sell when price breaks above 20-period high (fade upside breakout)
    - Buy when price breaks below 20-period low (fade downside breakout)

    EXIT LOGIC (INVERTED):
    - Stop: Breakout continuation (price moves further in breakout direction)
    - Target: Return to range (price reverts toward compression zone)

    FILTERS (SAME as Phase 10D):
    - Low volatility: ATR percentile < 10% on all timeframes (5m, 15m, 30m)
    - Minimum data: 200+ bars for indicator warmup
    """

    # Base strategy parameters (from Phase 10D)
    atr_period = 14
    percentile_window = 150
    volatility_threshold = 0.10
    breakout_period = 20

    # Mean reversion parameters (MODIFIED)
    stop_atr_multiple = 2.0      # Stop on breakout continuation
    target_atr_multiple = 4.0    # Target range reversion
    max_hold_bars = 100          # Same as Phase 10D
    risk_per_trade = 0.02        # Same as Phase 10D

    # Regime parameters (DISABLED for baseline test)
    regime_enabled = False

    def init(self):
        """Initialize indicators (reused from Phase 10D)."""

        # Validate data integrity
        if self.data.df.isnull().sum().sum() > 0:
            raise ValueError(f"NaN values in input data: {self.data.df.isnull().sum()}")

        if len(self.data.df) < 200:
            raise ValueError(f"Insufficient data: {len(self.data.df)} bars (need ≥200)")

        df_5m = self.data.df.copy()
        atr_5m = calculate_atr(df_5m, self.atr_period)

        # Multi-timeframe ATR
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

        # Low volatility filter
        self.low_vol_filter = self.I(
            lambda: (
                (atr_5m_pct < self.volatility_threshold) &
                (atr_15m_aligned < self.volatility_threshold) &
                (atr_30m_aligned < self.volatility_threshold)
            ).astype(int),
            name='LowVolFilter'
        )

        # Breakout bands
        self.breakout_high = self.I(
            lambda: df_5m['High'].rolling(self.breakout_period).max(),
            name='BreakoutHigh'
        )
        self.breakout_low = self.I(
            lambda: df_5m['Low'].rolling(self.breakout_period).min(),
            name='BreakoutLow'
        )

        self.atr = self.I(lambda: atr_5m, name='ATR')

        # Position tracking
        self.entry_bar = None
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None

    def next(self):
        """Execute mean reversion strategy."""

        # Exit management
        if self.position:
            bars_held = len(self.data) - self.entry_bar
            current_price = self.data.Close[-1]

            # Exit conditions
            should_exit = False
            if bars_held >= self.max_hold_bars:
                should_exit = True  # Timeout
            elif self.position.is_long and self.data.Low[-1] <= self.stop_loss:
                should_exit = True  # Stop hit
            elif self.position.is_short and self.data.High[-1] >= self.stop_loss:
                should_exit = True  # Stop hit
            elif self.position.is_long and self.data.High[-1] >= self.take_profit:
                should_exit = True  # Target hit
            elif self.position.is_short and self.data.Low[-1] <= self.take_profit:
                should_exit = True  # Target hit

            if should_exit:
                self.position.close()
                # Reset tracking
                self.entry_bar = None
                self.entry_price = None
                self.stop_loss = None
                self.take_profit = None
                return

        # Entry logic (MEAN REVERSION - INVERTED)
        if not self._check_entry_conditions():
            return

        current_atr = self.atr[-1]

        # Validate ATR
        if pd.isna(current_atr) or current_atr == 0:
            raise RuntimeError(f"Invalid ATR at bar {len(self.data)}: {current_atr}")

        current_price = self.data.Close[-1]
        prev_high = self.breakout_high[-2]
        prev_low = self.breakout_low[-2]

        # INVERTED ENTRY: Fade breakouts (sell high, buy low)
        if current_price > prev_high:
            # Upside breakout → SELL (expect reversion down)
            stop_distance = self.stop_atr_multiple * current_atr
            position_fraction = self.risk_per_trade * current_price / stop_distance
            position_fraction = max(0.01, min(0.95, position_fraction))

            if position_fraction <= 0:
                raise AssertionError(f"Invalid position size: {position_fraction}")

            self.sell(size=position_fraction)
            self.entry_bar = len(self.data)
            self.entry_price = current_price
            self.stop_loss = current_price + stop_distance      # Stop on continuation UP
            self.take_profit = current_price - (self.target_atr_multiple * current_atr)  # Target reversion DOWN

        elif current_price < prev_low:
            # Downside breakout → BUY (expect reversion up)
            stop_distance = self.stop_atr_multiple * current_atr
            position_fraction = self.risk_per_trade * current_price / stop_distance
            position_fraction = max(0.01, min(0.95, position_fraction))

            if position_fraction <= 0:
                raise AssertionError(f"Invalid position size: {position_fraction}")

            self.buy(size=position_fraction)
            self.entry_bar = len(self.data)
            self.entry_price = current_price
            self.stop_loss = current_price - stop_distance      # Stop on continuation DOWN
            self.take_profit = current_price + (self.target_atr_multiple * current_atr)  # Target reversion UP

    def _check_entry_conditions(self) -> bool:
        """Check if entry conditions are met (reused from Phase 10D)."""

        # Filter 1: Low volatility
        if self.low_vol_filter[-1] != 1:
            return False

        # Filter 2: Minimum data
        if len(self.data) < max(self.percentile_window, self.breakout_period) + 50:
            return False

        # Filter 3: Valid ATR
        current_atr = self.atr[-1]
        if pd.isna(current_atr) or current_atr == 0:
            return False

        # Filter 4: Breakout signal
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
# VALIDATION FUNCTIONS
# ============================================================================

def compare_strategies(df: pd.DataFrame, symbol: str):
    """Compare Phase 10D breakout vs Phase 12A mean reversion."""
    from backtesting import Backtest

    # Import Phase 10D strategy for comparison - need to import directly
    # Copy the class definition inline to avoid module dependencies
    class BreakoutStrategy(Strategy):
        """Phase 10D breakout strategy (inline copy for comparison)."""
        atr_period = 14
        percentile_window = 150
        volatility_threshold = 0.10
        breakout_period = 20
        stop_atr_multiple = 2.0
        target_atr_multiple = 4.0
        max_hold_bars = 100
        risk_per_trade = 0.02
        regime_enabled = False  # Not used, but required by backtesting.py

        def init(self):
            df_5m = self.data.df.copy()
            atr_5m = calculate_atr(df_5m, self.atr_period)

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

            atr_5m_pct = calculate_percentile_rank(atr_5m, self.percentile_window)
            atr_15m_pct = calculate_percentile_rank(atr_15m, self.percentile_window)
            atr_30m_pct = calculate_percentile_rank(atr_30m, self.percentile_window)

            atr_15m_aligned = atr_15m_pct.reindex(df_5m.index, method='ffill')
            atr_30m_aligned = atr_30m_pct.reindex(df_5m.index, method='ffill')

            self.low_vol_filter = self.I(
                lambda: (
                    (atr_5m_pct < self.volatility_threshold) &
                    (atr_15m_aligned < self.volatility_threshold) &
                    (atr_30m_aligned < self.volatility_threshold)
                ).astype(int),
                name='LowVolFilter'
            )

            self.breakout_high = self.I(
                lambda: df_5m['High'].rolling(self.breakout_period).max(),
                name='BreakoutHigh'
            )
            self.breakout_low = self.I(
                lambda: df_5m['Low'].rolling(self.breakout_period).min(),
                name='BreakoutLow'
            )

            self.atr = self.I(lambda: atr_5m, name='ATR')
            self.entry_bar = None
            self.entry_price = None
            self.stop_loss = None
            self.take_profit = None

        def next(self):
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

            if not self._check_entry_conditions():
                return

            current_atr = self.atr[-1]
            current_price = self.data.Close[-1]
            prev_high = self.breakout_high[-2]
            prev_low = self.breakout_low[-2]

            # ORIGINAL BREAKOUT LOGIC
            if current_price > prev_high:
                stop_distance = self.stop_atr_multiple * current_atr
                position_fraction = self.risk_per_trade * current_price / stop_distance
                position_fraction = max(0.01, min(0.95, position_fraction))

                self.buy(size=position_fraction)
                self.entry_bar = len(self.data)
                self.entry_price = current_price
                self.stop_loss = current_price - stop_distance
                self.take_profit = current_price + (self.target_atr_multiple * current_atr)

            elif current_price < prev_low:
                stop_distance = self.stop_atr_multiple * current_atr
                position_fraction = self.risk_per_trade * current_price / stop_distance
                position_fraction = max(0.01, min(0.95, position_fraction))

                self.sell(size=position_fraction)
                self.entry_bar = len(self.data)
                self.entry_price = current_price
                self.stop_loss = current_price + stop_distance
                self.take_profit = current_price - (self.target_atr_multiple * current_atr)

        def _check_entry_conditions(self):
            if self.low_vol_filter[-1] != 1:
                return False
            if len(self.data) < max(self.percentile_window, self.breakout_period) + 50:
                return False
            current_atr = self.atr[-1]
            if pd.isna(current_atr) or current_atr == 0:
                return False
            current_price = self.data.Close[-1]
            prev_high = self.breakout_high[-2]
            prev_low = self.breakout_low[-2]
            if current_price > prev_high or current_price < prev_low:
                return True
            return False

    print(f"\n{'='*70}")
    print(f"{symbol} - BREAKOUT vs MEAN REVERSION COMPARISON")
    print(f"{'='*70}")
    print(f"Data: {len(df):,} bars ({df.index[0].date()} to {df.index[-1].date()})")

    # Breakout (Phase 10D baseline)
    print(f"\nRunning Phase 10D breakout strategy...")
    bt_breakout = Backtest(df, BreakoutStrategy, cash=10_000_000, commission=0.0002, margin=0.05, exclusive_orders=True)
    stats_breakout = bt_breakout.run(regime_enabled=False)

    # Mean Reversion (Phase 12A)
    print(f"Running Phase 12A mean reversion strategy...")
    bt_reversion = Backtest(df, MeanReversionRegimeStrategy, cash=10_000_000, commission=0.0002, margin=0.05, exclusive_orders=True)
    stats_reversion = bt_reversion.run()

    # Results
    print(f"\n{'='*70}")
    print("RESULTS COMPARISON")
    print(f"{'='*70}")
    print(f"\nBreakout (Phase 10D):")
    print(f"  Return:   {stats_breakout['Return [%]']:+.2f}%")
    print(f"  Trades:   {stats_breakout['# Trades']}")
    print(f"  Win Rate: {stats_breakout['Win Rate [%]']:.1f}%")
    print(f"  Sharpe:   {stats_breakout['Sharpe Ratio']:+.2f}")

    print(f"\nMean Reversion (Phase 12A):")
    print(f"  Return:   {stats_reversion['Return [%]']:+.2f}%")
    print(f"  Trades:   {stats_reversion['# Trades']}")
    print(f"  Win Rate: {stats_reversion['Win Rate [%]']:.1f}%")
    print(f"  Sharpe:   {stats_reversion['Sharpe Ratio']:+.2f}")

    print(f"\nImprovement:")
    print(f"  Return:   {stats_reversion['Return [%]'] - stats_breakout['Return [%]']:+.2f}pp")
    print(f"  Win Rate: {stats_reversion['Win Rate [%]'] - stats_breakout['Win Rate [%]']:+.1f}pp")
    print(f"  Trades:   {stats_reversion['# Trades'] - stats_breakout['# Trades']:+d}")

    return {
        'symbol': symbol,
        'breakout_return': stats_breakout['Return [%]'],
        'breakout_trades': stats_breakout['# Trades'],
        'breakout_win_rate': stats_breakout['Win Rate [%]'],
        'reversion_return': stats_reversion['Return [%]'],
        'reversion_trades': stats_reversion['# Trades'],
        'reversion_win_rate': stats_reversion['Win Rate [%]'],
        'return_improvement': stats_reversion['Return [%]'] - stats_breakout['Return [%]'],
        'win_rate_improvement': stats_reversion['Win Rate [%]'] - stats_breakout['Win Rate [%]'],
    }


if __name__ == '__main__':
    """Quick validation on ETH."""

    data_dir = Path('/Users/terryli/eon/backtesting.py/user_strategies/data/raw/crypto_5m')
    csv_path = data_dir / 'binance_spot_ETHUSDT-5m_20220101-20250930_v2.10.0.csv'

    print("Loading ETH data...")
    df = load_5m_data(csv_path, n_bars=None)

    results = compare_strategies(df, 'ETH')

    # Gate 1 check
    print(f"\n{'='*70}")
    print("GATE 1 CHECK")
    print(f"{'='*70}")

    gate1_pass = (
        results['reversion_return'] > -50 and
        results['reversion_win_rate'] >= 50 and
        results['reversion_trades'] >= 50
    )

    print(f"Return > -50%:     {results['reversion_return']:+.2f}% - {'✅ PASS' if results['reversion_return'] > -50 else '❌ FAIL'}")
    print(f"Win Rate ≥ 50%:   {results['reversion_win_rate']:.1f}% - {'✅ PASS' if results['reversion_win_rate'] >= 50 else '❌ FAIL'}")
    print(f"Trades ≥ 50:      {results['reversion_trades']} - {'✅ PASS' if results['reversion_trades'] >= 50 else '❌ FAIL'}")

    if gate1_pass:
        print(f"\n✅ GATE 1: PASS - Proceed to Phase 12C (cross-asset validation)")
    else:
        print(f"\n❌ GATE 1: FAIL - Escalate to user for decision")
        raise RuntimeError("GATE 1 FAIL: Mean reversion strategy does not meet minimum criteria")
