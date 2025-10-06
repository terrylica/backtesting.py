#!/usr/bin/env python3
"""
Phase 14A: Dual Moving Average Crossover - Baseline

THEORETICAL FOUNDATION:
- Edwards & Magee (1948): Technical Analysis of Stock Trends
- Dennis & Eckhardt (1983): Turtle Trading System
- Jegadeesh & Titman (1993): Momentum and Reversal

STRATEGY:
- Entry: 50-MA crosses 200-MA
- Exit: Opposite crossover, stop loss, or trailing stop
- Position: Full size (95% of capital)

EXPECTED PERFORMANCE:
- Win rate: 40-45% (trend following baseline)
- Profit factor: >1.5
- Sharpe: >0.5

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
    """
    Phase 14A: Dual Moving Average Crossover (Trend Following).

    ENTRY:
    - Long: Fast MA (50) crosses above Slow MA (200)
    - Short: Fast MA (50) crosses below Slow MA (200)

    EXIT:
    - Opposite crossover signal
    - Stop loss: 2.0 × ATR from entry
    - Trailing stop: 3.0 × ATR from highest high (long) / lowest low (short)
    - Max hold: 500 bars (100 hours @ 5-min)

    POSITION SIZING:
    - Full position: 95% of capital
    """

    # Strategy parameters
    ma_fast_period = 50
    ma_slow_period = 200
    atr_period = 14
    stop_atr_multiple = 2.0
    trailing_atr_multiple = 3.0
    max_hold_bars = 500

    def init(self):
        """Initialize indicators."""

        # Validate data integrity
        if self.data.df.isnull().sum().sum() > 0:
            raise ValueError(f"NaN values in input data: {self.data.df.isnull().sum()}")

        if len(self.data.df) < 250:
            raise ValueError(f"Insufficient data: {len(self.data.df)} bars (need ≥250 for 200-MA)")

        df = self.data.df.copy()

        # Moving averages (out-of-the-box pandas)
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
        self.highest_high = None  # For long trailing stop
        self.lowest_low = None    # For short trailing stop

    def next(self):
        """Execute strategy logic."""

        # Exit management
        if self.position:
            self._manage_exits()
            return  # Don't check entries if in position

        # Entry conditions
        self._check_entries()

    def _check_entries(self):
        """Check for entry signals."""

        # Require minimum data for indicators
        if len(self.data) < max(self.ma_fast_period, self.ma_slow_period) + 10:
            return

        # Validate indicators
        if pd.isna(self.ma_fast[-1]) or pd.isna(self.ma_slow[-1]):
            raise RuntimeError(
                f"NaN MA at bar {len(self.data)}: "
                f"fast={self.ma_fast[-1]}, slow={self.ma_slow[-1]}"
            )

        if pd.isna(self.atr[-1]) or self.atr[-1] <= 0:
            raise RuntimeError(f"Invalid ATR at bar {len(self.data)}: {self.atr[-1]}")

        # Crossover detection (require [-2] and [-1] for crossover)
        if len(self.data) < 2:
            return

        ma_fast_prev = self.ma_fast[-2]
        ma_slow_prev = self.ma_slow[-2]
        ma_fast_curr = self.ma_fast[-1]
        ma_slow_curr = self.ma_slow[-1]

        current_price = self.data.Close[-1]
        current_atr = self.atr[-1]

        # LONG ENTRY: Fast crosses above slow
        if ma_fast_prev <= ma_slow_prev and ma_fast_curr > ma_slow_curr:
            self.buy(size=0.95)  # Full position
            self.entry_bar = len(self.data)
            self.entry_price = current_price
            self.stop_loss = current_price - (self.stop_atr_multiple * current_atr)
            self.highest_high = current_price  # Initialize trailing stop tracker

        # SHORT ENTRY: Fast crosses below slow
        elif ma_fast_prev >= ma_slow_prev and ma_fast_curr < ma_slow_curr:
            self.sell(size=0.95)  # Full position
            self.entry_bar = len(self.data)
            self.entry_price = current_price
            self.stop_loss = current_price + (self.stop_atr_multiple * current_atr)
            self.lowest_low = current_price  # Initialize trailing stop tracker

    def _manage_exits(self):
        """Manage position exits."""

        bars_held = len(self.data) - self.entry_bar
        current_price = self.data.Close[-1]
        current_atr = self.atr[-1]

        # Validate ATR
        if pd.isna(current_atr) or current_atr <= 0:
            raise RuntimeError(f"Invalid ATR during exit at bar {len(self.data)}: {current_atr}")

        # Exit condition 1: Max hold time
        if bars_held >= self.max_hold_bars:
            self.position.close()
            self._reset_position_tracking()
            return

        # Exit condition 2: Opposite crossover
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

        # Exit condition 3: Stop loss
        if self.position.is_long and self.data.Low[-1] <= self.stop_loss:
            self.position.close()
            self._reset_position_tracking()
            return

        if self.position.is_short and self.data.High[-1] >= self.stop_loss:
            self.position.close()
            self._reset_position_tracking()
            return

        # Exit condition 4: Trailing stop
        if self.position.is_long:
            # Update highest high
            if self.data.High[-1] > self.highest_high:
                self.highest_high = self.data.High[-1]

            # Calculate trailing stop
            trailing_stop = self.highest_high - (self.trailing_atr_multiple * current_atr)

            # Use the higher of initial stop and trailing stop
            self.stop_loss = max(self.stop_loss, trailing_stop)

            # Check if trailing stop hit
            if self.data.Low[-1] <= trailing_stop:
                self.position.close()
                self._reset_position_tracking()
                return

        if self.position.is_short:
            # Update lowest low
            if self.data.Low[-1] < self.lowest_low:
                self.lowest_low = self.data.Low[-1]

            # Calculate trailing stop
            trailing_stop = self.lowest_low + (self.trailing_atr_multiple * current_atr)

            # Use the lower of initial stop and trailing stop
            self.stop_loss = min(self.stop_loss, trailing_stop)

            # Check if trailing stop hit
            if self.data.High[-1] >= trailing_stop:
                self.position.close()
                self._reset_position_tracking()
                return

    def _reset_position_tracking(self):
        """Reset position tracking variables after exit."""
        self.entry_bar = None
        self.entry_price = None
        self.stop_loss = None
        self.highest_high = None
        self.lowest_low = None


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
# PHASE 14A VALIDATION
# ============================================================================

def run_phase_14a_validation():
    """Run Phase 14A validation with Gate 1 check."""

    data_dir = Path('/Users/terryli/eon/backtesting.py/user_strategies/data/raw/crypto_5m')
    csv_path = data_dir / 'binance_spot_ETHUSDT-5m_20220101-20250930_v2.10.0.csv'

    print("="*70)
    print("PHASE 14A: DUAL MA CROSSOVER - BASELINE VALIDATION")
    print("="*70)

    print("\nLoading ETH data...")
    df = load_5m_data(csv_path, n_bars=None)
    print(f"Loaded {len(df):,} bars ({df.index[0].date()} to {df.index[-1].date()})")

    # Run dual MA crossover strategy
    print("\nRunning Phase 14A dual MA crossover (50/200)...")
    bt = Backtest(df, DualMACrossoverStrategy, cash=10_000_000, commission=0.0002, margin=0.05, exclusive_orders=True)
    stats = bt.run()

    print(f"\n{'='*70}")
    print("PHASE 14A RESULTS")
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
    return_pct = stats['Return [%]']
    n_trades = stats['# Trades']
    sharpe = stats['Sharpe Ratio']

    gate1_criteria = [
        (win_rate >= 35, f"Win Rate ≥ 35%:  {win_rate:.1f}%"),
        (return_pct > 0, f"Return > 0%:     {return_pct:+.2f}%"),
        (n_trades >= 10, f"Trades ≥ 10:     {n_trades}"),
        (sharpe > 0, f"Sharpe > 0.0:    {sharpe:+.2f}")
    ]

    gate1_pass = all(criterion[0] for criterion in gate1_criteria)

    for passed, message in gate1_criteria:
        status = '✅ PASS' if passed else '❌ FAIL'
        print(f"{message} - {status}")

    if gate1_pass:
        print(f"\n✅ GATE 1: PASS - Proceed to Phase 14B (add confirmation filters)")
        if return_pct > 5:
            print(f"   Strong result: {return_pct:+.2f}% return exceeds target (>5%)")
    else:
        print(f"\n❌ GATE 1: FAIL - Adjust MA periods and re-validate")
        failed_criteria = [msg for passed, msg in gate1_criteria if not passed]
        for msg in failed_criteria:
            print(f"   {msg}")
        raise RuntimeError("GATE 1 FAIL: Dual MA crossover baseline does not meet minimum criteria")

    # Save results
    results_dir = Path('/Users/terryli/eon/backtesting.py/user_strategies/research/proven_strategies/results/phase_14_trend_following')
    results_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame([{
        'phase': '14A',
        'strategy': 'dual_ma_baseline',
        'ma_fast': 50,
        'ma_slow': 200,
        'return_pct': stats['Return [%]'],
        'n_trades': stats['# Trades'],
        'win_rate_pct': stats['Win Rate [%]'],
        'sharpe': stats['Sharpe Ratio'],
        'max_dd_pct': stats['Max. Drawdown [%]'],
    }])

    results_path = results_dir / 'phase_14a_baseline.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Results saved: {results_path}")

    return stats


if __name__ == '__main__':
    stats = run_phase_14a_validation()
