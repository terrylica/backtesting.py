#!/usr/bin/env python3
"""
Phase 15A: Bollinger Band Mean Reversion - Baseline

THEORETICAL FOUNDATION:
- John Bollinger (2001): Bollinger on Bollinger Bands
- J. Welles Wilder (1978): New Concepts in Technical Trading Systems

STRATEGY:
- Entry: Price touches BB ±2σ + RSI <30 or >70
- Exit: Mean reversion to BB middle, stop loss, or time stop
- Position: Full size (95% of capital)

HYPOTHESIS:
- Crypto 5-minute markets exhibit strong mean reversion from extremes
- Expected: Win rate >50%, positive returns

HARD STOP CRITERIA:
- If win rate <50% OR return ≤0%: ABANDON crypto 5-minute trading
- This is the FINAL TEST after 9 prior failed phases

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


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate RSI using standard Wilder smoothing.

    Formula:
    - RSI = 100 - (100 / (1 + RS))
    - RS = Average Gain / Average Loss
    - Average Gain/Loss: Wilder smoothing (EMA with alpha = 1/period)
    """
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Wilder smoothing: EMA with alpha = 1/period
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

    # Avoid division by zero
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    return rsi


class BollingerMeanReversionStrategy(Strategy):
    """
    Phase 15A: Bollinger Band Mean Reversion.

    ENTRY:
    - Long: Price ≤ BB lower + RSI < 30 (oversold)
    - Short: Price ≥ BB upper + RSI > 70 (overbought)

    EXIT:
    - Target: Price reaches BB middle (mean reversion complete)
    - Stop loss: 2.0 × ATR from entry
    - Time stop: 100 bars without mean reversion
    - Opposite signal: New extreme in opposite direction

    POSITION SIZING:
    - Full position: 95% of capital
    """

    # Bollinger Band parameters
    bb_period = 20
    bb_std = 2.0

    # RSI parameters
    rsi_period = 14
    rsi_oversold = 30
    rsi_overbought = 70

    # Risk management
    atr_period = 14
    stop_atr_multiple = 2.0
    max_hold_bars = 100

    def init(self):
        """Initialize indicators."""

        # Validate data integrity
        if self.data.df.isnull().sum().sum() > 0:
            raise ValueError(f"NaN values in input data: {self.data.df.isnull().sum()}")

        min_bars = max(self.bb_period, self.rsi_period, self.atr_period) + 50
        if len(self.data.df) < min_bars:
            raise ValueError(f"Insufficient data: {len(self.data.df)} bars (need ≥{min_bars})")

        df = self.data.df.copy()

        # Bollinger Bands (out-of-the-box pandas)
        self.bb_middle = self.I(
            lambda: df['Close'].rolling(self.bb_period).mean(),
            name='BB_Middle'
        )
        bb_std_calc = df['Close'].rolling(self.bb_period).std()
        self.bb_upper = self.I(
            lambda: self.bb_middle + (self.bb_std * bb_std_calc),
            name='BB_Upper'
        )
        self.bb_lower = self.I(
            lambda: self.bb_middle - (self.bb_std * bb_std_calc),
            name='BB_Lower'
        )

        # RSI (standard Wilder formula)
        self.rsi = self.I(
            lambda: calculate_rsi(df['Close'], self.rsi_period),
            name='RSI'
        )

        # ATR for stops
        self.atr = self.I(lambda: calculate_atr(df, self.atr_period), name='ATR')

        # Position tracking
        self.entry_bar = None
        self.entry_price = None
        self.stop_loss = None
        self.target_price = None
        self.position_type = None  # 'long' or 'short'

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
        if len(self.data) < max(self.bb_period, self.rsi_period) + 10:
            return

        # Validate indicators
        if pd.isna(self.bb_upper[-1]) or pd.isna(self.bb_lower[-1]) or pd.isna(self.bb_middle[-1]):
            raise RuntimeError(
                f"NaN Bollinger Band at bar {len(self.data)}: "
                f"upper={self.bb_upper[-1]}, lower={self.bb_lower[-1]}, middle={self.bb_middle[-1]}"
            )

        if self.bb_lower[-1] >= self.bb_upper[-1]:
            raise RuntimeError(
                f"Invalid BB at bar {len(self.data)}: "
                f"lower {self.bb_lower[-1]:.2f} >= upper {self.bb_upper[-1]:.2f}"
            )

        if pd.isna(self.rsi[-1]):
            raise RuntimeError(f"NaN RSI at bar {len(self.data)}: {self.rsi[-1]}")

        if pd.isna(self.atr[-1]) or self.atr[-1] <= 0:
            raise RuntimeError(f"Invalid ATR at bar {len(self.data)}: {self.atr[-1]}")

        current_price = self.data.Close[-1]
        current_rsi = self.rsi[-1]
        current_atr = self.atr[-1]

        # LONG ENTRY: Price touches lower BB + RSI oversold
        if (current_price <= self.bb_lower[-1] and
            current_rsi < self.rsi_oversold):

            self.buy(size=0.95)
            self.entry_bar = len(self.data)
            self.entry_price = current_price
            self.stop_loss = current_price - (self.stop_atr_multiple * current_atr)
            self.target_price = self.bb_middle[-1]  # Mean reversion target
            self.position_type = 'long'

        # SHORT ENTRY: Price touches upper BB + RSI overbought
        elif (current_price >= self.bb_upper[-1] and
              current_rsi > self.rsi_overbought):

            self.sell(size=0.95)
            self.entry_bar = len(self.data)
            self.entry_price = current_price
            self.stop_loss = current_price + (self.stop_atr_multiple * current_atr)
            self.target_price = self.bb_middle[-1]  # Mean reversion target
            self.position_type = 'short'

    def _manage_exits(self):
        """Manage position exits."""

        bars_held = len(self.data) - self.entry_bar
        current_price = self.data.Close[-1]
        current_rsi = self.rsi[-1]

        # Validate indicators
        if pd.isna(current_rsi):
            raise RuntimeError(f"NaN RSI during exit at bar {len(self.data)}")

        # Exit condition 1: Time stop
        if bars_held >= self.max_hold_bars:
            self.position.close()
            self._reset_position_tracking()
            return

        # Exit condition 2: Mean reversion target reached
        if self.position.is_long and current_price >= self.target_price:
            self.position.close()
            self._reset_position_tracking()
            return

        if self.position.is_short and current_price <= self.target_price:
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

        # Exit condition 4: Opposite extreme signal (reversal)
        if self.position.is_long:
            # Exit long if price touches upper BB and RSI overbought
            if current_price >= self.bb_upper[-1] and current_rsi > self.rsi_overbought:
                self.position.close()
                self._reset_position_tracking()
                return

        if self.position.is_short:
            # Exit short if price touches lower BB and RSI oversold
            if current_price <= self.bb_lower[-1] and current_rsi < self.rsi_oversold:
                self.position.close()
                self._reset_position_tracking()
                return

    def _reset_position_tracking(self):
        """Reset position tracking variables after exit."""
        self.entry_bar = None
        self.entry_price = None
        self.stop_loss = None
        self.target_price = None
        self.position_type = None


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
# PHASE 15A VALIDATION (WITH HARD STOP)
# ============================================================================

def run_phase_15a_validation():
    """Run Phase 15A validation with HARD STOP criteria."""

    data_dir = Path('/Users/terryli/eon/backtesting.py/user_strategies/data/raw/crypto_5m')
    csv_path = data_dir / 'binance_spot_ETHUSDT-5m_20220101-20250930_v2.10.0.csv'

    print("="*70)
    print("PHASE 15A: BOLLINGER BAND MEAN REVERSION - BASELINE VALIDATION")
    print("="*70)
    print("\n⚠️  FINAL TEST: Hard stop criteria enforced")
    print("   - If win rate <50% OR return ≤0%: ABANDON crypto 5-minute trading")
    print("   - This is phase 10/10 after 9 prior failed phases\n")

    print("Loading ETH data...")
    df = load_5m_data(csv_path, n_bars=None)
    print(f"Loaded {len(df):,} bars ({df.index[0].date()} to {df.index[-1].date()})")

    # Run Bollinger mean reversion strategy
    print("\nRunning Phase 15A Bollinger Band mean reversion...")
    print("  - BB: 20-period SMA, 2.0σ")
    print("  - RSI: 14-period (Wilder smoothing)")
    print("  - Entry: BB touch + RSI <30 or >70")
    print("  - Exit: BB middle, 2ATR stop, or 100-bar time stop\n")

    bt = Backtest(df, BollingerMeanReversionStrategy, cash=10_000_000, commission=0.0002, margin=0.05, exclusive_orders=True)
    stats = bt.run()

    print(f"{'='*70}")
    print("PHASE 15A RESULTS")
    print(f"{'='*70}")
    print(f"Return:   {stats['Return [%]']:+.2f}%")
    print(f"Trades:   {stats['# Trades']}")
    print(f"Win Rate: {stats['Win Rate [%]']:.1f}%")
    print(f"Sharpe:   {stats['Sharpe Ratio']:+.2f}")
    print(f"Max DD:   {stats['Max. Drawdown [%]']:.2f}%")

    # HARD STOP check
    print(f"\n{'='*70}")
    print("HARD STOP CHECK (Gate 1)")
    print(f"{'='*70}")

    win_rate = stats['Win Rate [%]']
    return_pct = stats['Return [%]']
    n_trades = stats['# Trades']
    sharpe = stats['Sharpe Ratio']

    hard_stop_criteria = [
        (win_rate >= 50, f"Win Rate ≥ 50%:  {win_rate:.1f}%"),
        (return_pct > 0, f"Return > 0%:     {return_pct:+.2f}%"),
        (n_trades >= 20, f"Trades ≥ 20:     {n_trades}"),
        (sharpe > 0, f"Sharpe > 0.0:    {sharpe:+.2f}")
    ]

    hard_stop_pass = all(criterion[0] for criterion in hard_stop_criteria)

    for passed, message in hard_stop_criteria:
        status = '✅ PASS' if passed else '❌ FAIL'
        print(f"{message} - {status}")

    if hard_stop_pass:
        print(f"\n✅ HARD STOP: PASS - Mean reversion works on crypto 5-minute!")
        print(f"   Proceed to Phase 15B (add RSI divergence filter)")
        print(f"\n🎉 BREAKTHROUGH: After 9 failed phases, mean reversion from extremes succeeds")
        print(f"   - Win rate: {win_rate:.1f}% (>50% random baseline)")
        print(f"   - Return: {return_pct:+.2f}% (profitable)")
        print(f"   - Crypto 5-minute IS tradeable with mean reversion strategy")
    else:
        print(f"\n❌ HARD STOP: FAIL - ABANDON crypto 5-minute trading")
        print(f"\nFAILED CRITERIA:")
        failed_criteria = [(passed, msg) for passed, msg in hard_stop_criteria if not passed]
        for _, msg in failed_criteria:
            print(f"   {msg}")

        print(f"\nCOMPLETE RESEARCH SUMMARY:")
        print(f"  - Total phases: 10 (Phase 8 through 15A)")
        print(f"  - Strategies tested: 13+ variations")
        print(f"  - Time invested: 25+ hours")
        print(f"  - Best results:")
        print(f"    • Compression + trend: 39.7% win rate, -100% return")
        print(f"    • MA crossover 100/300: 40.3% win rate, -100% return")
        print(f"    • BB mean reversion: {win_rate:.1f}% win rate, {return_pct:+.2f}% return")
        print(f"  - Viable strategies: 0 / 13")
        print(f"\nCONCLUSION:")
        print(f"  Crypto 5-minute markets are fundamentally unsuitable for directional strategies.")
        print(f"  All approaches fail: compression, trend following, mean reversion from extremes.")
        print(f"\nRECOMMENDED ALTERNATIVES:")
        print(f"  1. Change timeframe: Test 15-minute or 1-hour data")
        print(f"  2. Change asset class: Test traditional markets (stocks/forex)")
        print(f"  3. Market making: Bid-ask spread capture (institutional approach)")
        print(f"  4. Abandon HFT: Focus on daily/weekly strategies")

        raise RuntimeError(
            "HARD STOP TRIGGERED: Bollinger Band mean reversion fails - "
            "ABANDON crypto 5-minute trading after 10 phases"
        )

    # Save results
    results_dir = Path('/Users/terryli/eon/backtesting.py/user_strategies/research/mean_reversion_extremes/results/phase_15_mean_reversion')
    results_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame([{
        'phase': '15A',
        'strategy': 'bollinger_mean_reversion',
        'bb_period': 20,
        'bb_std': 2.0,
        'rsi_period': 14,
        'return_pct': stats['Return [%]'],
        'n_trades': stats['# Trades'],
        'win_rate_pct': stats['Win Rate [%]'],
        'sharpe': stats['Sharpe Ratio'],
        'max_dd_pct': stats['Max. Drawdown [%]'],
    }])

    results_path = results_dir / 'phase_15a_baseline.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Results saved: {results_path}")

    return stats


if __name__ == '__main__':
    stats = run_phase_15a_validation()
