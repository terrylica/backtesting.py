#!/usr/bin/env python3
"""
Multi-Timeframe Volatility Compression Breakout Strategy

HYPOTHESIS: Low volatility across multiple timeframes (5m, 15m, 30m) indicates
compression that precedes directional breakouts.

ENTRY RULES:
1. ATR(14) on 5m, 15m, AND 30m must ALL be in bottom 10% of their rolling 150-bar distributions
2. Wait for breakout confirmation: price exceeds 20-bar high (long) or 20-bar low (short)
3. Enter on next bar after breakout confirmation

EXIT RULES:
- Stop loss: 2x ATR from entry price
- Take profit: 4x ATR from entry price (2:1 reward/risk)
- Time stop: Close after 100 bars if neither hit

POSITION SIZING:
- Equal risk per trade: size = risk_capital / (2 * ATR)

This replaces the failed regime detection approach which showed no predictive power.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import warnings
warnings.filterwarnings('ignore')


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range.

    TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
    ATR = EMA(TR, period)
    """
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
    """
    Calculate rolling percentile rank (0-1) over window.

    Value of 0.10 means current value is at 10th percentile (bottom 10%).
    """
    def percentile_rank(x):
        if len(x) < 2:
            return np.nan
        current_value = x.iloc[-1]
        return (x < current_value).sum() / len(x)

    return series.rolling(window).apply(percentile_rank, raw=False)


class VolatilityBreakoutStrategy(Strategy):
    """
    Multi-timeframe volatility compression breakout strategy.

    Parameters:
        atr_period: Lookback for ATR calculation (default: 14)
        percentile_window: Window for percentile ranking (default: 150)
        volatility_threshold: Maximum percentile to enter (default: 0.10 = bottom 10%)
        breakout_period: Lookback for high/low breakout (default: 20)
        stop_atr_multiple: Stop loss as multiple of ATR (default: 2.0)
        target_atr_multiple: Take profit as multiple of ATR (default: 4.0)
        max_hold_bars: Maximum bars to hold position (default: 100)
        risk_per_trade: Risk as fraction of equity (default: 0.02 = 2%)
    """

    # Strategy parameters (can be optimized)
    atr_period = 14
    percentile_window = 150
    volatility_threshold = 0.10
    breakout_period = 20
    stop_atr_multiple = 2.0
    target_atr_multiple = 4.0
    max_hold_bars = 100
    risk_per_trade = 0.02

    def init(self):
        """Initialize indicators and multi-timeframe data."""

        # Get 5-minute data (base timeframe)
        df_5m = self.data.df.copy()

        # Calculate 5m ATR
        atr_5m = calculate_atr(df_5m, self.atr_period)

        # Resample to 15m
        df_15m = df_5m.resample('15min').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        atr_15m = calculate_atr(df_15m, self.atr_period)

        # Resample to 30m
        df_30m = df_5m.resample('30min').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        atr_30m = calculate_atr(df_30m, self.atr_period)

        # Calculate percentile ranks
        atr_5m_pct = calculate_percentile_rank(atr_5m, self.percentile_window)
        atr_15m_pct = calculate_percentile_rank(atr_15m, self.percentile_window)
        atr_30m_pct = calculate_percentile_rank(atr_30m, self.percentile_window)

        # Align multi-timeframe data back to 5m index (forward fill)
        atr_15m_aligned = atr_15m_pct.reindex(df_5m.index, method='ffill')
        atr_30m_aligned = atr_30m_pct.reindex(df_5m.index, method='ffill')

        # Create composite volatility filter: ALL must be in bottom 10%
        self.low_vol_filter = self.I(
            lambda: (
                (atr_5m_pct < self.volatility_threshold) &
                (atr_15m_aligned < self.volatility_threshold) &
                (atr_30m_aligned < self.volatility_threshold)
            ).astype(int),
            name='LowVolFilter'
        )

        # Breakout levels
        self.breakout_high = self.I(
            lambda: df_5m['High'].rolling(self.breakout_period).max(),
            name='BreakoutHigh'
        )
        self.breakout_low = self.I(
            lambda: df_5m['Low'].rolling(self.breakout_period).min(),
            name='BreakoutLow'
        )

        # ATR for position sizing and stops
        self.atr = self.I(lambda: atr_5m, name='ATR')

        # Track entry bar for time-based exit
        self.entry_bar = None
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None

    def next(self):
        """Execute strategy logic on each bar."""

        # Exit management for existing position
        if self.position:
            bars_held = len(self.data) - self.entry_bar

            # Time-based exit
            if bars_held >= self.max_hold_bars:
                self.position.close()
                return

            # Stop loss
            if self.position.is_long and self.data.Low[-1] <= self.stop_loss:
                self.position.close()
                return
            elif self.position.is_short and self.data.High[-1] >= self.stop_loss:
                self.position.close()
                return

            # Take profit
            if self.position.is_long and self.data.High[-1] >= self.take_profit:
                self.position.close()
                return
            elif self.position.is_short and self.data.Low[-1] <= self.take_profit:
                self.position.close()
                return

            return  # Hold position

        # Entry logic: only if no position
        # Check if volatility filter is active
        if self.low_vol_filter[-1] != 1:
            return

        # Need sufficient data
        if len(self.data) < max(self.percentile_window, self.breakout_period) + 50:
            return

        current_atr = self.atr[-1]
        if pd.isna(current_atr) or current_atr == 0:
            return

        # Check for breakout
        current_price = self.data.Close[-1]
        prev_high = self.breakout_high[-2]  # Previous bar's 20-bar high
        prev_low = self.breakout_low[-2]    # Previous bar's 20-bar low

        # Upside breakout: price exceeds 20-bar high
        if current_price > prev_high:
            # Calculate position size based on equal risk
            # We want: (position_value / equity) * (stop_distance / price) = risk_per_trade
            # So: position_fraction = risk_per_trade * price / stop_distance
            stop_distance = self.stop_atr_multiple * current_atr

            # Fraction of equity to allocate
            position_fraction = self.risk_per_trade * current_price / stop_distance

            # Ensure position is reasonable (between 0.01 and 0.95 of equity)
            position_fraction = max(0.01, min(0.95, position_fraction))

            # Enter long
            self.buy(size=position_fraction)
            self.entry_bar = len(self.data)
            self.entry_price = current_price
            self.stop_loss = current_price - stop_distance
            self.take_profit = current_price + (self.target_atr_multiple * current_atr)

        # Downside breakout: price breaks below 20-bar low
        elif current_price < prev_low:
            # Calculate position size
            stop_distance = self.stop_atr_multiple * current_atr

            # Fraction of equity to allocate
            position_fraction = self.risk_per_trade * current_price / stop_distance

            # Ensure position is reasonable
            position_fraction = max(0.01, min(0.95, position_fraction))

            # Enter short
            self.sell(size=position_fraction)
            self.entry_bar = len(self.data)
            self.entry_price = current_price
            self.stop_loss = current_price + stop_distance
            self.take_profit = current_price - (self.target_atr_multiple * current_atr)


def load_5m_data(csv_path: Path, n_bars: int = 100000) -> pd.DataFrame:
    """Load recent 5-minute BTC data."""
    print(f"Loading last {n_bars:,} bars from {csv_path.name}...")

    df = pd.read_csv(csv_path, skiprows=10)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    df = df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df = df.tail(n_bars)

    print(f"  Loaded {len(df):,} bars from {df.index[0]} to {df.index[-1]}")
    return df


def main():
    """Run volatility breakout strategy backtest."""

    print("="*70)
    print("MULTI-TIMEFRAME VOLATILITY COMPRESSION BREAKOUT STRATEGY")
    print("="*70)
    print()
    print("Strategy Logic:")
    print("  1. Monitor ATR(14) on 5m, 15m, and 30m timeframes")
    print("  2. Calculate rolling 150-bar percentile rank for each ATR")
    print("  3. Enter ONLY when ALL three timeframes in bottom 10%")
    print("  4. Wait for breakout: 20-bar high (long) or low (short)")
    print("  5. Exit: 2x ATR stop loss, 4x ATR take profit, or 100-bar time limit")
    print("  6. Position sizing: Equal 2% risk per trade")
    print()
    print("="*70)
    print()

    # Load data
    csv_path = Path('user_strategies/data/raw/crypto_5m/binance_spot_BTCUSDT-5m_20220101-20250930_v2.10.0.csv')
    df = load_5m_data(csv_path, n_bars=100000)

    # Run backtest
    print("Running backtest...")
    bt = Backtest(
        df,
        VolatilityBreakoutStrategy,
        cash=10_000_000,
        commission=0.0002,  # 2 basis points
        margin=0.05,        # 20x leverage available
        exclusive_orders=True
    )

    stats = bt.run()

    # Display results
    print()
    print("="*70)
    print("BACKTEST RESULTS")
    print("="*70)
    print()
    print(f"Period: {df.index[0]} to {df.index[-1]}")
    print(f"Total Bars: {len(df):,}")
    print()
    print(f"Return [%]:              {stats['Return [%]']:>12.2f}%")
    print(f"Buy & Hold Return [%]:   {stats['Buy & Hold Return [%]']:>12.2f}%")
    print(f"Max Drawdown [%]:        {stats['Max. Drawdown [%]']:>12.2f}%")
    print(f"Sharpe Ratio:            {stats['Sharpe Ratio']:>12.2f}")
    print(f"Sortino Ratio:           {stats['Sortino Ratio']:>12.2f}")
    print()
    print(f"Total Trades:            {stats['# Trades']:>12}")
    print(f"Win Rate [%]:            {stats['Win Rate [%]']:>12.2f}%")
    print(f"Best Trade [%]:          {stats['Best Trade [%]']:>12.2f}%")
    print(f"Worst Trade [%]:         {stats['Worst Trade [%]']:>12.2f}%")
    print(f"Avg Trade [%]:           {stats['Avg. Trade [%]']:>12.2f}%")
    print()
    print(f"Max Trade Duration:      {str(stats['Max. Trade Duration']):>12}")
    print(f"Avg Trade Duration:      {str(stats['Avg. Trade Duration']):>12}")
    print()
    print(f"Exposure Time [%]:       {stats['Exposure Time [%]']:>12.2f}%")
    print()
    print("="*70)
    print()

    # Analysis
    return_pct = stats['Return [%]']
    bnh_return = stats['Buy & Hold Return [%]']
    n_trades = stats['# Trades']
    win_rate = stats['Win Rate [%]']
    sharpe = stats['Sharpe Ratio']

    print("ANALYSIS:")
    print()

    if n_trades == 0:
        print("❌ NO TRADES EXECUTED")
        print("   → Volatility filter too restrictive (never all 3 timeframes in bottom 10%)")
        print("   → OR breakout conditions never met")
        print()
        print("Recommendations:")
        print("   1. Relax threshold: Try 15% or 20% instead of 10%")
        print("   2. Require only 2 of 3 timeframes in low volatility")
        print("   3. Shorten percentile window: Try 100 bars instead of 150")
    elif return_pct > bnh_return and sharpe > 1.0:
        print(f"✓ STRATEGY OUTPERFORMS BUY & HOLD")
        print(f"   → Alpha: +{return_pct - bnh_return:.2f}% vs benchmark")
        print(f"   → Sharpe {sharpe:.2f} indicates risk-adjusted outperformance")
        print(f"   → {n_trades} trades with {win_rate:.1f}% win rate")
        print()
        print("   This validates the volatility compression → breakout hypothesis!")
    elif n_trades > 0 and win_rate < 45:
        print(f"✗ STRATEGY SHOWS POOR WIN RATE ({win_rate:.1f}%)")
        print(f"   → {n_trades} trades executed but low success rate")
        print(f"   → Return: {return_pct:.2f}% vs Buy & Hold: {bnh_return:.2f}%")
        print()
        print("Possible issues:")
        print("   1. Breakout signals are false (price reverses after breaking)")
        print("   2. Stops too tight (2x ATR may be insufficient)")
        print("   3. Low volatility doesn't predict breakout direction")
    else:
        print(f"~ MIXED RESULTS")
        print(f"   → {n_trades} trades, {win_rate:.1f}% win rate")
        print(f"   → Return: {return_pct:.2f}% vs Buy & Hold: {bnh_return:.2f}%")
        print(f"   → Sharpe: {sharpe:.2f}")
        print()
        print("   Further investigation needed.")

    print()
    print("="*70)

    return stats, bt


if __name__ == '__main__':
    stats, bt = main()
