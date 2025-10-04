#!/usr/bin/env python3
"""
Phase 10: Regime-Aware Volatility Breakout Strategy

HYPOTHESIS: Adding regime filtering (skip trades during unfavorable streaks ≥5)
transforms the failed Phase 7 strategy (-95% return) into a profitable strategy.

REGIME LOGIC:
1. Track MAE/MFE for each closed trade
2. Define favorable: MFE/|MAE| >= 2.0 (matches Phase 10 definition)
3. Maintain unfavorable_streak counter
4. Skip ALL entry signals when unfavorable_streak >= 5

TEMPORAL INTEGRITY:
- Decision to skip/take trade uses ONLY past closed trade outcomes
- Current trade's outcome is unknown when making decision
- Streak updated AFTER trade closes

BASE STRATEGY (from Phase 7):
- Entry: Multi-timeframe volatility compression + 20-bar breakout
- Exit: 2x ATR stop, 4x ATR target, 100-bar time limit
"""

import pandas as pd
import numpy as np
from pathlib import Path
from backtesting import Backtest, Strategy
import warnings
warnings.filterwarnings('ignore')


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
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
    """Calculate rolling percentile rank (0-1)."""
    def percentile_rank(x):
        if len(x) < 2:
            return np.nan
        current_value = x.iloc[-1]
        return (x < current_value).sum() / len(x)

    return series.rolling(window).apply(percentile_rank, raw=False)


class RegimeAwareVolatilityBreakout(Strategy):
    """
    Volatility breakout strategy with regime filtering.

    NEW PARAMETERS:
        regime_threshold: Skip trades when unfavorable_streak >= N (default: 5)
        regime_enabled: Enable/disable regime filtering (default: True)
    """

    # Base strategy parameters
    atr_period = 14
    percentile_window = 150
    volatility_threshold = 0.10
    breakout_period = 20
    stop_atr_multiple = 2.0
    target_atr_multiple = 4.0
    max_hold_bars = 100
    risk_per_trade = 0.02

    # Regime parameters
    regime_threshold = 5
    regime_enabled = True

    # Class-level storage for regime history (shared across runs)
    _regime_history_storage = []

    def init(self):
        """Initialize indicators and regime tracking."""

        # Clear class-level storage at start of backtest
        RegimeAwareVolatilityBreakout._regime_history_storage = []

        # Multi-timeframe volatility setup (same as Phase 7)
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

        # Position tracking
        self.entry_bar = None
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None

        # ===== REGIME TRACKING (NEW) =====
        self.unfavorable_streak = 0
        self.favorable_streak = 0
        self.regime_history = []  # List of dicts

        # Current trade MAE/MFE tracking
        self.current_trade_max_favorable = 0.0
        self.current_trade_max_adverse = 0.0
        self.trade_entry_price = None

        # Stats tracking
        self.total_trades_attempted = 0
        self.trades_skipped_by_regime = 0
        self.closed_trades_count = 0

    def next(self):
        """Execute strategy with regime filtering."""

        # ===== EXIT MANAGEMENT =====
        if self.position:
            bars_held = len(self.data) - self.entry_bar
            current_price = self.data.Close[-1]

            # Update MAE/MFE tracking
            if self.position.is_long:
                # Favorable = profit, Adverse = loss
                current_profit_pct = (current_price - self.trade_entry_price) / self.trade_entry_price * 100
                current_loss_pct = (self.data.Low[-1] - self.trade_entry_price) / self.trade_entry_price * 100

                self.current_trade_max_favorable = max(self.current_trade_max_favorable, current_profit_pct)
                self.current_trade_max_adverse = min(self.current_trade_max_adverse, current_loss_pct)

            else:  # short position
                # Favorable = profit (price down), Adverse = loss (price up)
                current_profit_pct = (self.trade_entry_price - current_price) / self.trade_entry_price * 100
                current_loss_pct = (self.trade_entry_price - self.data.High[-1]) / self.trade_entry_price * 100

                self.current_trade_max_favorable = max(self.current_trade_max_favorable, current_profit_pct)
                self.current_trade_max_adverse = min(self.current_trade_max_adverse, current_loss_pct)

            # Check exit conditions
            should_exit = False

            # Time-based exit
            if bars_held >= self.max_hold_bars:
                should_exit = True

            # Stop loss
            elif self.position.is_long and self.data.Low[-1] <= self.stop_loss:
                should_exit = True
            elif self.position.is_short and self.data.High[-1] >= self.stop_loss:
                should_exit = True

            # Take profit
            elif self.position.is_long and self.data.High[-1] >= self.take_profit:
                should_exit = True
            elif self.position.is_short and self.data.Low[-1] <= self.take_profit:
                should_exit = True

            if should_exit:
                self.position.close()

                # ===== UPDATE REGIME STATE (after close) =====
                mfe = self.current_trade_max_favorable
                mae = self.current_trade_max_adverse

                # Calculate ratio (match Phase 10 definition)
                # favorable = MFE / |MAE| >= 2.0
                if mae == 0:
                    # No adverse movement - extremely favorable
                    ratio = np.inf
                    is_favorable = True
                else:
                    ratio = mfe / abs(mae)
                    is_favorable = (ratio >= 2.0)

                # Update streaks
                if is_favorable:
                    self.favorable_streak += 1
                    self.unfavorable_streak = 0
                else:
                    self.unfavorable_streak += 1
                    self.favorable_streak = 0

                # Record outcome (both instance and class level)
                outcome_dict = {
                    'bar': len(self.data),
                    'favorable': is_favorable,
                    'mfe': mfe,
                    'mae': mae,
                    'ratio': ratio,
                    'unfavorable_streak': self.unfavorable_streak,
                    'favorable_streak': self.favorable_streak,
                }
                self.regime_history.append(outcome_dict)
                RegimeAwareVolatilityBreakout._regime_history_storage.append(outcome_dict)

                self.closed_trades_count += 1

                # Reset MAE/MFE for next trade
                self.current_trade_max_favorable = 0.0
                self.current_trade_max_adverse = 0.0
                self.trade_entry_price = None

                return
            else:
                return  # Hold position

        # ===== ENTRY LOGIC =====
        # Check if regime allows trading
        if self.regime_enabled and self.unfavorable_streak >= self.regime_threshold:
            # Track that we skipped a signal (if there would have been one)
            if self._check_entry_conditions():
                self.trades_skipped_by_regime += 1
            return  # Skip all entries during unfavorable regime

        # Proceed with normal entry logic
        if not self._check_entry_conditions():
            return

        current_atr = self.atr[-1]
        current_price = self.data.Close[-1]
        prev_high = self.breakout_high[-2]
        prev_low = self.breakout_low[-2]

        self.total_trades_attempted += 1

        # Upside breakout
        if current_price > prev_high:
            stop_distance = self.stop_atr_multiple * current_atr
            position_fraction = self.risk_per_trade * current_price / stop_distance
            position_fraction = max(0.01, min(0.95, position_fraction))

            self.buy(size=position_fraction)
            self.entry_bar = len(self.data)
            self.entry_price = current_price
            self.stop_loss = current_price - stop_distance
            self.take_profit = current_price + (self.target_atr_multiple * current_atr)

            # Initialize MAE/MFE tracking
            self.trade_entry_price = current_price
            self.current_trade_max_favorable = 0.0
            self.current_trade_max_adverse = 0.0

        # Downside breakout
        elif current_price < prev_low:
            stop_distance = self.stop_atr_multiple * current_atr
            position_fraction = self.risk_per_trade * current_price / stop_distance
            position_fraction = max(0.01, min(0.95, position_fraction))

            self.sell(size=position_fraction)
            self.entry_bar = len(self.data)
            self.entry_price = current_price
            self.stop_loss = current_price + stop_distance
            self.take_profit = current_price - (self.target_atr_multiple * current_atr)

            # Initialize MAE/MFE tracking
            self.trade_entry_price = current_price
            self.current_trade_max_favorable = 0.0
            self.current_trade_max_adverse = 0.0

    def _check_entry_conditions(self) -> bool:
        """Check if entry conditions are met (without entering)."""
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

        # Check if breakout signal exists
        if current_price > prev_high or current_price < prev_low:
            return True

        return False


def load_5m_data(csv_path: Path, n_bars: int = 100000) -> pd.DataFrame:
    """Load 5-minute crypto data."""
    print(f"Loading last {n_bars:,} bars from {csv_path.name}...")

    df = pd.read_csv(csv_path, skiprows=10)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    df = df.rename(columns={
        'open': 'Open', 'high': 'High', 'low': 'Low',
        'close': 'Close', 'volume': 'Volume'
    })

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].tail(n_bars)

    print(f"  Loaded {len(df):,} bars from {df.index[0]} to {df.index[-1]}")
    return df


def run_comparison_backtest(df: pd.DataFrame, symbol: str, output_dir: Path):
    """
    Run both baseline and regime-filtered backtests for comparison.

    Returns: (baseline_stats, regime_stats)
    """
    print(f"\n{'='*70}")
    print(f"COMPARING BASELINE VS REGIME-FILTERED: {symbol}")
    print(f"{'='*70}\n")

    # Baseline (regime disabled)
    print("Running BASELINE backtest (regime filtering OFF)...")
    bt_baseline = Backtest(
        df,
        RegimeAwareVolatilityBreakout,
        cash=10_000_000,
        commission=0.0002,
        margin=0.05,
        exclusive_orders=True
    )

    stats_baseline = bt_baseline.run(regime_enabled=False)

    print(f"  Baseline: {stats_baseline['# Trades']} trades, {stats_baseline['Return [%]']:.2f}% return")

    # Regime-filtered
    print("\nRunning REGIME-FILTERED backtest (threshold=5)...")
    bt_regime = Backtest(
        df,
        RegimeAwareVolatilityBreakout,
        cash=10_000_000,
        commission=0.0002,
        margin=0.05,
        exclusive_orders=True
    )

    stats_regime = bt_regime.run(regime_enabled=True, regime_threshold=5)

    print(f"  Regime-filtered: {stats_regime['# Trades']} trades, {stats_regime['Return [%]']:.2f}% return")

    # Extract regime stats from class-level storage
    regime_history_df = pd.DataFrame(RegimeAwareVolatilityBreakout._regime_history_storage)

    # Calculate metrics
    if len(regime_history_df) > 0:
        favorable_rate = regime_history_df['favorable'].mean() * 100

        print(f"\nRegime Statistics:")
        print(f"  Total closed trades: {len(regime_history_df)}")
        print(f"  Favorable rate: {favorable_rate:.2f}%")
        print(f"  Max unfavorable streak: {regime_history_df['unfavorable_streak'].max()}")

    return stats_baseline, stats_regime, regime_history_df


def generate_comparison_report(baseline, regime, regime_history, symbol, output_dir):
    """Generate detailed comparison report."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # Summary table
    comparison = pd.DataFrame({
        'metric': [
            'Return [%]',
            'Buy & Hold Return [%]',
            'Max Drawdown [%]',
            'Sharpe Ratio',
            'Sortino Ratio',
            '# Trades',
            'Win Rate [%]',
            'Avg Trade [%]',
            'Best Trade [%]',
            'Worst Trade [%]',
            'Exposure Time [%]',
        ],
        'baseline': [
            baseline['Return [%]'],
            baseline['Buy & Hold Return [%]'],
            baseline['Max. Drawdown [%]'],
            baseline['Sharpe Ratio'],
            baseline['Sortino Ratio'],
            baseline['# Trades'],
            baseline['Win Rate [%]'],
            baseline['Avg. Trade [%]'],
            baseline['Best Trade [%]'],
            baseline['Worst Trade [%]'],
            baseline['Exposure Time [%]'],
        ],
        'regime_filtered': [
            regime['Return [%]'],
            regime['Buy & Hold Return [%]'],
            regime['Max. Drawdown [%]'],
            regime['Sharpe Ratio'],
            regime['Sortino Ratio'],
            regime['# Trades'],
            regime['Win Rate [%]'],
            regime['Avg. Trade [%]'],
            regime['Best Trade [%]'],
            regime['Worst Trade [%]'],
            regime['Exposure Time [%]'],
        ],
    })

    comparison['improvement'] = comparison['regime_filtered'] - comparison['baseline']
    comparison['pct_change'] = (comparison['improvement'] / comparison['baseline'].abs()) * 100

    csv_path = output_dir / f'{symbol}_comparison.csv'
    comparison.to_csv(csv_path, index=False)
    print(f"\n  ✓ Saved: {csv_path}")

    # Regime history
    if len(regime_history) > 0:
        regime_path = output_dir / f'{symbol}_regime_history.csv'
        regime_history.to_csv(regime_path, index=False)
        print(f"  ✓ Saved: {regime_path}")

    # Markdown report
    report = f"""# Phase 10: Regime-Aware Strategy Results - {symbol}

**Symbol:** {symbol}
**Period:** {baseline['Start']} to {baseline['End']}
**Regime Threshold:** ≥5 consecutive unfavorable outcomes

---

## Performance Comparison

| Metric | Baseline | Regime-Filtered | Improvement |
|--------|----------|-----------------|-------------|
| **Return [%]** | {baseline['Return [%]']:.2f}% | {regime['Return [%]']:.2f}% | {regime['Return [%]'] - baseline['Return [%]']:+.2f}% |
| **Sharpe Ratio** | {baseline['Sharpe Ratio']:.2f} | {regime['Sharpe Ratio']:.2f} | {regime['Sharpe Ratio'] - baseline['Sharpe Ratio']:+.2f} |
| **Max Drawdown** | {baseline['Max. Drawdown [%]']:.2f}% | {regime['Max. Drawdown [%]']:.2f}% | {regime['Max. Drawdown [%]'] - baseline['Max. Drawdown [%]']:+.2f}% |
| **# Trades** | {baseline['# Trades']} | {regime['# Trades']} | {regime['# Trades'] - baseline['# Trades']:+} |
| **Win Rate** | {baseline['Win Rate [%]']:.2f}% | {regime['Win Rate [%]']:.2f}% | {regime['Win Rate [%]'] - baseline['Win Rate [%]']:+.2f}% |

---

## Regime Analysis

"""

    if len(regime_history) > 0:
        fav_rate = regime_history['favorable'].mean() * 100
        mean_fav_streak = regime_history[regime_history['favorable']]['favorable_streak'].mean()
        mean_unfav_streak = regime_history[~regime_history['favorable']]['unfavorable_streak'].mean()
        max_unfav_streak = regime_history['unfavorable_streak'].max()

        report += f"- **Favorable rate (MFE/MAE ≥ 2.0):** {fav_rate:.2f}%\n"
        report += f"- **Mean favorable streak:** {mean_fav_streak:.2f} trades\n"
        report += f"- **Mean unfavorable streak:** {mean_unfav_streak:.2f} trades\n"
        report += f"- **Max unfavorable streak:** {max_unfav_streak} trades\n"

    report += "\n---\n\n## Verdict\n\n"

    return_improvement = regime['Return [%]'] - baseline['Return [%]']

    if return_improvement > 50:
        report += "✅ **MAJOR IMPROVEMENT**\n\n"
        report += f"Regime filtering transforms strategy from {baseline['Return [%]']:.2f}% to {regime['Return [%]']:.2f}% (+{return_improvement:.2f}%).\n"
    elif return_improvement > 0:
        report += "✓ **IMPROVEMENT**\n\n"
        report += f"Regime filtering improves returns by {return_improvement:.2f} percentage points.\n"
    else:
        report += "✗ **NO IMPROVEMENT**\n\n"
        report += f"Regime filtering does not improve performance ({return_improvement:.2f}% change).\n"

    md_path = output_dir / f'{symbol}_REGIME_STRATEGY_REPORT.md'
    with open(md_path, 'w') as f:
        f.write(report)
    print(f"  ✓ Saved: {md_path}")

    return comparison


def main():
    """Run full regime-aware strategy validation."""

    print("="*70)
    print("PHASE 10: REGIME-AWARE STRATEGY BACKTEST")
    print("="*70)
    print("\nStrategy:")
    print("  Base: Multi-timeframe volatility compression + breakout (Phase 7)")
    print("  Enhancement: Skip trades when unfavorable_streak >= 5")
    print("  Favorable definition: MFE/|MAE| >= 2.0 (matches Phase 10)")
    print("\nHypothesis:")
    print("  Regime filtering transforms -95% baseline → positive returns")
    print("="*70)

    output_dir = Path('/tmp/regime_strategy_results')

    # Test on BTC first (most data in Phase 10)
    csv_path = Path('user_strategies/data/raw/crypto_5m/binance_spot_BTCUSDT-5m_20220101-20250930_v2.10.0.csv')
    df = load_5m_data(csv_path, n_bars=100000)

    baseline_stats, regime_stats, regime_history = run_comparison_backtest(df, 'BTC', output_dir)

    print(f"\n{'='*70}")
    print("GENERATING COMPARISON REPORT")
    print(f"{'='*70}")

    comparison_df = generate_comparison_report(
        baseline_stats, regime_stats, regime_history, 'BTC', output_dir
    )

    # Final verdict
    print(f"\n{'='*70}")
    print("FINAL VERDICT")
    print(f"{'='*70}\n")

    baseline_return = baseline_stats['Return [%]']
    regime_return = regime_stats['Return [%]']
    improvement = regime_return - baseline_return

    print(f"Baseline strategy:       {baseline_return:>10.2f}%")
    print(f"Regime-filtered:         {regime_return:>10.2f}%")
    print(f"Improvement:             {improvement:>10.2f} percentage points\n")

    if improvement > 50:
        print("✅ MAJOR SUCCESS: Regime filtering dramatically improves performance")
        print("   → Validates Phase 9 breakthrough (regime discovery)")
        print("   → Validates Phase 10 retrospective simulation")
        print("   → Ready for production deployment consideration")
    elif improvement > 0:
        print("✓ SUCCESS: Regime filtering improves performance")
        print("   → Further optimization recommended")
    else:
        print("❌ FAILURE: Regime filtering does not improve performance")
        print("   → Investigate discrepancy with Phase 10 simulation")
        print("   → Possible issues: MAE/MFE calculation, temporal alignment")

    print(f"\n{'='*70}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*70}")

    return baseline_stats, regime_stats, regime_history


if __name__ == '__main__':
    baseline, regime, history = main()
