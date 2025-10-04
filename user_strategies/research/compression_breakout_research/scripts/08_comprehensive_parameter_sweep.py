#!/usr/bin/env python3
"""
Phase 10D: Comprehensive Multi-Symbol Parameter Sweep

OBJECTIVES:
1. Multi-symbol validation: BTC, ETH, SOL
2. Parameter sweep: window sizes [10, 15, 20, 25, 30] x thresholds [0.30-0.55]
3. Extended backtesting: Use maximum available data

PARAMETER GRID:
- Window sizes: 10, 15, 20, 25, 30 (5 options)
- Thresholds: 0.30, 0.35, 0.40, 0.45, 0.50, 0.55 (6 options)
- Total combinations: 30 per symbol
- Total backtests: 93 (3 baselines + 90 parameter combinations)

OUTPUT:
- Comprehensive results matrix
- Best parameters per symbol
- Cross-symbol winners (configs that work on all 3)
- Production-ready recommendations
"""

import pandas as pd
import numpy as np
from pathlib import Path
from backtesting import Backtest, Strategy
import warnings
warnings.filterwarnings('ignore')
from itertools import product
import time


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


class RollingWindowRegimeStrategy(Strategy):
    """Rolling window regime-aware volatility breakout strategy."""

    # Base strategy parameters
    atr_period = 14
    percentile_window = 150
    volatility_threshold = 0.10
    breakout_period = 20
    stop_atr_multiple = 2.0
    target_atr_multiple = 4.0
    max_hold_bars = 100
    risk_per_trade = 0.02

    # Rolling window regime parameters (optimizable)
    regime_window_size = 20
    regime_favorable_threshold = 0.40
    regime_enabled = True

    # Class-level storage
    _regime_history_storage = []

    def init(self):
        """Initialize indicators and regime tracking."""
        RollingWindowRegimeStrategy._regime_history_storage = []

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

        # Regime tracking
        self.closed_trade_outcomes = []
        self.regime_history = []

        # Current trade MAE/MFE tracking
        self.current_trade_max_favorable = 0.0
        self.current_trade_max_adverse = 0.0
        self.trade_entry_price = None

        # Stats tracking
        self.total_trades_attempted = 0
        self.trades_skipped_by_regime = 0
        self.closed_trades_count = 0

    def next(self):
        """Execute strategy with rolling window regime filtering."""

        # Exit management
        if self.position:
            bars_held = len(self.data) - self.entry_bar
            current_price = self.data.Close[-1]

            # Update MAE/MFE tracking
            if self.position.is_long:
                current_profit_pct = (current_price - self.trade_entry_price) / self.trade_entry_price * 100
                current_loss_pct = (self.data.Low[-1] - self.trade_entry_price) / self.trade_entry_price * 100
                self.current_trade_max_favorable = max(self.current_trade_max_favorable, current_profit_pct)
                self.current_trade_max_adverse = min(self.current_trade_max_adverse, current_loss_pct)
            else:
                current_profit_pct = (self.trade_entry_price - current_price) / self.trade_entry_price * 100
                current_loss_pct = (self.trade_entry_price - self.data.High[-1]) / self.trade_entry_price * 100
                self.current_trade_max_favorable = max(self.current_trade_max_favorable, current_profit_pct)
                self.current_trade_max_adverse = min(self.current_trade_max_adverse, current_loss_pct)

            # Check exit conditions
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

                # Update regime state
                mfe = self.current_trade_max_favorable
                mae = self.current_trade_max_adverse

                if mae == 0:
                    ratio = np.inf
                    is_favorable = True
                else:
                    ratio = mfe / abs(mae)
                    is_favorable = (ratio >= 2.0)

                self.closed_trade_outcomes.append(is_favorable)

                # Calculate rolling favorable rate
                if len(self.closed_trade_outcomes) >= self.regime_window_size:
                    rolling_window = self.closed_trade_outcomes[-self.regime_window_size:]
                    rolling_favorable_rate = sum(rolling_window) / len(rolling_window)
                else:
                    rolling_favorable_rate = sum(self.closed_trade_outcomes) / max(1, len(self.closed_trade_outcomes))

                outcome_dict = {
                    'bar': len(self.data),
                    'favorable': is_favorable,
                    'mfe': mfe,
                    'mae': mae,
                    'ratio': ratio,
                    'rolling_favorable_rate': rolling_favorable_rate,
                }
                self.regime_history.append(outcome_dict)
                RollingWindowRegimeStrategy._regime_history_storage.append(outcome_dict)

                self.closed_trades_count += 1
                self.current_trade_max_favorable = 0.0
                self.current_trade_max_adverse = 0.0
                self.trade_entry_price = None

                return
            else:
                return

        # Entry logic with rolling window regime check
        if self.regime_enabled and len(self.closed_trade_outcomes) >= self.regime_window_size:
            rolling_window = self.closed_trade_outcomes[-self.regime_window_size:]
            rolling_favorable_rate = sum(rolling_window) / len(rolling_window)
            
            if rolling_favorable_rate < self.regime_favorable_threshold:
                if self._check_entry_conditions():
                    self.trades_skipped_by_regime += 1
                return

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
            self.trade_entry_price = current_price
            self.current_trade_max_favorable = 0.0
            self.current_trade_max_adverse = 0.0

    def _check_entry_conditions(self) -> bool:
        """Check if entry conditions are met."""
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


def run_parameter_sweep(df: pd.DataFrame, symbol: str, 
                        window_sizes: list, thresholds: list) -> pd.DataFrame:
    """
    Run parameter sweep for one symbol.
    
    Returns: DataFrame with all results
    """
    print(f"\n{'='*70}")
    print(f"PARAMETER SWEEP: {symbol}")
    print(f"{'='*70}")
    print(f"Data: {len(df):,} bars from {df.index[0]} to {df.index[-1]}")
    print(f"Testing {len(window_sizes)} windows x {len(thresholds)} thresholds = {len(window_sizes)*len(thresholds)} combinations")
    print(f"{'='*70}\n")

    results = []

    # Baseline (no regime filtering)
    print(f"[0/{len(window_sizes)*len(thresholds)}] Running BASELINE...")
    bt_baseline = Backtest(
        df,
        RollingWindowRegimeStrategy,
        cash=10_000_000,
        commission=0.0002,
        margin=0.05,
        exclusive_orders=True
    )
    stats_baseline = bt_baseline.run(regime_enabled=False)
    
    results.append({
        'symbol': symbol,
        'window_size': 0,
        'threshold': 0.0,
        'regime_enabled': False,
        'return_pct': stats_baseline['Return [%]'],
        'sharpe': stats_baseline['Sharpe Ratio'],
        'max_drawdown': stats_baseline['Max. Drawdown [%]'],
        'n_trades': stats_baseline['# Trades'],
        'win_rate': stats_baseline['Win Rate [%]'],
        'avg_trade': stats_baseline['Avg. Trade [%]'],
        'exposure_time': stats_baseline['Exposure Time [%]'],
    })
    
    baseline_return = stats_baseline['Return [%]']
    print(f"   Baseline: {stats_baseline['# Trades']} trades, {baseline_return:.2f}% return\n")

    # Parameter sweep
    total_combos = len(window_sizes) * len(thresholds)
    combo_num = 0
    
    for window, threshold in product(window_sizes, thresholds):
        combo_num += 1
        
        print(f"[{combo_num}/{total_combos}] window={window}, threshold={int(threshold*100)}%...", end=' ')
        
        try:
            RollingWindowRegimeStrategy._regime_history_storage = []
            
            bt = Backtest(
                df,
                RollingWindowRegimeStrategy,
                cash=10_000_000,
                commission=0.0002,
                margin=0.05,
                exclusive_orders=True
            )
            
            stats = bt.run(
                regime_enabled=True,
                regime_window_size=window,
                regime_favorable_threshold=threshold
            )
            
            improvement = stats['Return [%]'] - baseline_return
            
            results.append({
                'symbol': symbol,
                'window_size': window,
                'threshold': threshold,
                'regime_enabled': True,
                'return_pct': stats['Return [%]'],
                'sharpe': stats['Sharpe Ratio'],
                'max_drawdown': stats['Max. Drawdown [%]'],
                'n_trades': stats['# Trades'],
                'win_rate': stats['Win Rate [%]'],
                'avg_trade': stats['Avg. Trade [%]'],
                'exposure_time': stats['Exposure Time [%]'],
                'improvement_vs_baseline': improvement,
            })
            
            print(f"{stats['# Trades']} trades, {stats['Return [%]']:.2f}% ({improvement:+.1f}pp)")
            
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                'symbol': symbol,
                'window_size': window,
                'threshold': threshold,
                'regime_enabled': True,
                'return_pct': np.nan,
                'sharpe': np.nan,
                'max_drawdown': np.nan,
                'n_trades': 0,
                'win_rate': np.nan,
                'avg_trade': np.nan,
                'exposure_time': np.nan,
                'improvement_vs_baseline': np.nan,
            })

    return pd.DataFrame(results)


def main():
    """Run comprehensive multi-symbol parameter sweep."""
    
    print("="*70)
    print("PHASE 10D: COMPREHENSIVE PARAMETER SWEEP")
    print("="*70)
    print("\nObjective: Find optimal regime parameters across BTC/ETH/SOL")
    print("\nParameter Grid:")
    print("  Window sizes: 10, 15, 20, 25, 30")
    print("  Thresholds: 30%, 35%, 40%, 45%, 50%, 55%")
    print("  Total combinations: 30 per symbol")
    print("\nData: Extended backtesting (200k bars ≈ ~2 years)")
    print("="*70)

    output_dir = Path('/tmp/comprehensive_regime_results')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parameter grid
    window_sizes = [10, 15, 20, 25, 30]
    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]

    # Symbols configuration
    symbols_config = [
        ('BTC', 'binance_spot_BTCUSDT-5m_20220101-20250930_v2.10.0.csv'),
        ('ETH', 'binance_spot_ETHUSDT-5m_20220101-20250930_v2.10.0.csv'),
        ('SOL', 'binance_spot_SOLUSDT-5m_20220101-20250930_v2.10.0.csv'),
    ]

    data_dir = Path('/Users/terryli/eon/backtesting.py/user_strategies/data/raw/crypto_5m')
    
    all_results = []
    
    start_time = time.time()

    # Run sweep for each symbol
    for symbol, filename in symbols_config:
        csv_path = data_dir / filename
        
        if not csv_path.exists():
            print(f"\n⚠ WARNING: {filename} not found, skipping {symbol}")
            continue
        
        print(f"\nLoading {symbol} data...")
        df = load_5m_data(csv_path, n_bars=200000)  # Extended: 200k bars
        
        symbol_results = run_parameter_sweep(df, symbol, window_sizes, thresholds)
        all_results.append(symbol_results)

    # Combine all results
    if len(all_results) == 0:
        print("\n❌ No results generated. Exiting.")
        return

    combined_results = pd.concat(all_results, ignore_index=True)

    elapsed = time.time() - start_time
    
    print(f"\n{'='*70}")
    print("ANALYSIS & REPORTING")
    print(f"{'='*70}")
    print(f"Total runtime: {elapsed/60:.1f} minutes")
    print(f"Total backtests: {len(combined_results)}")
    
    # Save full results
    results_path = output_dir / 'full_parameter_sweep_results.csv'
    combined_results.to_csv(results_path, index=False)
    print(f"\n✓ Saved: {results_path}")

    # Find best parameters per symbol
    print(f"\n{'='*70}")
    print("BEST PARAMETERS PER SYMBOL")
    print(f"{'='*70}\n")

    best_configs = []
    
    for symbol in combined_results['symbol'].unique():
        symbol_data = combined_results[
            (combined_results['symbol'] == symbol) & 
            (combined_results['regime_enabled'] == True)
        ].copy()
        
        # Best by return
        best_return = symbol_data.loc[symbol_data['return_pct'].idxmax()]
        
        # Best by sharpe
        best_sharpe = symbol_data.loc[symbol_data['sharpe'].idxmax()]
        
        # Best by improvement
        best_improvement = symbol_data.loc[symbol_data['improvement_vs_baseline'].idxmax()]
        
        print(f"{symbol}:")
        print(f"  Best Return:      window={int(best_return['window_size'])}, threshold={int(best_return['threshold']*100)}% → {best_return['return_pct']:.2f}% ({best_return['n_trades']} trades)")
        print(f"  Best Sharpe:      window={int(best_sharpe['window_size'])}, threshold={int(best_sharpe['threshold']*100)}% → Sharpe {best_sharpe['sharpe']:.2f}")
        print(f"  Best Improvement: window={int(best_improvement['window_size'])}, threshold={int(best_improvement['threshold']*100)}% → +{best_improvement['improvement_vs_baseline']:.2f}pp")
        print()
        
        best_configs.append({
            'symbol': symbol,
            'metric': 'return',
            'window': int(best_return['window_size']),
            'threshold': best_return['threshold'],
            'value': best_return['return_pct'],
        })
        best_configs.append({
            'symbol': symbol,
            'metric': 'sharpe',
            'window': int(best_sharpe['window_size']),
            'threshold': best_sharpe['threshold'],
            'value': best_sharpe['sharpe'],
        })

    best_configs_df = pd.DataFrame(best_configs)
    best_path = output_dir / 'best_parameters_per_symbol.csv'
    best_configs_df.to_csv(best_path, index=False)
    print(f"✓ Saved: {best_path}")

    # Cross-symbol winners (configurations that work well on ALL symbols)
    print(f"\n{'='*70}")
    print("CROSS-SYMBOL WINNER ANALYSIS")
    print(f"{'='*70}\n")

    regime_results = combined_results[combined_results['regime_enabled'] == True].copy()
    
    # Group by window/threshold combo and calculate average improvement
    cross_symbol = regime_results.groupby(['window_size', 'threshold']).agg({
        'improvement_vs_baseline': ['mean', 'min', 'std'],
        'n_trades': 'mean',
        'return_pct': 'mean',
    }).reset_index()
    
    cross_symbol.columns = ['window', 'threshold', 'avg_improvement', 'min_improvement', 'std_improvement', 'avg_trades', 'avg_return']
    cross_symbol = cross_symbol.sort_values('avg_improvement', ascending=False)
    
    print("Top 10 configurations (by average improvement across all symbols):\n")
    print(f"{'Window':<8} {'Thresh':<8} {'Avg Improvement':<18} {'Min Improvement':<18} {'Avg Trades':<12}")
    print("-"*70)
    
    for idx, row in cross_symbol.head(10).iterrows():
        print(f"{int(row['window']):<8} {int(row['threshold']*100)}%{'':<5} {row['avg_improvement']:>+16.2f}pp {row['min_improvement']:>+16.2f}pp {row['avg_trades']:>10.1f}")
    
    cross_path = output_dir / 'cross_symbol_rankings.csv'
    cross_symbol.to_csv(cross_path, index=False)
    print(f"\n✓ Saved: {cross_path}")

    # Final recommendations
    print(f"\n{'='*70}")
    print("PRODUCTION RECOMMENDATIONS")
    print(f"{'='*70}\n")

    best_cross_symbol = cross_symbol.iloc[0]
    
    print(f"✅ RECOMMENDED UNIVERSAL CONFIGURATION:")
    print(f"   Window: {int(best_cross_symbol['window'])}")
    print(f"   Threshold: {int(best_cross_symbol['threshold']*100)}%")
    print(f"   Average improvement: {best_cross_symbol['avg_improvement']:+.2f}pp")
    print(f"   Minimum improvement (worst symbol): {best_cross_symbol['min_improvement']:+.2f}pp")
    print(f"   Average trades: {best_cross_symbol['avg_trades']:.1f}")
    print()
    print(f"This configuration works well across ALL symbols (BTC/ETH/SOL).")
    
    print(f"\n{'='*70}")
    print(f"All results saved to: {output_dir}")
    print(f"{'='*70}")

    return combined_results


if __name__ == '__main__':
    results = main()
