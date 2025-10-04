#!/usr/bin/env python3
"""
Volatility Compression Breakout Research Tool

RESEARCH HYPOTHESIS:
Does multi-timeframe volatility compression predict clean directional breakouts?

METHODOLOGY:
1. Detect compression: 5m/15m/30m ATR all in bottom N% of 150-bar distribution
2. Identify breakouts: Price exceeds 20-bar high (bullish) or low (bearish)
3. Measure quality: MAE/MFE ratio over multiple forward horizons
4. Define success: MFE/|MAE| ≥ 2.0 (bullish) or |MAE|/MFE ≥ 2.0 (bearish)

PARAMETERS TO TEST:
- Compression thresholds: 5%, 10%, 15%, 20%
- Forward horizons: 10, 20, 30, 50, 100 bars
- Symbols: BTC, ETH, SOL

OUTPUT:
- CSV: All breakout events with MAE/MFE metrics
- Tables: Statistical summaries per symbol/threshold/horizon
- Plots: Distribution histograms, comparative analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


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
    """Calculate rolling percentile rank (0-1) over window."""
    def percentile_rank(x):
        if len(x) < 2:
            return np.nan
        current_value = x.iloc[-1]
        return (x < current_value).sum() / len(x)

    return series.rolling(window).apply(percentile_rank, raw=False)


def detect_compression_events(df: pd.DataFrame,
                                threshold: float = 0.10,
                                atr_period: int = 14,
                                percentile_window: int = 150) -> pd.Series:
    """
    Detect when all three timeframes (5m, 15m, 30m) are in compression.

    Returns: Boolean series True when ALL timeframes in bottom threshold%
    """
    # 5m ATR
    atr_5m = calculate_atr(df, atr_period)
    atr_5m_pct = calculate_percentile_rank(atr_5m, percentile_window)

    # 15m ATR
    df_15m = df.resample('15min').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min',
        'Close': 'last', 'Volume': 'sum'
    }).dropna()
    atr_15m = calculate_atr(df_15m, atr_period)
    atr_15m_pct = calculate_percentile_rank(atr_15m, percentile_window)
    atr_15m_aligned = atr_15m_pct.reindex(df.index, method='ffill')

    # 30m ATR
    df_30m = df.resample('30min').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min',
        'Close': 'last', 'Volume': 'sum'
    }).dropna()
    atr_30m = calculate_atr(df_30m, atr_period)
    atr_30m_pct = calculate_percentile_rank(atr_30m, percentile_window)
    atr_30m_aligned = atr_30m_pct.reindex(df.index, method='ffill')

    # Compression: all three in bottom threshold%
    compression = (
        (atr_5m_pct < threshold) &
        (atr_15m_aligned < threshold) &
        (atr_30m_aligned < threshold)
    )

    return compression


def detect_breakouts(df: pd.DataFrame,
                     compression: pd.Series,
                     breakout_period: int = 20) -> pd.DataFrame:
    """
    Detect breakouts during compression periods.

    Returns: DataFrame with columns [timestamp, direction, entry_price]
    - direction: 'bullish' (break above 20-bar high) or 'bearish' (below 20-bar low)
    - entry_price: Close price of entry bar (bar AFTER breakout confirmation)
    """
    breakouts = []

    # Calculate rolling highs/lows
    rolling_high = df['High'].rolling(breakout_period).max()
    rolling_low = df['Low'].rolling(breakout_period).min()

    for i in range(breakout_period + 1, len(df) - 1):  # -1 to ensure entry bar exists
        # Check if in compression
        if not compression.iloc[i]:
            continue

        current_price = df['Close'].iloc[i]
        prev_high = rolling_high.iloc[i-1]  # Previous bar's 20-bar high
        prev_low = rolling_low.iloc[i-1]    # Previous bar's 20-bar low

        # Skip if NaN
        if pd.isna(prev_high) or pd.isna(prev_low):
            continue

        # Bullish breakout: price exceeds 20-bar high
        if current_price > prev_high:
            entry_bar = i + 1  # Enter on NEXT bar
            entry_price = df['Close'].iloc[entry_bar]

            breakouts.append({
                'timestamp': df.index[entry_bar],
                'entry_bar': entry_bar,
                'direction': 'bullish',
                'entry_price': entry_price
            })

        # Bearish breakout: price breaks below 20-bar low
        elif current_price < prev_low:
            entry_bar = i + 1
            entry_price = df['Close'].iloc[entry_bar]

            breakouts.append({
                'timestamp': df.index[entry_bar],
                'entry_bar': entry_bar,
                'direction': 'bearish',
                'entry_price': entry_price
            })

    return pd.DataFrame(breakouts)


def calculate_mae_mfe(df: pd.DataFrame,
                      entry_bar: int,
                      entry_price: float,
                      horizon: int) -> tuple:
    """
    Calculate MAE and MFE over forward horizon.

    Returns: (mfe_pct, mae_pct)
    - mfe_pct: Maximum Favorable Excursion as percentage
    - mae_pct: Maximum Adverse Excursion as percentage
    """
    # Ensure we have enough data
    if entry_bar + horizon >= len(df):
        return np.nan, np.nan

    # Future price range
    future_highs = df['High'].iloc[entry_bar:entry_bar + horizon]
    future_lows = df['Low'].iloc[entry_bar:entry_bar + horizon]

    # MFE: best possible gain (high - entry)
    mfe = (future_highs.max() - entry_price) / entry_price * 100

    # MAE: worst possible loss (low - entry)
    mae = (future_lows.min() - entry_price) / entry_price * 100

    return mfe, mae


def analyze_breakout_quality(df: pd.DataFrame,
                             breakouts: pd.DataFrame,
                             horizons: list) -> pd.DataFrame:
    """
    Analyze MAE/MFE quality for all breakouts across multiple horizons.

    Returns: DataFrame with columns:
    [timestamp, direction, entry_price, horizon, mfe, mae, ratio, favorable]
    """
    results = []

    for idx, breakout in breakouts.iterrows():
        entry_bar = breakout['entry_bar']
        entry_price = breakout['entry_price']
        direction = breakout['direction']

        for horizon in horizons:
            mfe, mae = calculate_mae_mfe(df, entry_bar, entry_price, horizon)

            # Skip if insufficient data
            if pd.isna(mfe) or pd.isna(mae):
                continue

            # Calculate ratio based on direction
            if direction == 'bullish':
                # For bullish: want MFE (upside) > |MAE| (downside)
                ratio = mfe / abs(mae) if mae != 0 else np.inf
                favorable = ratio >= 2.0
            else:
                # For bearish: want |MAE| (downside) > MFE (upside)
                ratio = abs(mae) / mfe if mfe != 0 else np.inf
                favorable = ratio >= 2.0

            results.append({
                'timestamp': breakout['timestamp'],
                'direction': direction,
                'entry_price': entry_price,
                'horizon': horizon,
                'mfe': mfe,
                'mae': mae,
                'ratio': ratio,
                'favorable': favorable
            })

    return pd.DataFrame(results)


def analyze_symbol(symbol: str,
                   csv_path: Path,
                   thresholds: list,
                   horizons: list,
                   n_bars: int = 100000) -> pd.DataFrame:
    """
    Complete analysis for one symbol across all thresholds and horizons.

    Returns: DataFrame with all breakout events
    """
    print(f"\n{'='*70}")
    print(f"ANALYZING {symbol}")
    print(f"{'='*70}")

    # Load data
    print(f"Loading last {n_bars:,} bars...")
    df = pd.read_csv(csv_path, skiprows=10)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low',
                            'close': 'Close', 'volume': 'Volume'})
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].tail(n_bars)

    print(f"  Loaded {len(df):,} bars from {df.index[0]} to {df.index[-1]}")

    # Analyze across all thresholds
    all_results = []

    for threshold in thresholds:
        print(f"\n  Compression threshold: {threshold*100:.0f}%")

        # Detect compression
        compression = detect_compression_events(df, threshold=threshold)
        n_compression = compression.sum()
        print(f"    Compression bars: {n_compression:,} ({n_compression/len(df)*100:.1f}%)")

        # Detect breakouts
        breakouts = detect_breakouts(df, compression)
        print(f"    Breakouts detected: {len(breakouts)}")

        if len(breakouts) == 0:
            print(f"    ⚠ No breakouts found at {threshold*100:.0f}% threshold")
            continue

        # Analyze quality
        results = analyze_breakout_quality(df, breakouts, horizons)
        results['symbol'] = symbol
        results['threshold'] = threshold

        all_results.append(results)

        # Quick stats
        if len(results) > 0:
            n_bullish = (results['direction'] == 'bullish').sum()
            n_bearish = (results['direction'] == 'bearish').sum()
            print(f"    Breakout events: {len(results)} ({n_bullish} bullish, {n_bearish} bearish)")

    if len(all_results) == 0:
        return pd.DataFrame()

    return pd.concat(all_results, ignore_index=True)


def generate_summary_table(results: pd.DataFrame) -> pd.DataFrame:
    """
    Generate statistical summary table.

    Columns: symbol, threshold, horizon, direction, n_events, median_ratio,
             pct_favorable, q25_ratio, q75_ratio
    """
    if len(results) == 0:
        return pd.DataFrame()

    summary = results.groupby(['symbol', 'threshold', 'horizon', 'direction']).agg({
        'ratio': ['count', 'median', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)],
        'favorable': 'mean'
    }).reset_index()

    # Flatten column names
    summary.columns = ['symbol', 'threshold', 'horizon', 'direction',
                       'n_events', 'median_ratio', 'q25_ratio', 'q75_ratio', 'pct_favorable']

    # Convert to percentage
    summary['pct_favorable'] = summary['pct_favorable'] * 100

    return summary


def create_visualizations(results: pd.DataFrame, summary: pd.DataFrame, output_dir: Path):
    """Generate research visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(results) == 0:
        print("  No data to visualize")
        return

    # 1. Ratio distribution histogram per symbol
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    symbols = results['symbol'].unique()

    for ax, symbol in zip(axes, symbols):
        symbol_data = results[results['symbol'] == symbol]['ratio']
        ax.hist(symbol_data[symbol_data < 10], bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(2.0, color='red', linestyle='--', linewidth=2, label='Threshold (2.0)')
        ax.axvline(symbol_data.median(), color='green', linestyle='-', linewidth=2,
                   label=f'Median ({symbol_data.median():.2f})')
        ax.set_xlabel('MFE/|MAE| Ratio')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{symbol} - Ratio Distribution')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'ratio_distributions.png', dpi=150)
    plt.close()
    print(f"  ✓ Saved: ratio_distributions.png")

    # 2. Success rate by horizon
    if len(summary) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))

        for symbol in summary['symbol'].unique():
            symbol_data = summary[summary['symbol'] == symbol]
            grouped = symbol_data.groupby('horizon')['pct_favorable'].mean()
            ax.plot(grouped.index, grouped.values, marker='o', linewidth=2, label=symbol)

        ax.axhline(50, color='gray', linestyle='--', label='Random (50%)')
        ax.set_xlabel('Forward Horizon (bars)')
        ax.set_ylabel('% Favorable (Ratio ≥ 2.0)')
        ax.set_title('Breakout Quality vs Forward Horizon')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'success_by_horizon.png', dpi=150)
        plt.close()
        print(f"  ✓ Saved: success_by_horizon.png")

    # 3. Threshold sensitivity
    if len(summary) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))

        for symbol in summary['symbol'].unique():
            symbol_data = summary[summary['symbol'] == symbol]
            grouped = symbol_data.groupby('threshold')['pct_favorable'].mean()
            ax.plot(grouped.index * 100, grouped.values, marker='o', linewidth=2, label=symbol)

        ax.axhline(50, color='gray', linestyle='--', label='Random (50%)')
        ax.set_xlabel('Compression Threshold (%)')
        ax.set_ylabel('% Favorable (Ratio ≥ 2.0)')
        ax.set_title('Breakout Quality vs Compression Threshold')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'success_by_threshold.png', dpi=150)
        plt.close()
        print(f"  ✓ Saved: success_by_threshold.png")


def main():
    """Run complete volatility compression research analysis."""

    print("="*70)
    print("VOLATILITY COMPRESSION BREAKOUT QUALITY RESEARCH")
    print("="*70)
    print("\nResearch Question:")
    print("  Does multi-timeframe volatility compression predict clean breakouts?")
    print("\nMethodology:")
    print("  • Compression: 5m/15m/30m ATR all in bottom N% (test 5/10/15/20%)")
    print("  • Breakouts: Price exceeds 20-bar high/low")
    print("  • Quality: MFE/|MAE| ratio over 10/20/30/50/100-bar horizons")
    print("  • Success: Ratio ≥ 2.0 (2x favorable vs adverse excursion)")
    print("\nSymbols: BTC, ETH, SOL")
    print("="*70)

    # Configuration - paths relative to script location
    SCRIPT_DIR = Path(__file__).parent
    RESEARCH_DIR = SCRIPT_DIR.parent
    PROJECT_ROOT = RESEARCH_DIR.parent.parent  # user_strategies/

    data_dir = PROJECT_ROOT / 'data' / 'raw' / 'crypto_5m'
    output_dir = RESEARCH_DIR / 'results' / 'phase_8_mae_mfe_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)

    symbols_config = [
        ('BTC', 'binance_spot_BTCUSDT-5m_20220101-20250930_v2.10.0.csv'),
        ('ETH', 'binance_spot_ETHUSDT-5m_20220101-20250930_v2.10.0.csv'),
        ('SOL', 'binance_spot_SOLUSDT-5m_20220101-20250930_v2.10.0.csv'),
    ]

    thresholds = [0.05, 0.10, 0.15, 0.20]  # 5%, 10%, 15%, 20%
    horizons = [10, 20, 30, 50, 100]       # bars

    # Analyze each symbol
    all_results = []

    for symbol, filename in symbols_config:
        csv_path = data_dir / filename

        if not csv_path.exists():
            print(f"\n⚠ WARNING: {filename} not found, skipping {symbol}")
            continue

        results = analyze_symbol(symbol, csv_path, thresholds, horizons, n_bars=100000)

        if len(results) > 0:
            all_results.append(results)

    # Combine all results
    if len(all_results) == 0:
        print("\n❌ No results generated. Exiting.")
        return

    combined_results = pd.concat(all_results, ignore_index=True)

    # Generate summary statistics
    print(f"\n{'='*70}")
    print("GENERATING STATISTICAL SUMMARIES")
    print(f"{'='*70}")

    summary = generate_summary_table(combined_results)

    # Save outputs
    print("\nSaving outputs...")

    # CSV: Raw data
    csv_file = output_dir / 'breakout_events_raw.csv'
    combined_results.to_csv(csv_file, index=False)
    print(f"  ✓ CSV (raw events): {csv_file}")

    # CSV: Summary table
    summary_file = output_dir / 'breakout_summary_statistics.csv'
    summary.to_csv(summary_file, index=False)
    print(f"  ✓ CSV (summary): {summary_file}")

    # Visualizations
    print("\nGenerating visualizations...")
    create_visualizations(combined_results, summary, output_dir)

    # Print key findings
    print(f"\n{'='*70}")
    print("KEY FINDINGS")
    print(f"{'='*70}\n")

    # Overall statistics
    total_events = len(combined_results)
    pct_favorable = (combined_results['favorable'].sum() / total_events * 100)

    print(f"Total breakout events analyzed: {total_events:,}")
    print(f"Overall favorable rate: {pct_favorable:.1f}%")
    print()

    # Per symbol
    for symbol in combined_results['symbol'].unique():
        symbol_data = combined_results[combined_results['symbol'] == symbol]
        n_events = len(symbol_data)
        pct_fav = symbol_data['favorable'].mean() * 100
        median_ratio = symbol_data['ratio'].median()

        print(f"{symbol}:")
        print(f"  Events: {n_events:,}")
        print(f"  Favorable rate: {pct_fav:.1f}%")
        print(f"  Median ratio: {median_ratio:.2f}")

        # Check statistical significance
        if n_events >= 30:
            if pct_fav >= 55:
                verdict = "✓ PREDICTIVE"
            elif pct_fav >= 45:
                verdict = "~ NEUTRAL"
            else:
                verdict = "✗ POOR"
        else:
            verdict = "⚠ INSUFFICIENT DATA (<30 events)"

        print(f"  Verdict: {verdict}")
        print()

    # Best configuration
    if len(summary) > 0:
        best_config = summary.loc[summary['pct_favorable'].idxmax()]
        print("Best configuration:")
        print(f"  Symbol: {best_config['symbol']}")
        print(f"  Threshold: {best_config['threshold']*100:.0f}%")
        print(f"  Horizon: {best_config['horizon']} bars")
        print(f"  Direction: {best_config['direction']}")
        print(f"  Success rate: {best_config['pct_favorable']:.1f}%")
        print(f"  Sample size: {int(best_config['n_events'])}")

    print(f"\n{'='*70}")
    print(f"Analysis complete. Results saved to: {output_dir}")
    print(f"{'='*70}")

    return combined_results, summary


if __name__ == '__main__':
    results, summary = main()
