#!/usr/bin/env python3
"""
Streak Entropy Analysis: Discovering Hidden Structure in Breakout Sequences

RESEARCH QUESTION:
Do favorable/unfavorable breakouts cluster (low entropy) or scatter randomly (high entropy)?

HYPOTHESIS:
If streaks are NON-RANDOM (clustered), we've found hidden market regimes where
compression breakouts work/fail predictably. "Deriving order from chaos."

METHODOLOGY:
1. Convert 34,375 breakout events into chronological sequences (per symbol)
2. Identify streaks of consecutive favorable/unfavorable outcomes
3. Statistical tests:
   - Runs Test: Are sequences more clustered than random?
   - Streak distribution: Do we see longer streaks than chance predicts?
   - Baseline comparisons: vs Bernoulli random, shuffled data, binomial expectation
4. Regime detection: Map time periods as favorable/unfavorable clusters
5. Cross-symbol analysis: Do BTC/ETH/SOL enter regimes simultaneously?
6. Configuration ranking: Which threshold/horizon combos show most structure?

OUTPUT:
- Statistical tables with runs test p-values, entropy scores
- Streak length histograms vs random baselines
- Regime calendars showing clustered periods
- Cross-symbol synchronization analysis
- Ranked configurations by structure strength
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib
matplotlib.use('Agg')


def calculate_runs_test(sequence):
    """
    Wald-Wolfowitz Runs Test for randomness.

    H0: Sequence is random
    H1: Sequence has structure (clustering or alternating)

    Returns: (z_statistic, p_value, interpretation)
    """
    n = len(sequence)

    # Count runs (consecutive identical values)
    runs = 1
    for i in range(1, n):
        if sequence[i] != sequence[i-1]:
            runs += 1

    # Count favorable and unfavorable
    n_favorable = sum(sequence)
    n_unfavorable = n - n_favorable

    # Avoid division by zero
    if n_favorable == 0 or n_unfavorable == 0:
        return np.nan, np.nan, "All same outcome"

    # Expected number of runs under randomness
    expected_runs = (2 * n_favorable * n_unfavorable) / n + 1

    # Variance of runs
    var_runs = (2 * n_favorable * n_unfavorable * (2 * n_favorable * n_unfavorable - n)) / (n**2 * (n - 1))

    if var_runs == 0:
        return np.nan, np.nan, "Zero variance"

    # Z-statistic
    z = (runs - expected_runs) / np.sqrt(var_runs)

    # Two-tailed p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    # Interpretation
    if p_value < 0.05:
        if z < 0:
            interpretation = "CLUSTERED (too few runs)"
        else:
            interpretation = "ALTERNATING (too many runs)"
    else:
        interpretation = "RANDOM"

    return z, p_value, interpretation


def extract_streaks(sequence):
    """
    Extract all streaks from a binary sequence.

    Returns: list of (streak_type, streak_length) tuples
    - streak_type: 'favorable' or 'unfavorable'
    - streak_length: number of consecutive outcomes
    """
    if len(sequence) == 0:
        return []

    streaks = []
    current_value = sequence[0]
    current_length = 1

    for i in range(1, len(sequence)):
        if sequence[i] == current_value:
            current_length += 1
        else:
            # End of streak
            streak_type = 'favorable' if current_value else 'unfavorable'
            streaks.append((streak_type, current_length))
            current_value = sequence[i]
            current_length = 1

    # Don't forget last streak
    streak_type = 'favorable' if current_value else 'unfavorable'
    streaks.append((streak_type, current_length))

    return streaks


def calculate_streak_statistics(streaks):
    """Calculate summary statistics for streaks."""
    if len(streaks) == 0:
        return {}

    favorable_lengths = [length for type_, length in streaks if type_ == 'favorable']
    unfavorable_lengths = [length for type_, length in streaks if type_ == 'unfavorable']

    stats_dict = {
        'n_streaks': len(streaks),
        'n_favorable_streaks': len(favorable_lengths),
        'n_unfavorable_streaks': len(unfavorable_lengths),
        'mean_favorable_length': np.mean(favorable_lengths) if favorable_lengths else 0,
        'mean_unfavorable_length': np.mean(unfavorable_lengths) if unfavorable_lengths else 0,
        'max_favorable_length': max(favorable_lengths) if favorable_lengths else 0,
        'max_unfavorable_length': max(unfavorable_lengths) if unfavorable_lengths else 0,
        'median_favorable_length': np.median(favorable_lengths) if favorable_lengths else 0,
        'median_unfavorable_length': np.median(unfavorable_lengths) if unfavorable_lengths else 0,
    }

    # Balance metric: ratio of mean streak lengths
    if stats_dict['mean_unfavorable_length'] > 0:
        stats_dict['streak_balance'] = stats_dict['mean_favorable_length'] / stats_dict['mean_unfavorable_length']
    else:
        stats_dict['streak_balance'] = np.nan

    return stats_dict


def simulate_random_baseline(n_events, win_rate, n_simulations=1000):
    """
    Simulate random Bernoulli sequences with given win rate.

    Returns: distribution of max streak lengths under randomness
    """
    max_favorable_streaks = []
    max_unfavorable_streaks = []

    for _ in range(n_simulations):
        # Generate random sequence
        random_seq = np.random.random(n_events) < win_rate

        # Extract streaks
        streaks = extract_streaks(random_seq)

        favorable_lengths = [length for type_, length in streaks if type_ == 'favorable']
        unfavorable_lengths = [length for type_, length in streaks if type_ == 'unfavorable']

        max_favorable_streaks.append(max(favorable_lengths) if favorable_lengths else 0)
        max_unfavorable_streaks.append(max(unfavorable_lengths) if unfavorable_lengths else 0)

    return {
        'max_favorable_p95': np.percentile(max_favorable_streaks, 95),
        'max_unfavorable_p95': np.percentile(max_unfavorable_streaks, 95),
        'mean_max_favorable': np.mean(max_favorable_streaks),
        'mean_max_unfavorable': np.mean(max_unfavorable_streaks),
    }


def analyze_symbol_streaks(df, symbol):
    """
    Complete streak analysis for one symbol.

    df: Full breakout events dataframe
    symbol: BTC, ETH, or SOL

    Returns: dict with all streak statistics and tests
    """
    print(f"\n{'='*70}")
    print(f"ANALYZING STREAKS: {symbol}")
    print(f"{'='*70}")

    # Filter to symbol and sort chronologically
    symbol_df = df[df['symbol'] == symbol].copy()
    symbol_df = symbol_df.sort_values('timestamp')

    print(f"Total events: {len(symbol_df):,}")

    # Create binary sequence: True = favorable, False = unfavorable
    sequence = symbol_df['favorable'].values

    win_rate = sequence.mean()
    print(f"Favorable rate: {win_rate*100:.1f}%")

    # Runs test
    z_stat, p_value, interpretation = calculate_runs_test(sequence)
    print(f"\nRuns Test:")
    print(f"  Z-statistic: {z_stat:.3f}")
    print(f"  P-value: {p_value:.4f}")
    print(f"  Interpretation: {interpretation}")

    # Extract streaks
    streaks = extract_streaks(sequence)
    streak_stats = calculate_streak_statistics(streaks)

    print(f"\nStreak Statistics:")
    print(f"  Total streaks: {streak_stats['n_streaks']}")
    print(f"  Favorable streaks: {streak_stats['n_favorable_streaks']}")
    print(f"  Unfavorable streaks: {streak_stats['n_unfavorable_streaks']}")
    print(f"  Mean favorable length: {streak_stats['mean_favorable_length']:.2f}")
    print(f"  Mean unfavorable length: {streak_stats['mean_unfavorable_length']:.2f}")
    print(f"  Max favorable streak: {streak_stats['max_favorable_length']}")
    print(f"  Max unfavorable streak: {streak_stats['max_unfavorable_length']}")
    print(f"  Streak balance (F/U): {streak_stats['streak_balance']:.2f}")

    # Random baseline
    print(f"\nComparing to random baseline...")
    baseline = simulate_random_baseline(len(sequence), win_rate, n_simulations=1000)

    print(f"  Expected max favorable (95%): {baseline['max_favorable_p95']:.1f}")
    print(f"  Observed max favorable: {streak_stats['max_favorable_length']}")

    if streak_stats['max_favorable_length'] > baseline['max_favorable_p95']:
        print(f"  ✓ Observed exceeds random baseline (STRUCTURED)")
    else:
        print(f"  ✗ Within random range")

    print(f"\n  Expected max unfavorable (95%): {baseline['max_unfavorable_p95']:.1f}")
    print(f"  Observed max unfavorable: {streak_stats['max_unfavorable_length']}")

    if streak_stats['max_unfavorable_length'] > baseline['max_unfavorable_p95']:
        print(f"  ✓ Observed exceeds random baseline (STRUCTURED)")
    else:
        print(f"  ✗ Within random range")

    # Shuffle test (preserve win rate, randomize order)
    shuffled_seq = sequence.copy()
    np.random.shuffle(shuffled_seq)
    shuffled_streaks = extract_streaks(shuffled_seq)
    shuffled_stats = calculate_streak_statistics(shuffled_streaks)

    print(f"\nShuffled Data Comparison:")
    print(f"  Actual max unfavorable: {streak_stats['max_unfavorable_length']}")
    print(f"  Shuffled max unfavorable: {shuffled_stats['max_unfavorable_length']}")

    # Build result dictionary
    result = {
        'symbol': symbol,
        'n_events': len(sequence),
        'win_rate': win_rate,
        'runs_test_z': z_stat,
        'runs_test_p': p_value,
        'runs_interpretation': interpretation,
        **streak_stats,
        **{f'baseline_{k}': v for k, v in baseline.items()},
        'shuffled_max_unfavorable': shuffled_stats['max_unfavorable_length'],
        'sequence': sequence,
        'streaks': streaks,
        'timestamps': symbol_df['timestamp'].values,
    }

    return result


def create_regime_calendar(timestamps, sequence, streaks, symbol):
    """
    Create visual calendar showing favorable/unfavorable regimes.

    Identifies clusters of consecutive unfavorable outcomes.
    """
    # Convert to DataFrame for easy manipulation
    df = pd.DataFrame({
        'timestamp': pd.to_datetime(timestamps),
        'favorable': sequence
    })
    df = df.set_index('timestamp')

    # Mark streaks in the dataframe
    streak_id = 0
    streak_type_list = []
    streak_id_list = []

    for streak_type, streak_length in streaks:
        for _ in range(streak_length):
            streak_type_list.append(streak_type)
            streak_id_list.append(streak_id)
        streak_id += 1

    df['streak_type'] = streak_type_list
    df['streak_id'] = streak_id_list

    # Find significant unfavorable regimes (streaks ≥ 5)
    unfavorable_regimes = []

    for streak_id, group in df.groupby('streak_id'):
        if group['streak_type'].iloc[0] == 'unfavorable' and len(group) >= 5:
            unfavorable_regimes.append({
                'start': group.index.min(),
                'end': group.index.max(),
                'length': len(group),
                'symbol': symbol
            })

    print(f"\nSignificant Unfavorable Regimes (≥5 consecutive):")
    if len(unfavorable_regimes) == 0:
        print(f"  None found")
    else:
        for i, regime in enumerate(unfavorable_regimes, 1):
            print(f"  {i}. {regime['start']} to {regime['end']} ({regime['length']} events)")

    return df, unfavorable_regimes


def cross_symbol_regime_analysis(results_dict):
    """
    Test if BTC/ETH/SOL enter unfavorable regimes simultaneously.
    """
    print(f"\n{'='*70}")
    print(f"CROSS-SYMBOL REGIME SYNCHRONIZATION")
    print(f"{'='*70}")

    # Extract regime periods for each symbol
    symbols = ['BTC', 'ETH', 'SOL']
    all_regimes = []

    for symbol in symbols:
        if symbol not in results_dict:
            continue

        result = results_dict[symbol]
        _, regimes = create_regime_calendar(
            result['timestamps'],
            result['sequence'],
            result['streaks'],
            symbol
        )
        all_regimes.extend(regimes)

    if len(all_regimes) == 0:
        print("No significant regimes found for analysis")
        return None

    # Convert to DataFrame
    regime_df = pd.DataFrame(all_regimes)

    # Check for overlapping periods
    print(f"\nTotal significant regimes: {len(regime_df)}")
    print(f"  BTC: {len(regime_df[regime_df['symbol']=='BTC'])}")
    print(f"  ETH: {len(regime_df[regime_df['symbol']=='ETH'])}")
    print(f"  SOL: {len(regime_df[regime_df['symbol']=='SOL'])}")

    # Find simultaneous regimes (all 3 symbols in unfavorable regime at same time)
    # This is complex - simplify by checking if regime start dates cluster
    regime_df['start_date'] = pd.to_datetime(regime_df['start']).dt.date

    date_clusters = regime_df.groupby('start_date')['symbol'].apply(list)
    simultaneous = date_clusters[date_clusters.apply(lambda x: len(x) >= 2)]

    if len(simultaneous) > 0:
        print(f"\n✓ Found {len(simultaneous)} dates with multi-symbol regimes:")
        for date, symbols in simultaneous.items():
            print(f"  {date}: {', '.join(symbols)}")
    else:
        print(f"\n✗ No simultaneous multi-symbol regimes detected")

    return regime_df


def rank_configurations_by_structure(df):
    """
    Analyze entropy/structure for each [symbol, threshold, horizon, direction] combo.

    Rank configurations by how non-random their sequences are.
    """
    print(f"\n{'='*70}")
    print(f"RANKING CONFIGURATIONS BY STRUCTURE")
    print(f"{'='*70}")

    config_results = []

    # Group by configuration
    for (symbol, threshold, horizon, direction), group in df.groupby(['symbol', 'threshold', 'horizon', 'direction']):
        if len(group) < 30:  # Skip small samples
            continue

        # Sort chronologically
        group = group.sort_values('timestamp')
        sequence = group['favorable'].values

        # Runs test
        z_stat, p_value, interpretation = calculate_runs_test(sequence)

        # Streaks
        streaks = extract_streaks(sequence)
        streak_stats = calculate_streak_statistics(streaks)

        config_results.append({
            'symbol': symbol,
            'threshold': threshold,
            'horizon': horizon,
            'direction': direction,
            'n_events': len(sequence),
            'win_rate': sequence.mean(),
            'runs_p': p_value,
            'runs_z': z_stat,
            'max_unfavorable_streak': streak_stats['max_unfavorable_length'],
            'mean_unfavorable_streak': streak_stats['mean_unfavorable_length'],
            'streak_balance': streak_stats['streak_balance'],
            'is_clustered': (p_value < 0.05 and z_stat < 0) if not np.isnan(p_value) else False,
        })

    config_df = pd.DataFrame(config_results)

    # Rank by structure (lowest p-value = most non-random)
    config_df = config_df.sort_values('runs_p')

    print(f"\nTop 10 Most Structured Configurations (lowest runs test p-value):")
    print(f"{'Symbol':<6} {'Thresh':<6} {'Horiz':<6} {'Dir':<8} {'N':<6} {'P-val':<8} {'MaxStrk':<8} {'Interp'}")
    print(f"{'-'*70}")

    for idx, row in config_df.head(10).iterrows():
        interp = "CLUST" if row['is_clustered'] else "RAND"
        print(f"{row['symbol']:<6} {row['threshold']*100:>4.0f}% {row['horizon']:>5} {row['direction']:<8} "
              f"{row['n_events']:>5} {row['runs_p']:>7.4f} {row['max_unfavorable_streak']:>7} {interp}")

    return config_df


def create_visualizations(results_dict, output_dir):
    """Generate comprehensive visualizations."""

    # 1. Streak length distributions
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    symbols = ['BTC', 'ETH', 'SOL']

    for idx, symbol in enumerate(symbols):
        if symbol not in results_dict:
            continue

        result = results_dict[symbol]
        streaks = result['streaks']

        # Favorable streaks
        ax = axes[0, idx]
        favorable_lengths = [length for type_, length in streaks if type_ == 'favorable']
        if favorable_lengths:
            ax.hist(favorable_lengths, bins=range(1, max(favorable_lengths)+2), alpha=0.7, edgecolor='black', color='green')
            ax.axvline(result['baseline_max_favorable_p95'], color='red', linestyle='--', linewidth=2,
                       label=f"Random 95% ({result['baseline_max_favorable_p95']:.1f})")
            ax.axvline(result['max_favorable_length'], color='blue', linestyle='-', linewidth=2,
                       label=f"Observed ({result['max_favorable_length']})")
            ax.set_xlabel('Streak Length')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{symbol} - Favorable Streaks')
            ax.legend()
            ax.grid(alpha=0.3)

        # Unfavorable streaks
        ax = axes[1, idx]
        unfavorable_lengths = [length for type_, length in streaks if type_ == 'unfavorable']
        if unfavorable_lengths:
            ax.hist(unfavorable_lengths, bins=range(1, max(unfavorable_lengths)+2), alpha=0.7, edgecolor='black', color='red')
            ax.axvline(result['baseline_max_unfavorable_p95'], color='blue', linestyle='--', linewidth=2,
                       label=f"Random 95% ({result['baseline_max_unfavorable_p95']:.1f})")
            ax.axvline(result['max_unfavorable_length'], color='darkred', linestyle='-', linewidth=2,
                       label=f"Observed ({result['max_unfavorable_length']})")
            ax.set_xlabel('Streak Length')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{symbol} - Unfavorable Streaks')
            ax.legend()
            ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'streak_distributions.png', dpi=150)
    plt.close()
    print(f"  ✓ Saved: streak_distributions.png")

    # 2. Regime timeline
    fig, axes = plt.subplots(3, 1, figsize=(20, 12), sharex=True)

    for idx, symbol in enumerate(symbols):
        if symbol not in results_dict:
            continue

        result = results_dict[symbol]
        timestamps = pd.to_datetime(result['timestamps'])
        sequence = result['sequence']

        ax = axes[idx]

        # Plot as scatter: green = favorable, red = unfavorable
        favorable_times = timestamps[sequence == True]
        unfavorable_times = timestamps[sequence == False]

        ax.scatter(unfavorable_times, [1]*len(unfavorable_times), c='red', alpha=0.6, s=10, label='Unfavorable')
        ax.scatter(favorable_times, [1]*len(favorable_times), c='green', alpha=0.6, s=10, label='Favorable')

        # Highlight long unfavorable streaks
        streaks = result['streaks']
        event_idx = 0
        for streak_type, streak_length in streaks:
            if streak_type == 'unfavorable' and streak_length >= 5:
                start_time = timestamps[event_idx]
                end_time = timestamps[event_idx + streak_length - 1]
                ax.axvspan(start_time, end_time, alpha=0.2, color='red')
            event_idx += streak_length

        ax.set_ylabel(symbol, fontsize=12, fontweight='bold')
        ax.set_yticks([])
        ax.legend(loc='upper right')
        ax.grid(alpha=0.3)

        if idx == 0:
            ax.set_title('Regime Timeline: Favorable/Unfavorable Breakouts Over Time\n(Shaded = Unfavorable streaks ≥5)', fontsize=14)

    axes[-1].set_xlabel('Date', fontsize=12)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / 'regime_timeline.png', dpi=150)
    plt.close()
    print(f"  ✓ Saved: regime_timeline.png")


def main():
    """Run complete streak entropy analysis."""

    print("="*70)
    print("STREAK ENTROPY ANALYSIS")
    print("="*70)
    print("\nResearch Question:")
    print("  Do favorable/unfavorable outcomes cluster (low entropy) or scatter randomly?")
    print("\nHypothesis:")
    print("  If sequences show STRUCTURE (non-random clustering), we've found hidden")
    print("  market regimes where compression breakouts predictably work/fail.")
    print("\nMethodology:")
    print("  • Runs Test: Statistical test for sequence randomness")
    print("  • Streak analysis: Length distributions vs random baselines")
    print("  • Regime detection: Identify clustered periods")
    print("  • Cross-symbol sync: Test if regimes align across BTC/ETH/SOL")
    print("="*70)

    # Configuration - paths relative to script location
    SCRIPT_DIR = Path(__file__).parent
    RESEARCH_DIR = SCRIPT_DIR.parent

    # Load existing data from Phase 8 results
    data_file = RESEARCH_DIR / 'results' / 'phase_8_mae_mfe_analysis' / 'breakout_events_raw.csv'
    output_dir = RESEARCH_DIR / 'results' / 'phase_9_streak_entropy_breakthrough'

    if not data_file.exists():
        print(f"\n❌ ERROR: {data_file} not found")
        print("Run 03_mae_mfe_analysis.py first to generate Phase 8 data.")
        return

    print(f"\nLoading data from {data_file}...")
    df = pd.read_csv(data_file)
    print(f"  Loaded {len(df):,} breakout events")

    # Analyze each symbol
    results_dict = {}

    for symbol in ['BTC', 'ETH', 'SOL']:
        result = analyze_symbol_streaks(df, symbol)
        results_dict[symbol] = result

    # Cross-symbol analysis
    regime_df = cross_symbol_regime_analysis(results_dict)

    # Configuration ranking
    config_df = rank_configurations_by_structure(df)

    # Save outputs
    print(f"\n{'='*70}")
    print("SAVING OUTPUTS")
    print(f"{'='*70}")

    # Summary table
    summary_rows = []
    for symbol, result in results_dict.items():
        summary_rows.append({
            'symbol': symbol,
            'n_events': result['n_events'],
            'win_rate_pct': result['win_rate'] * 100,
            'runs_test_z': result['runs_test_z'],
            'runs_test_p': result['runs_test_p'],
            'interpretation': result['runs_interpretation'],
            'mean_favorable_streak': result['mean_favorable_length'],
            'mean_unfavorable_streak': result['mean_unfavorable_length'],
            'max_favorable_streak': result['max_favorable_length'],
            'max_unfavorable_streak': result['max_unfavorable_length'],
            'streak_balance_ratio': result['streak_balance'],
            'random_baseline_max_unfav_95pct': result['baseline_max_unfavorable_p95'],
            'exceeds_baseline': 'YES' if result['max_unfavorable_length'] > result['baseline_max_unfavorable_p95'] else 'NO',
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_file = output_dir / 'streak_analysis_summary.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"  ✓ CSV: {summary_file}")

    # Configuration rankings
    config_file = output_dir / 'configuration_entropy_rankings.csv'
    config_df.to_csv(config_file, index=False)
    print(f"  ✓ CSV: {config_file}")

    # Regime periods
    if regime_df is not None and len(regime_df) > 0:
        regime_file = output_dir / 'unfavorable_regimes.csv'
        regime_df.to_csv(regime_file, index=False)
        print(f"  ✓ CSV: {regime_file}")

    # Visualizations
    print("\nGenerating visualizations...")
    create_visualizations(results_dict, output_dir)

    # Final summary
    print(f"\n{'='*70}")
    print("KEY FINDINGS")
    print(f"{'='*70}\n")

    for symbol, result in results_dict.items():
        print(f"{symbol}:")
        print(f"  Runs Test: {result['runs_interpretation']}")
        print(f"  P-value: {result['runs_test_p']:.4f}")
        print(f"  Max unfavorable streak: {result['max_unfavorable_length']}")
        print(f"  Random baseline (95%): {result['baseline_max_unfavorable_p95']:.1f}")

        if result['max_unfavorable_length'] > result['baseline_max_unfavorable_p95']:
            print(f"  ✓ STRUCTURE DETECTED: Observed streak exceeds random expectation")
        else:
            print(f"  ✗ Within random range")

        print(f"  Streak balance (Fav/Unfav): {result['streak_balance']:.2f}")
        print()

    # Overall verdict
    clustered_count = sum(1 for r in results_dict.values()
                          if 'CLUSTERED' in r['runs_interpretation'])

    if clustered_count >= 2:
        print("VERDICT: ✓ STRUCTURE FOUND")
        print("  Multiple symbols show non-random clustering of favorable/unfavorable outcomes.")
        print("  This suggests market regimes exist where compression breakouts predictably fail.")
        print("\nIMPLICATION:")
        print("  → Can use streak detection to AVOID trading during unfavorable regimes")
        print("  → Monitor current streak length as regime indicator")
        print("  → Wait for regime flip (unfavorable → favorable) before entering trades")
    else:
        print("VERDICT: ✗ NO CONSISTENT STRUCTURE")
        print("  Sequences appear random (high entropy).")
        print("  Favorable/unfavorable outcomes are unpredictably scattered.")
        print("\nIMPLICATION:")
        print("  → No hidden regimes to exploit")
        print("  → 33% favorable rate is uniform, not clustered")
        print("  → Further confirms compression breakouts are not viable")

    print(f"\n{'='*70}")
    print(f"Analysis complete. Results saved to: {output_dir}")
    print(f"{'='*70}")

    return results_dict, config_df, regime_df


if __name__ == '__main__':
    results, configs, regimes = main()
