#!/usr/bin/env python3
"""
Phase 10: Regime-Aware Trading Validation

HYPOTHESIS:
Skipping trades during unfavorable streaks (≥5 consecutive) improves win rate from 33% → 50%+

METHODOLOGY:
1. Load 34,375 breakout events from Phase 8 (with favorable/unfavorable outcomes)
2. Per-symbol chronological simulation:
   - Track rolling streak of favorable/unfavorable outcomes
   - LOOKAHEAD PREVENTION: Use only PAST outcomes to decide CURRENT trade
   - Skip trade when unfavorable_streak >= threshold (default: 5)
3. Calculate win rates: baseline (all trades) vs regime-filtered (skipped trades removed)
4. Train/test split: Oct 2024-Apr 2025 (train) / May 2025-Sep 2025 (test)
5. Statistical significance: Chi-square test, binomial test

SUCCESS CRITERIA:
- Win rate improvement: 33% → 50%+ (crosses random threshold)
- Statistical significance: P < 0.05
- Consistent improvement across train/test periods
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def load_phase8_data(csv_path: Path) -> pd.DataFrame:
    """
    Load Phase 8 breakout events and deduplicate.

    Each breakout appears 5 times (one per horizon). We'll use horizon=50 as canonical.
    """
    print("="*70)
    print("LOADING PHASE 8 DATA")
    print("="*70)

    df = pd.read_csv(csv_path)
    print(f"Total rows: {len(df):,}")

    # Filter to horizon=50 (middle of range) to get unique breakouts
    df_unique = df[df['horizon'] == 50].copy()
    print(f"Unique breakouts (horizon=50): {len(df_unique):,}")

    # Convert timestamp to datetime
    df_unique['timestamp'] = pd.to_datetime(df_unique['timestamp'])

    # Sort chronologically (critical for temporal integrity)
    df_unique = df_unique.sort_values('timestamp').reset_index(drop=True)

    print(f"Date range: {df_unique['timestamp'].min()} to {df_unique['timestamp'].max()}")
    print(f"\nPer-symbol counts:")
    for symbol in df_unique['symbol'].unique():
        count = len(df_unique[df_unique['symbol'] == symbol])
        fav_rate = df_unique[df_unique['symbol'] == symbol]['favorable'].mean() * 100
        print(f"  {symbol}: {count:,} events, {fav_rate:.1f}% favorable")

    return df_unique


def simulate_regime_aware_trading(df: pd.DataFrame,
                                   symbol: str,
                                   regime_threshold: int = 5,
                                   verbose: bool = True) -> dict:
    """
    Simulate regime-aware trading for one symbol.

    LOOKAHEAD PREVENTION:
    - Decision to trade event[i] uses only outcomes from events[0:i-1]
    - Current event's outcome is NOT known when making trade decision

    Returns: dict with baseline/filtered win rates, trades taken/skipped
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"SIMULATING: {symbol}")
        print(f"{'='*70}")

    # Filter to symbol
    symbol_df = df[df['symbol'] == symbol].copy()
    symbol_df = symbol_df.sort_values('timestamp').reset_index(drop=True)

    n_events = len(symbol_df)

    # Track regime state
    unfavorable_streak = 0
    favorable_streak = 0

    # Track decisions
    trades_taken = []
    trades_skipped = []

    for idx, row in symbol_df.iterrows():
        # DECISION POINT: Should we take this trade?
        # We can ONLY use outcomes from previous events (idx 0 to idx-1)

        skip_trade = (unfavorable_streak >= regime_threshold)

        if skip_trade:
            trades_skipped.append(idx)
        else:
            trades_taken.append(idx)

        # AFTER making decision, observe outcome and update streaks
        outcome = row['favorable']

        if outcome:
            favorable_streak += 1
            unfavorable_streak = 0
        else:
            unfavorable_streak += 1
            favorable_streak = 0

    # Calculate metrics
    baseline_outcomes = symbol_df['favorable'].values
    filtered_outcomes = symbol_df.iloc[trades_taken]['favorable'].values
    skipped_outcomes = symbol_df.iloc[trades_skipped]['favorable'].values if trades_skipped else np.array([])

    baseline_win_rate = baseline_outcomes.mean() * 100
    filtered_win_rate = filtered_outcomes.mean() * 100 if len(filtered_outcomes) > 0 else 0
    skipped_win_rate = skipped_outcomes.mean() * 100 if len(skipped_outcomes) > 0 else 0

    if verbose:
        print(f"\nBaseline (all trades):")
        print(f"  Total trades: {n_events:,}")
        print(f"  Win rate: {baseline_win_rate:.2f}%")

        print(f"\nRegime-filtered (threshold={regime_threshold}):")
        print(f"  Trades taken: {len(trades_taken):,} ({len(trades_taken)/n_events*100:.1f}%)")
        print(f"  Trades skipped: {len(trades_skipped):,} ({len(trades_skipped)/n_events*100:.1f}%)")
        print(f"  Win rate (taken): {filtered_win_rate:.2f}%")
        print(f"  Win rate (skipped): {skipped_win_rate:.2f}%")
        print(f"  Improvement: {filtered_win_rate - baseline_win_rate:+.2f} percentage points")

    return {
        'symbol': symbol,
        'n_events': n_events,
        'baseline_win_rate': baseline_win_rate,
        'filtered_win_rate': filtered_win_rate,
        'skipped_win_rate': skipped_win_rate,
        'trades_taken': len(trades_taken),
        'trades_skipped': len(trades_skipped),
        'improvement_pct': filtered_win_rate - baseline_win_rate,
        'baseline_outcomes': baseline_outcomes,
        'filtered_outcomes': filtered_outcomes,
        'skipped_outcomes': skipped_outcomes,
    }


def calculate_statistical_significance(baseline_outcomes, filtered_outcomes):
    """
    Test if improvement is statistically significant.

    Tests:
    1. Chi-square test: Are win rates significantly different?
    2. Binomial test: Is filtered win rate > 50% (better than random)?
    """
    # Chi-square test
    baseline_wins = baseline_outcomes.sum()
    baseline_losses = len(baseline_outcomes) - baseline_wins

    filtered_wins = filtered_outcomes.sum()
    filtered_losses = len(filtered_outcomes) - filtered_wins

    contingency_table = np.array([
        [baseline_wins, baseline_losses],
        [filtered_wins, filtered_losses]
    ])

    chi2, p_chi2, dof, expected = stats.chi2_contingency(contingency_table)

    # Binomial test (is filtered win rate significantly > 50%?)
    binomial_result = stats.binomtest(filtered_wins, len(filtered_outcomes), 0.5, alternative='greater')
    p_binomial = binomial_result.pvalue

    return {
        'chi2_statistic': chi2,
        'chi2_p_value': p_chi2,
        'binomial_p_value': p_binomial,
        'chi2_significant': p_chi2 < 0.05,
        'binomial_significant': p_binomial < 0.05,
    }


def train_test_split_validation(df: pd.DataFrame,
                                 regime_threshold: int = 5,
                                 split_date: str = '2025-05-01') -> dict:
    """
    Validate regime effect on train/test periods.

    Train: Oct 2024 - Apr 2025
    Test: May 2025 - Sep 2025
    """
    print(f"\n{'='*70}")
    print(f"TRAIN/TEST SPLIT VALIDATION")
    print(f"{'='*70}")

    split_dt = pd.to_datetime(split_date)

    train_df = df[df['timestamp'] < split_dt].copy()
    test_df = df[df['timestamp'] >= split_dt].copy()

    print(f"\nTrain period: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
    print(f"  Events: {len(train_df):,}")

    print(f"\nTest period: {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")
    print(f"  Events: {len(test_df):,}")

    # Simulate on both periods
    results = {'train': {}, 'test': {}}

    for period_name, period_df in [('train', train_df), ('test', test_df)]:
        print(f"\n{'='*70}")
        print(f"{period_name.upper()} PERIOD RESULTS")
        print(f"{'='*70}")

        period_results = {}
        for symbol in ['BTC', 'ETH', 'SOL']:
            result = simulate_regime_aware_trading(
                period_df, symbol, regime_threshold, verbose=True
            )
            period_results[symbol] = result

        # Aggregate
        total_baseline = np.concatenate([r['baseline_outcomes'] for r in period_results.values()])
        total_filtered = np.concatenate([r['filtered_outcomes'] for r in period_results.values()])

        aggregate_baseline = total_baseline.mean() * 100
        aggregate_filtered = total_filtered.mean() * 100

        print(f"\n{period_name.upper()} AGGREGATE:")
        print(f"  Baseline win rate: {aggregate_baseline:.2f}%")
        print(f"  Filtered win rate: {aggregate_filtered:.2f}%")
        print(f"  Improvement: {aggregate_filtered - aggregate_baseline:+.2f} percentage points")

        # Statistical significance
        stats_result = calculate_statistical_significance(total_baseline, total_filtered)
        print(f"\n  Statistical Tests:")
        print(f"    Chi-square p-value: {stats_result['chi2_p_value']:.6f} {'✓ SIGNIFICANT' if stats_result['chi2_significant'] else '✗ Not significant'}")
        print(f"    Binomial p-value (>50%): {stats_result['binomial_p_value']:.6f} {'✓ SIGNIFICANT' if stats_result['binomial_significant'] else '✗ Not significant'}")

        results[period_name] = {
            'per_symbol': period_results,
            'aggregate_baseline': aggregate_baseline,
            'aggregate_filtered': aggregate_filtered,
            'improvement': aggregate_filtered - aggregate_baseline,
            'statistics': stats_result,
        }

    return results


def generate_summary_report(results: dict, output_dir: Path):
    """Generate comprehensive summary report."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"GENERATING SUMMARY REPORT")
    print(f"{'='*70}")

    # Summary table
    rows = []

    for period in ['train', 'test']:
        for symbol, result in results[period]['per_symbol'].items():
            rows.append({
                'period': period.upper(),
                'symbol': symbol,
                'n_events': result['n_events'],
                'baseline_win_rate': result['baseline_win_rate'],
                'filtered_win_rate': result['filtered_win_rate'],
                'improvement_pct': result['improvement_pct'],
                'trades_taken': result['trades_taken'],
                'trades_skipped': result['trades_skipped'],
                'pct_skipped': result['trades_skipped'] / result['n_events'] * 100,
            })

    summary_df = pd.DataFrame(rows)

    # Save CSV
    csv_path = output_dir / 'regime_validation_summary.csv'
    summary_df.to_csv(csv_path, index=False)
    print(f"  ✓ Saved: {csv_path}")

    # Aggregate statistics
    agg_rows = []
    for period in ['train', 'test']:
        agg_rows.append({
            'period': period.upper(),
            'baseline_win_rate': results[period]['aggregate_baseline'],
            'filtered_win_rate': results[period]['aggregate_filtered'],
            'improvement_pct': results[period]['improvement'],
            'chi2_p_value': results[period]['statistics']['chi2_p_value'],
            'binomial_p_value': results[period]['statistics']['binomial_p_value'],
            'chi2_significant': results[period]['statistics']['chi2_significant'],
            'binomial_significant': results[period]['statistics']['binomial_significant'],
        })

    agg_df = pd.DataFrame(agg_rows)
    agg_path = output_dir / 'aggregate_statistics.csv'
    agg_df.to_csv(agg_path, index=False)
    print(f"  ✓ Saved: {agg_path}")

    # Markdown report
    report = f"""# Phase 10: Regime-Aware Trading Validation Results

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Regime Threshold:** ≥5 consecutive unfavorable outcomes

---

## Executive Summary

### Hypothesis
Skipping trades during unfavorable streaks (≥5 consecutive) improves win rate from 33% → 50%+

### Results

#### Train Period (Oct 2024 - Apr 2025)
- **Baseline win rate:** {results['train']['aggregate_baseline']:.2f}%
- **Filtered win rate:** {results['train']['aggregate_filtered']:.2f}%
- **Improvement:** {results['train']['improvement']:+.2f} percentage points
- **Chi-square p-value:** {results['train']['statistics']['chi2_p_value']:.6f} {'✓ SIGNIFICANT' if results['train']['statistics']['chi2_significant'] else '✗ Not significant'}
- **Binomial p-value (>50%):** {results['train']['statistics']['binomial_p_value']:.6f} {'✓ SIGNIFICANT' if results['train']['statistics']['binomial_significant'] else '✗ Not significant'}

#### Test Period (May 2025 - Sep 2025)
- **Baseline win rate:** {results['test']['aggregate_baseline']:.2f}%
- **Filtered win rate:** {results['test']['aggregate_filtered']:.2f}%
- **Improvement:** {results['test']['improvement']:+.2f} percentage points
- **Chi-square p-value:** {results['test']['statistics']['chi2_p_value']:.6f} {'✓ SIGNIFICANT' if results['test']['statistics']['chi2_significant'] else '✗ Not significant'}
- **Binomial p-value (>50%):** {results['test']['statistics']['binomial_p_value']:.6f} {'✓ SIGNIFICANT' if results['test']['statistics']['binomial_significant'] else '✗ Not significant'}

---

## Conclusion

"""

    # Determine verdict
    train_success = results['train']['aggregate_filtered'] >= 50.0
    test_success = results['test']['aggregate_filtered'] >= 50.0
    train_sig = results['train']['statistics']['chi2_significant']
    test_sig = results['test']['statistics']['chi2_significant']

    if train_success and test_success and train_sig and test_sig:
        verdict = "✅ **HYPOTHESIS VALIDATED**"
        explanation = "Regime filtering successfully improves win rate to >50% with statistical significance in both train and test periods."
    elif train_success and train_sig:
        verdict = "⚠️ **PARTIAL VALIDATION**"
        explanation = "Regime filtering works in train period but does not generalize to test period. Possible overfitting or regime shift."
    else:
        verdict = "❌ **HYPOTHESIS REJECTED**"
        explanation = "Regime filtering does not improve win rate to >50%. The clustering observed in Phase 9 does not translate to exploitable trading edges."

    report += f"{verdict}\n\n{explanation}\n\n"

    # Per-symbol breakdown
    report += "## Per-Symbol Results\n\n"
    report += "### Train Period\n\n"
    train_table = summary_df[summary_df['period'] == 'TRAIN']
    report += train_table.to_string(index=False)
    report += "\n\n### Test Period\n\n"
    test_table = summary_df[summary_df['period'] == 'TEST']
    report += test_table.to_string(index=False)

    # Save markdown
    md_path = output_dir / 'REGIME_VALIDATION_REPORT.md'
    with open(md_path, 'w') as f:
        f.write(report)
    print(f"  ✓ Saved: {md_path}")

    return summary_df, agg_df


def main():
    """Run complete regime validation analysis."""

    print("="*70)
    print("PHASE 10: REGIME-AWARE TRADING VALIDATION")
    print("="*70)
    print("\nHypothesis:")
    print("  Skipping trades during unfavorable streaks (≥5) improves win rate 33% → 50%+")
    print("\nMethodology:")
    print("  • Load 34,375 breakout events from Phase 8")
    print("  • Simulate regime-aware trading (per-symbol, lookahead prevention)")
    print("  • Train/test split: Oct 2024-Apr 2025 / May 2025-Sep 2025")
    print("  • Statistical tests: Chi-square, binomial")
    print("\nSuccess Criteria:")
    print("  • Win rate ≥50% (crosses random threshold)")
    print("  • Statistical significance P < 0.05")
    print("  • Consistent improvement across train/test")
    print("="*70)

    # Paths
    phase8_data = Path('user_strategies/research/compression_breakout_research/results/phase_8_mae_mfe_analysis/breakout_events_raw.csv')
    output_dir = Path('/tmp/regime_validation_results')

    # Load data
    df = load_phase8_data(phase8_data)

    # Run train/test validation
    results = train_test_split_validation(df, regime_threshold=5, split_date='2025-05-01')

    # Generate report
    summary_df, agg_df = generate_summary_report(results, output_dir)

    # Final verdict
    print(f"\n{'='*70}")
    print("FINAL VERDICT")
    print(f"{'='*70}\n")

    train_filtered = results['train']['aggregate_filtered']
    test_filtered = results['test']['aggregate_filtered']

    print(f"Train period filtered win rate: {train_filtered:.2f}%")
    print(f"Test period filtered win rate: {test_filtered:.2f}%")

    if train_filtered >= 50.0 and test_filtered >= 50.0:
        print("\n✅ SUCCESS: Regime filtering achieves >50% win rate in both periods")
        print("   → Hypothesis VALIDATED")
        print("   → Ready for full backtesting.py strategy implementation")
    elif train_filtered >= 50.0:
        print("\n⚠️ PARTIAL: Works in train but not test period")
        print("   → Possible regime shift or overfitting")
        print("   → Investigate test period characteristics before proceeding")
    else:
        print("\n❌ FAILURE: Does not achieve 50% win rate")
        print("   → Hypothesis REJECTED")
        print("   → Clustering does not translate to exploitable edge")

    print(f"\n{'='*70}")
    print(f"Analysis complete. Results saved to: {output_dir}")
    print(f"{'='*70}")

    return results


if __name__ == '__main__':
    results = main()
