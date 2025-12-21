# Phase 10: Regime-Aware Trading Validation Results

**Date:** 2025-10-04 15:52:51
**Regime Threshold:** ≥5 consecutive unfavorable outcomes

---

## Executive Summary

### Hypothesis

Skipping trades during unfavorable streaks (≥5 consecutive) improves win rate from 33% → 50%+

### Results

#### Train Period (Oct 2024 - Apr 2025)

- **Baseline win rate:** 35.60%
- **Filtered win rate:** 53.99%
- **Improvement:** +18.39 percentage points
- **Chi-square p-value:** 0.000000 ✓ SIGNIFICANT
- **Binomial p-value (>50%):** 0.000141 ✓ SIGNIFICANT

#### Test Period (May 2025 - Sep 2025)

- **Baseline win rate:** 33.19%
- **Filtered win rate:** 58.47%
- **Improvement:** +25.27 percentage points
- **Chi-square p-value:** 0.000000 ✓ SIGNIFICANT
- **Binomial p-value (>50%):** 0.000000 ✓ SIGNIFICANT

---

## Conclusion

✅ **HYPOTHESIS VALIDATED**

Regime filtering successfully improves win rate to >50% with statistical significance in both train and test periods.

## Per-Symbol Results

### Train Period

period symbol n_events baseline_win_rate filtered_win_rate improvement_pct trades_taken trades_skipped pct_skipped
TRAIN BTC 1461 34.976044 55.802469 20.826425 810 651 44.558522
TRAIN ETH 907 36.383682 52.120141 15.736459 566 341 37.596472
TRAIN SOL 1199 35.779817 53.426573 17.646757 715 484 40.366972

### Test Period

period symbol n_events baseline_win_rate filtered_win_rate improvement_pct trades_taken trades_skipped pct_skipped
TEST BTC 1460 36.027397 60.101652 24.074255 787 673 46.095890
TEST ETH 756 34.126984 54.892601 20.765617 419 337 44.576720
TEST SOL 1092 28.754579 58.909853 30.155274 477 615 56.318681
