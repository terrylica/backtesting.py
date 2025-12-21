# Crypto 5-Minute Trading Strategy Research Summary

**Status**: SUPERSEDED by Complete Termination Document
**Version**: 2.1.0 (SUPERSEDED)
**Date**: 2025-10-05
**Scope**: ETH 5-minute data (2022-2025, 394k bars, 3.75 years)
**Decision**: ABANDON crypto 5-minute trading research

**⚠️ This document is SUPERSEDED by:**
`CRYPTO_INTRADAY_RESEARCH_TERMINATION.md` (v1.0.0 FINAL)

**Scope expansion**: Research continued to Phase 16A-16B (higher timeframes), all failed.

---

## Executive Summary

**HARD STOP TRIGGERED**: All directional strategies fail on crypto 5-minute data.

**Research Effort**:

- **Phases executed**: 10 (Phases 8-15A)
- **Strategies tested**: 14 variations
- **Time invested**: 25+ hours
- **Best win rate**: 40.3% (MA 100/300)
- **Best return**: -99.36% (mean reversion compression - least catastrophic)
- **Viable strategies**: 0 / 14 (0%)

**CONCLUSION**: **Crypto 5-minute markets are fundamentally unsuitable for directional trading strategies.**

**DECISION**: **ABANDON crypto 5-minute trading. Recommended: Test 15-min/1-hour timeframes or traditional markets.**

---

## Complete Results Timeline

| Phase   | Strategy                            | Win Rate  | Return       | Trades  | Status                     |
| ------- | ----------------------------------- | --------- | ------------ | ------- | -------------------------- |
| 8       | Compression detection               | N/A       | N/A          | N/A     | Setup                      |
| 9       | Streak entropy                      | N/A       | N/A          | N/A     | Analysis                   |
| 10D     | Compression breakout                | 36.3%     | -100.00%     | 846     | Failed                     |
| 11      | Extended validation                 | ~30%      | -57.07%      | 10      | Failed                     |
| 12A     | Mean reversion (compression)        | 28.7%     | -99.36%      | 1,221   | Failed                     |
| 13A     | Compression + trend filter          | 39.7%     | -100.00%     | 863     | Failed                     |
| 14A     | Dual MA crossover (50/200)          | 30.2%     | -100.00%     | 358     | Failed                     |
| 14A     | **Parameter sweep (best: 100/300)** | **40.3%** | **-100.00%** | **139** | **Failed (best win rate)** |
| **15A** | **BB mean reversion**               | **35.7%** | **-100.00%** | **277** | **Failed (HARD STOP)**     |

### Performance Rankings (Best to Worst Win Rate)

1. **MA 100/300** (Phase 14A): 40.3% win rate, -100.00% return, 139 trades ← BEST WIN RATE
2. Compression + trend filter (Phase 13A): 39.7% win rate, -100.00% return, 863 trades
3. MA 50/100 (Phase 14A): 37.1% win rate, -100.00% return, 518 trades
4. Compression breakout (Phase 10D): 36.3% win rate, -100.00% return, 846 trades
5. **BB mean reversion (Phase 15A)**: 35.7% win rate, -100.00% return, 277 trades ← FINAL TEST
6. MA 50/300 (Phase 14A): 33.4% win rate, -99.99% return, 1,487 trades
7. MA 20/300 (Phase 14A): 33.2% win rate, -99.99% return, 431 trades
8. MA 100/200 (Phase 14A): 32.5% win rate, -99.99% return, 627 trades
9. MA 20/100 (Phase 14A): 31.0% win rate, -99.99% return, 509 trades
10. MA 20/200 (Phase 14A): 30.6% win rate, -99.99% return, 432 trades
11. MA 50/200 (Phase 14A): 30.2% win rate, -100.00% return, 358 trades
12. Mean reversion from compression (Phase 12A): 28.7% win rate, -99.36% return, 1,221 trades ← WORST

**Observation**: ALL 14 strategies have win rates <45% (worse than random 50%) and returns ≈ -100% (total capital loss).

---

## Key Findings

### Finding 1: Mean Reversion from Extremes Fails (Phase 15A) ⚠️ HARD STOP

**Hypothesis**: Crypto 5-minute exhibits strong mean reversion from Bollinger Band ±2σ extremes.

**Strategy**: Bollinger Band (20-period, 2σ) + RSI (14-period) confirmation

- Entry: Price touches BB ±2σ + RSI <30 or >70
- Exit: Mean reversion to BB middle, 2ATR stop, 100-bar time stop

**Result**: 35.7% win rate, -100.00% return, 277 trades

**Conclusion**: Mean reversion from statistical extremes does NOT work on crypto 5-minute.

**Hard Stop Triggered**:

- Win rate 35.7% < 50% (need random baseline)
- Return -100% ≤ 0% (need profitability)
- Decision: ABANDON crypto 5-minute trading

---

### Finding 2: Compression Research (Phases 8-13A)

**Hypothesis**: Volatility compression zones predict breakout direction.

**Approaches tested**:

1. Follow breakouts (Phase 10D): 36.3% win rate
2. Fade breakouts (Phase 12A): 28.7% win rate (WORSE)
3. Add trend filter (Phase 13A): 39.7% win rate (better but insufficient)

**Conclusion**: Compression zones are weak predictive signals regardless of direction or filters.

---

### Finding 2: Proven Trend Following (Phase 14A)

**Hypothesis**: Dual MA crossover (40+ years validation) will work on crypto.

**Approaches tested**: 8 MA combinations

- Fast MA: [20, 50, 100]
- Slow MA: [100, 200, 300]

**Best result**: MA 100/300 with 40.3% win rate, -100% return

**Conclusion**: Trend following DOES NOT work on crypto 5-minute data, regardless of parameters.

---

### Finding 3: Win Rate vs Return Paradox

**Observation**: Improving win rate does NOT guarantee positive returns.

**Evidence**:

- MA 100/300: 40.3% win rate, -100% return
- Compression + trend: 39.7% win rate, -100% return
- MA 50/100: 37.1% win rate, -100% return

**Explanation**:

- Winning trades: Small gains
- Losing trades: Hit stops or max drawdown
- Result: Losses exceed wins despite win rate improvement

**Implication**: Win rate is necessary but not sufficient. Need positive expectancy (avg_win × win_rate > avg_loss × loss_rate).

---

### Finding 4: Crypto 5-Minute Market Characteristics

**Hypothesis**: Crypto 5-minute markets are fundamentally different from traditional markets.

**Supporting Evidence**:

1. **Trend following fails**:
   - MA crossovers: 30-40% win rate (worse than random)
   - Expected on stocks: 40-45% win rate, positive returns
   - On crypto 5-min: All configurations lose 100%

2. **Mean reversion from compression fails**:
   - Fading breakouts: 28.7% win rate
   - Expected: If breakouts fail 70%, fading should win 70%
   - Reality: Fading also fails

3. **High-frequency characteristics**:
   - 5-minute bars: 288 per day
   - High volatility: Large intraday swings
   - Noise dominance: Signal-to-noise ratio very low

**Conclusion**: Crypto 5-minute markets may be dominated by:

- Market maker algorithms
- High-frequency trading noise
- Microstructure effects (bid-ask bounce, spread dynamics)
- Mean reversion at very short timescales (<1 hour)

---

## Decision Matrix

### Option A: Test Phase 15 (Mean Reversion from Extremes) ⭐ RECOMMENDED

**Rationale**:

- All trend-based strategies fail (compression + MA crossovers)
- Evidence suggests mean-reverting market
- Mean reversion from compression failed (28.7% win rate)
- But: Tested wrong entry signal (compression zones)
- Next: Test mean reversion from EXTREMES (Bollinger Bands, RSI)

**Strategy**: Bollinger Band (2σ) touch + RSI divergence

- Entry: Price touches BB ±2σ AND RSI shows divergence
- Exit: Mean reversion to BB midline
- Expected: 55-60% win rate if hypothesis correct

**Timeline**: 1 day (implementation + validation)
**Probability of success**: ~50%

**Risk**: If this also fails, crypto 5-min may be fundamentally untradeable with directional strategies.

---

### Option B: Change Timeframe to 15-Minute or Hourly ⚠️ SCOPE CHANGE

**Rationale**:

- 5-minute may be too short for signal persistence
- 15-minute or hourly may have better signal-to-noise
- Dual MA crossover validated on longer timeframes

**Test**: Run MA 50/200 on 15-minute or 1-hour data

**Timeline**: 2 hours (resample data + re-run)
**Probability of success**: ~60% (higher timeframe = better trends)

**Risk**: Out of scope for current 5-minute research focus.

---

### Option C: Change Asset Class to Traditional Markets ⚠️ SCOPE CHANGE

**Rationale**:

- Proven strategies work on stocks/futures
- Crypto may be fundamentally different
- Test same strategies on S&P 500 or forex

**Test**: Dual MA crossover on SPY or EUR/USD

**Timeline**: 1 day (new data source + validation)
**Probability of success**: ~80% (proven on traditional markets)

**Risk**: Abandons crypto trading focus entirely.

---

### Option D: Test Market Making (Bid-Ask Spread Capture) ⚠️ HIGH COMPLEXITY

**Rationale**:

- Directional strategies fail
- Market making works in choppy/mean-reverting markets
- Requires bid-ask spread data and inventory management

**Strategy**: Post limit orders on both sides, capture spread

**Timeline**: 3-5 days (implementation complexity)
**Probability of success**: ~70% (works in institutional crypto)

**Risk**: Requires exchange maker rebates, complex inventory risk management.

---

### Option E: Abandon Crypto 5-Minute Trading ⭐ PRUDENT

**Rationale**:

- 9 phases tested, ALL fail
- Best: 40.3% win rate, -100% return
- Evidence: Market structure incompatible with directional strategies
- Resource investment: 20+ hours with zero viable results

**Alternative Focus**:

1. Crypto daily/weekly trading (longer trends)
2. Traditional markets (stocks/futures) with proven strategies
3. Market making (institutional approach)
4. Portfolio/macro strategies (not HFT)

**Timeline**: Immediate
**Outcome**: Redirect effort to validated approaches

**Risk**: Sunk cost fallacy if crypto 5-min is actually tradeable.

---

## Recommendation

### Immediate Action: Test Option A (Phase 15) with Hard Stop

**Plan**:

1. Implement Phase 15: Mean Reversion from Bollinger Band Extremes
2. Hard stop criteria:
   - If win rate < 50%: ABANDON crypto 5-minute entirely
   - If return < 0%: ABANDON crypto 5-minute entirely
   - If both fail: Evidence conclusive that 5-min untradeable

**Timeline**: 1 day maximum

**Rationale**:

- One final test of mean reversion hypothesis
- If fails: 10 phases tested, all approaches exhausted
- If succeeds: Validates mean-reverting market hypothesis

---

### Long-Term Recommendations

**If Phase 15 Succeeds** (win rate >50%, return >0%):

1. Validate on BTC, SOL (cross-asset)
2. Add regime filters (volatility-adaptive)
3. Optimize parameters (BB period, RSI thresholds)
4. Walk-forward validation
5. Production deployment

**If Phase 15 Fails** (win rate <50% OR return <0%):

1. **ABANDON crypto 5-minute trading**
2. Redirect to:
   - Crypto daily/weekly (longer timeframe)
   - Traditional markets with proven strategies
   - Market making (institutional approach)

---

## Research Artifacts

### Compression Research Files

```
user_strategies/research/compression_breakout_research/
├── scripts/
│   ├── 08_rolling_window_regime_strategy.py        # Phase 10D
│   ├── 09_mean_reversion_strategy.py               # Phase 12A
│   └── 10_ensemble_trend_filter.py                 # Phase 13A
├── results/
│   └── phase_13_ensemble/
│       ├── PHASE_13A_GATE1_FAILURE_REPORT.md
│       └── phase_13a_trend_filter.csv
├── PHASE_12_MEAN_REVERSION_IMPLEMENTATION.md
└── PHASE_13_ENSEMBLE_IMPLEMENTATION.md
```

### Proven Strategies Files

```
user_strategies/research/proven_strategies/
├── scripts/
│   ├── 01_dual_ma_crossover.py                     # Phase 14A baseline
│   └── 01a_dual_ma_parameter_sweep.py              # Phase 14A sweep
├── results/
│   └── phase_14_trend_following/
│       ├── PHASE_14A_GATE1_FAILURE_REPORT.md
│       ├── phase_14a_baseline.csv
│       └── phase_14a_parameter_sweep.csv
└── PHASE_14_PROVEN_STRATEGIES_IMPLEMENTATION.md
```

### Summary Files

```
user_strategies/research/
└── CRYPTO_5MIN_RESEARCH_SUMMARY.md                 # This file
```

---

## Lessons Learned

### Technical Lessons

1. **Win rate ≠ Profitability**: Can improve win rate without improving returns
2. **Signal validation**: Compression zones are not predictive regardless of filters
3. **Parameter sensitivity**: MA combinations 20-300 all fail, not just 50/200
4. **Market structure matters**: Crypto 5-min behaves differently from traditional markets

### Process Lessons

1. **Decision gates work**: Gate criteria prevented continued investment in failing approaches
2. **Parameter sweeps valuable**: Quickly ruled out parameter optimization as solution
3. **Cross-validation critical**: Single config success doesn't guarantee universal viability
4. **Sunk cost awareness**: 9 phases is sufficient evidence to abandon approach

---

## Statistics

### Resource Investment

- **Phases**: 10 (Phase 8-15A)
- **Scripts written**: 15+
- **Strategies tested**: 14 variations
- **MA combinations**: 8
- **Total trades analyzed**: 7,000+
- **Time invested**: 25+ hours

### Performance Summary

- **Best win rate**: 40.3% (MA 100/300)
- **Worst win rate**: 28.7% (mean reversion compression)
- **Best return**: -99.36% (mean reversion compression - least catastrophic)
- **Average win rate**: ~33%
- **Average return**: ~-100%

### Conclusion Statistics

- **Strategies with positive return**: 0 / 14 (0%)
- **Strategies with >50% win rate**: 0 / 14 (0%)
- **Strategies with >45% win rate**: 0 / 14 (0%)
- **Strategies meeting baseline criteria**: 0 / 14 (0%)
- **Research outcome**: HARD STOP TRIGGERED at Phase 15A

---

## Version History

### v2.0.0 (2025-10-05) - FINAL

- Phase 15A complete: Bollinger Band mean reversion failed
- Hard stop triggered: Win rate 35.7%, return -100%
- Research CONCLUDED: 10 phases, 14 strategies, all failed
- Decision: ABANDON crypto 5-minute trading
- Recommendation: Test 15-min/1-hour or traditional markets

### v1.0.0 (2025-10-05)

- Initial summary after Phase 14A completion
- Compression research (Phases 8-13A) archived
- Trend following (Phase 14A) tested and failed
- Recommendation: Phase 15 (mean reversion from extremes) with hard stop

---

## References

**Archived Research** (ALL FAILED):

- Compression Breakout Research (Phases 8-13A): ABANDONED
- Proven Trend Following (Phase 14A): FAILED
- Mean Reversion from Extremes (Phase 15A): FAILED (HARD STOP)

**Final Status**:

- Research CONCLUDED: 10 phases, 14 strategies, 0 viable
- Decision: ABANDON crypto 5-minute trading
- Hard stop triggered at Phase 15A

**Recommended Next Steps**:

1. **Option A**: Test 15-minute or 1-hour crypto data (higher timeframe)
2. **Option B**: Test traditional markets (S&P 500, EUR/USD) with same strategies
3. **Option C**: Market making approach (institutional, complex)
4. **Option D**: Abandon HFT, focus on daily/weekly strategies

**Documentation**:

- Complete research summary: `CRYPTO_5MIN_RESEARCH_SUMMARY.md` (this file)
- Phase 15A failure report: `mean_reversion_extremes/results/phase_15_mean_reversion/PHASE_15A_HARD_STOP_FAILURE_REPORT.md`
- All phase implementations: `compression_breakout_research/`, `proven_strategies/`, `mean_reversion_extremes/`
