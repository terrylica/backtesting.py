# Phase 16: Timeframe Change Implementation (15-Minute & 1-Hour)

**Status**: COMPLETED - Both Gates Failed
**Version**: 1.1.0 (FINAL)
**Date**: 2025-10-05
**Prerequisites**: Phase 15A hard stop triggered, crypto 5-minute trading abandoned
**Outcome**: Higher timeframes performed WORSE than 5-minute (counter-intuitive finding)
**Rationale**: Test if higher timeframes (15-min, 1-hour) provide better signal-to-noise ratio

---

## Objective

Validate if crypto directional strategies work on 15-minute and 1-hour timeframes after universal failure on 5-minute data.

**Hypothesis**: Higher timeframes exhibit lower noise and better trend persistence, enabling profitable directional strategies.

**Success Criteria**: Win rate ≥45% AND return >0% on at least one timeframe.

---

## Service Level Objectives (SLOs)

### Availability

- Script execution success rate: ≥99%
- Data resampling success rate: 100%
- Indicator calculation availability: 100%

### Correctness

- Resampling accuracy: 100% (OHLC aggregation validated)
- Indicator calculation precision: ±0.01%
- Trade execution accuracy: 100% (no phantom trades)
- Temporal integrity: Zero lookahead bias tolerance

### Observability

- Logging coverage: 100% of resampling operations
- Trade-by-trade audit trail: Required
- Timeframe comparison: Side-by-side metrics
- Bar count validation: Log before/after resampling

### Maintainability

- Code reuse: ≥80% from Phase 14A and 15A
- Out-of-the-box resampling: pandas.resample() only
- Inline documentation: Resampling logic
- Version tracking: Git commit on completion

**Excluded**: Performance optimization, security hardening

---

## Theoretical Foundation

### Timeframe and Signal Quality

**Principle**: Higher timeframes reduce noise, improve signal-to-noise ratio

**Academic Support**:

- "Market Microstructure" (O'Hara, 1995): Intraday noise decreases with aggregation
- "Evidence of Market Inefficiency" (Lo & MacKinlay, 1988): Longer horizons exhibit autocorrelation
- Practitioner consensus: Daily > hourly > 5-minute for trend strategies

**Expected Improvements**:

- **15-minute**: 3× aggregation → 3× noise reduction (expected)
- **1-hour**: 12× aggregation → 12× noise reduction (expected)
- **Trend persistence**: Longer timeframes → trends last multiple bars
- **Whipsaw reduction**: Fewer false signals from HFT/market maker noise

---

## Architecture

### Data Resampling (Out-of-the-Box)

**Pandas resample() method**:

```python
# 5-minute → 15-minute (3:1 aggregation)
df_15m = df_5m.resample('15min').agg({
    'Open': 'first',   # First 5-min bar's open
    'High': 'max',     # Highest high across 3 bars
    'Low': 'min',      # Lowest low across 3 bars
    'Close': 'last',   # Last 5-min bar's close
    'Volume': 'sum'    # Total volume across 3 bars
}).dropna()

# 5-minute → 1-hour (12:1 aggregation)
df_1h = df_5m.resample('1h').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
}).dropna()
```

**Validation**:

- Check bar counts: 15-min should be ~1/3 of 5-min, 1-hour should be ~1/12
- Verify no gaps: continuous timestamps
- Assert OHLC integrity: Open ≤ High, Low ≤ Close, etc.

---

## Implementation Phases

### Phase 16A: 15-Minute MA Crossover (Day 1, Part 1)

**File**: `scripts/01_ma_crossover_15min.py`

**Implementation**:

1. Load 5-minute ETH data (2022-2025, 394k bars)
2. Resample to 15-minute using pandas.resample()
3. Run MA crossover (100/300) - best from Phase 14A
4. Run MA crossover (50/200) - traditional baseline

**Expected Data**:

- 5-minute: 394,272 bars
- 15-minute: ~131,424 bars (3:1 ratio)
- Coverage: Same date range (2022-2025)

**Success Criteria** (either config):

- Win rate ≥ 45% (improvement from 5-min best 40.3%)
- Return > 0% (profitability)
- Trades ≥ 20 (sufficient sample)
- Sharpe > 0.5 (meaningful risk-adjusted return)

**Action if PASS**: Proceed to Phase 16C (cross-asset validation on 15-min)
**Action if FAIL**: Proceed to Phase 16B (test 1-hour)

---

### Phase 16B: 1-Hour MA Crossover (Day 1, Part 2)

**File**: `scripts/02_ma_crossover_1hour.py`

**Implementation**:

1. Resample 5-minute data to 1-hour
2. Run MA crossover (100/300)
3. Run MA crossover (50/200)
4. Run MA crossover (20/50) - shorter for lower bar count

**Expected Data**:

- 5-minute: 394,272 bars
- 1-hour: ~32,856 bars (12:1 ratio)
- Coverage: Same date range (2022-2025)

**Success Criteria** (any config):

- Win rate ≥ 45%
- Return > 0%
- Trades ≥ 10 (lower threshold due to fewer bars)
- Sharpe > 0.5

**Action if PASS**: Proceed to Phase 16D (optimize 1-hour parameters)
**Action if FAIL**: Proceed to Phase 16E (test BB mean reversion on both timeframes)

---

### Phase 16C: 15-Minute Cross-Asset Validation (Day 2) - CONDITIONAL

**Only execute if Phase 16A PASSES.**

**File**: `scripts/03_ma_crossover_15min_cross_asset.py`

**Test**: Run best 15-min config on ETH, BTC, SOL

**Success Criteria**:

- Win rate ≥ 45% on ≥2 assets
- Return > 0% on ≥2 assets
- Consistent behavior (not asset-specific)

**Action if PASS**: Deploy 15-min strategy to production testing
**Action if FAIL**: Single-asset strategy (ETH only)

---

### Phase 16D: 1-Hour Cross-Asset Validation (Day 2) - CONDITIONAL

**Only execute if Phase 16B PASSES.**

**File**: `scripts/04_ma_crossover_1hour_cross_asset.py`

**Test**: Run best 1-hour config on ETH, BTC, SOL

**Success Criteria**:

- Win rate ≥ 45% on ≥2 assets
- Return > 0% on ≥2 assets

**Action if PASS**: Deploy 1-hour strategy to production testing
**Action if FAIL**: Single-asset strategy (ETH only)

---

### Phase 16E: Bollinger Band Mean Reversion on Higher Timeframes (Day 3) - CONDITIONAL

**Only execute if Phase 16A AND 16B both FAIL.**

**File**: `scripts/05_bollinger_reversion_higher_timeframes.py`

**Rationale**: MA crossover failed on 5-min AND higher timeframes → test mean reversion

**Implementation**:

- Test BB mean reversion (from Phase 15A) on 15-minute data
- Test BB mean reversion on 1-hour data
- Use same parameters: 20-period BB, 2σ, RSI 14

**Success Criteria**:

- Win rate ≥ 50% on either timeframe
- Return > 0%

**Action if PASS**: Validate on cross-assets
**Action if FAIL**: ABANDON crypto directional trading entirely, escalate to Option B (traditional markets)

---

## File Structure

```
user_strategies/research/timeframe_analysis/
├── scripts/
│   ├── 01_ma_crossover_15min.py              # Phase 16A: 15-min MA
│   ├── 02_ma_crossover_1hour.py              # Phase 16B: 1-hour MA
│   ├── 03_ma_crossover_15min_cross_asset.py  # Phase 16C: 15-min multi-asset
│   ├── 04_ma_crossover_1hour_cross_asset.py  # Phase 16D: 1-hour multi-asset
│   └── 05_bollinger_reversion_higher_timeframes.py  # Phase 16E: BB fallback
├── results/
│   └── phase_16_timeframe_analysis/
│       ├── phase_16a_15min_ma.csv
│       ├── phase_16b_1hour_ma.csv
│       ├── phase_16c_15min_cross_asset.csv
│       ├── phase_16d_1hour_cross_asset.csv
│       ├── phase_16e_bb_higher_tf.csv
│       └── PHASE_16_TIMEFRAME_VALIDATION_REPORT.md
└── PHASE_16_TIMEFRAME_CHANGE_IMPLEMENTATION.md  # This file
```

---

## Code Reuse Policy

**Reuse from prior phases**:

- ✅ MA crossover strategy class (Phase 14A)
- ✅ BB mean reversion strategy class (Phase 15A)
- ✅ `calculate_atr()` function
- ✅ `calculate_rsi()` function
- ✅ `load_5m_data()` function
- ✅ Position tracking logic
- ✅ Error handling patterns

**Out-of-the-box resampling**:

- ✅ pandas.resample() for timeframe aggregation
- ✅ OHLC aggregation rules (first, max, min, last, sum)

**Modify**:

- Data loading: Add resampling step after load
- Validation: Add bar count checks post-resampling

---

## Error Handling Policy

**No silent failures. No defaults. No retries.**

### Resampling Validation

```python
# Example: Assert resampling integrity
if df_15m.isnull().sum().sum() > 0:
    raise ValueError(f"NaN values after resampling: {df_15m.isnull().sum()}")

expected_bars_15m = len(df_5m) / 3
actual_bars_15m = len(df_15m)
if abs(actual_bars_15m - expected_bars_15m) / expected_bars_15m > 0.05:
    raise RuntimeError(
        f"Resampling ratio incorrect: expected ~{expected_bars_15m:.0f} bars, "
        f"got {actual_bars_15m}"
    )

# Validate OHLC integrity
if (df_15m['Low'] > df_15m['High']).any():
    raise RuntimeError("OHLC integrity violated: Low > High after resampling")
if (df_15m['Open'] > df_15m['High']).any() or (df_15m['Open'] < df_15m['Low']).any():
    raise RuntimeError("OHLC integrity violated: Open outside High/Low range")
```

### Strategy Validation

```python
# Example: Validate strategy results
if stats['Return [%]'] <= 0 and stats['Win Rate [%]'] < 45:
    print(f"⚠️  Strategy failed on {timeframe}: "
          f"Win rate {stats['Win Rate [%]']:.1f}%, Return {stats['Return [%]']:+.2f}%")
    # Continue to next timeframe (no raise - this is expected outcome)
```

**All errors propagate. Failed strategies logged but don't halt execution.**

---

## Decision Gates

### Gate 1: End of Phase 16A (15-Minute)

**Metric**: MA crossover performance on 15-minute data

**Criteria** (either 100/300 or 50/200):

- Win rate ≥ 45%
- Return > 0%
- Trades ≥ 20
- Sharpe > 0.5

**PASS**: Proceed to Phase 16C (cross-asset)
**FAIL**: Proceed to Phase 16B (test 1-hour)

---

### Gate 2: End of Phase 16B (1-Hour)

**Metric**: MA crossover performance on 1-hour data

**Criteria** (any MA config):

- Win rate ≥ 45%
- Return > 0%
- Trades ≥ 10
- Sharpe > 0.5

**PASS**: Proceed to Phase 16D (cross-asset)
**FAIL**: Proceed to Phase 16E (test BB mean reversion)

---

### Gate 3: End of Phase 16E (BB Mean Reversion Fallback)

**Metric**: BB mean reversion on 15-min OR 1-hour

**Criteria** (either timeframe):

- Win rate ≥ 50%
- Return > 0%

**PASS**: Validate on cross-assets, deploy strategy
**FAIL**: **ABANDON crypto directional trading entirely**, escalate to Option B (traditional markets)

---

## Success Metrics

### Minimum Viable (Proceed to Cross-Asset)

- Win rate: ≥ 45%
- Return: > 0%
- Trades: ≥ 10 (1-hour) or ≥ 20 (15-min)
- Sharpe: > 0.5

### Production Ready

- Win rate: ≥ 50%
- Return: > 5%
- Sharpe: > 1.0
- Works on ≥2 assets (ETH, BTC, SOL)

---

## Alternative Actions (If All Timeframes Fail)

### If Phase 16A, 16B, AND 16E All Fail

**Evidence gathered**:

- 5-minute: 14 strategies failed (Phases 8-15A)
- 15-minute: MA crossover failed (Phase 16A)
- 1-hour: MA crossover failed (Phase 16B)
- 15-min/1-hour: BB mean reversion failed (Phase 16E)

**Conclusion**: Crypto markets fundamentally unsuitable for directional strategies across all intraday timeframes

**Recommended Action**: Option B - Test traditional markets (S&P 500, EUR/USD)

---

## Expected Outcomes

### Scenario 1: 15-Minute Works (Probability: ~40%)

**Expected Result**: Win rate 45-50%, return 5-15%, Sharpe 0.5-1.0
**Reason**: 3× noise reduction from 5-min, trends persist 3-5 bars
**Next Steps**: Cross-asset validation, deploy to production

### Scenario 2: 1-Hour Works (Probability: ~30%)

**Expected Result**: Win rate 48-55%, return 10-20%, Sharpe 0.8-1.2
**Reason**: 12× noise reduction, daily trends captured
**Next Steps**: Cross-asset validation, deploy to production

### Scenario 3: Both Fail, BB Works (Probability: ~10%)

**Expected Result**: Win rate 50-55% on BB mean reversion
**Reason**: Higher timeframes exhibit better mean reversion
**Next Steps**: Cross-asset validation

### Scenario 4: All Fail (Probability: ~20%)

**Result**: No viable crypto strategy across any timeframe
**Reason**: Market structure incompatible with retail directional strategies
**Next Steps**: Option B (traditional markets) or abandon directional trading

---

## Dependencies

### Python Packages (from pyproject.toml)

- `backtesting==0.6.5`
- `pandas==2.3.2` (resample() method)
- `numpy==2.3.3`

### Data Sources

- `user_strategies/data/raw/crypto_5m/binance_spot_ETHUSDT-5m_20220101-20250930_v2.10.0.csv`
- `user_strategies/data/raw/crypto_5m/binance_spot_BTCUSDT-5m_20220101-20250930_v2.10.0.csv` (Phase 16C/D)
- `user_strategies/data/raw/crypto_5m/binance_spot_SOLUSDT-5m_20220101-20250930_v2.10.0.csv` (Phase 16C/D)

### Prior Research (All 5-Minute, All Failed)

- Phase 8-13A: Compression research (best: 39.7% win rate, -100% return)
- Phase 14A: Trend following (best: 40.3% win rate, -100% return)
- Phase 15A: Mean reversion extremes (35.7% win rate, -100% return)
- Total: 10 phases, 14 strategies, 0 viable, HARD STOP triggered

---

## Version History

### v1.0.0 (2025-10-05)

- Initial implementation plan
- Timeframe change: 5-min → 15-min and 1-hour
- Reuse best strategies from Phases 14A (MA crossover) and 15A (BB reversion)
- Decision gates defined for each timeframe
- SLOs defined (availability, correctness, observability, maintainability)
- Option A from Phase 15A failure report

---

## References

**Supersedes**:

- Phase 8-15A: All 5-minute crypto research (ABANDONED)

**Implements**:

- Option A: Timeframe change (15-min, 1-hour)
- Resampling: pandas.resample() out-of-the-box

**Theoretical Sources**:

- O'Hara (1995): Market Microstructure
- Lo & MacKinlay (1988): Market inefficiency and autocorrelation

**Next Phase** (if PASS):

- Phase 16C/D: Cross-asset validation
- Production deployment

**Next Action** (if FAIL):

- Option B: Traditional markets (S&P 500, EUR/USD)
- Complete abandonment of crypto directional trading
