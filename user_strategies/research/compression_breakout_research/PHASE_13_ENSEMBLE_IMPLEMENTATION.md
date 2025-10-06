# Phase 13: Multi-Factor Ensemble Strategy Implementation

**Status**: FAILED at Gate 1
**Version**: 1.1.0 (updated with Phase 13A results)
**Date**: 2025-10-05
**Prerequisites**: Phase 12A failure analysis complete
**Outcome**: Trend filter improved win rate to 39.7% but insufficient for viability

---

## Objective

Validate if volatility compression becomes exploitable when combined with confirming indicators via ensemble pattern.

**Hypothesis**: Compression detection (Phase 8-10) is necessary but not sufficient. Adding trend/volume/momentum filters will improve win rate from 36% to 55%+.

---

## Service Level Objectives (SLOs)

### Availability
- Script execution success rate: ≥99%
- Data loading success rate: 100%

### Correctness
- Indicator calculation precision: ±0.01%
- Trade execution accuracy: 100% (no phantom trades)
- Temporal integrity: Zero lookahead bias tolerance

### Observability
- Logging coverage: 100% of filter decisions
- Trade-by-trade audit trail: Required
- Per-filter impact tracking: CSV output

### Maintainability
- Code reuse from Phase 12A: ≥80%
- Inline documentation: All filter logic
- Version tracking: Git commit per phase gate

**Excluded**: Performance optimization, security hardening

---

## Architecture

### Filter Pipeline

```
Entry Candidate (compression breakout detected)
    ↓
Filter 1: Trend Alignment (50-SMA slope)
    ↓ [PASS if aligned]
Filter 2: Volume Confirmation (>1.5× avg)
    ↓ [PASS if above threshold]
Filter 3: Momentum Range (RSI ∈ [40,60])
    ↓ [PASS if in neutral zone]
Execute Trade (all filters passed)
```

### Filter Definitions

**Filter 1: Trend Alignment**
- Indicator: 50-period SMA slope
- Long condition: slope > 0 (uptrend)
- Short condition: slope < 0 (downtrend)
- Purpose: Prevent counter-trend breakouts

**Filter 2: Volume Confirmation**
- Indicator: Volume ratio (current / 20-period avg)
- Entry condition: ratio > 1.5
- Purpose: Validate breakout strength

**Filter 3: Momentum Filter**
- Indicator: RSI (14-period)
- Entry condition: 40 ≤ RSI ≤ 60
- Purpose: Avoid overbought/oversold exhaustion

**Filter 4: Volatility Compression** (existing)
- Indicator: ATR percentile rank < 0.10
- Purpose: Setup condition (from Phase 10D)

---

## Implementation Phases

### Phase 13A: Trend Filter Addition (Day 1)

**File**: `scripts/10_ensemble_trend_filter.py`

**Changes from Phase 12A**:
```python
# Add to init():
self.sma_50 = self.I(lambda: df_5m['Close'].rolling(50).mean(), name='SMA50')

# Add to next() before entry:
sma_slope = self.sma_50[-1] - self.sma_50[-2]
if current_price > prev_high and sma_slope <= 0:
    return  # Block long entry in downtrend
if current_price < prev_low and sma_slope >= 0:
    return  # Block short entry in uptrend
```

**Test**: ETH full dataset
**Gate 1 Criteria**:
- Win rate > 40% (vs 36.3% baseline)
- Trades ≥ 100 (sufficient sample)

**Action if FAIL**: ABORT Phase 13, recommend proven strategies

**EXECUTION RESULTS (2025-10-05)**:

| Metric | Phase 10D Baseline | Phase 13A (Trend Filter) | Delta |
|--------|-------------------|-------------------------|-------|
| Return [%] | -100.00 | -100.00 | 0.00pp |
| # Trades | 846 | 863 | +17 |
| Win Rate [%] | 36.3 | 39.7 | **+3.4pp** |
| Sharpe Ratio | -23.33 | -13.68 | +9.65 |
| Max DD [%] | -100.00 | -100.00 | 0.00pp |

**Gate 1 Evaluation**:
- Win Rate > 40%: 39.7% ❌ FAIL (-0.3pp shortfall)
- Trades ≥ 100: 863 ✅ PASS (+763 surplus)

**Status**: ❌ **GATE 1 FAIL** - 1/2 criteria passed

**Analysis**:
- Trend filter DID improve win rate (+3.4pp from 36.3% to 39.7%)
- Improvement insufficient to meet 40% threshold
- Return remains catastrophic (-100%)
- Strategy still loses all capital despite improved win rate

**Conclusion**: Trend alignment alone cannot rescue compression-based entry signals. Compression zones remain fundamentally unpredictable regardless of trend context.

---

### Phase 13B: Volume Filter Addition (Day 2) - NOT EXECUTED

**File**: `scripts/11_ensemble_trend_volume.py`

**Changes from Phase 13A**:
```python
# Add to init():
self.volume_avg = self.I(lambda: df_5m['Volume'].rolling(20).mean(), name='VolumeAvg')

# Add to next() before entry:
volume_ratio = self.data.Volume[-1] / self.volume_avg[-1]
if volume_ratio < 1.5:
    return  # Block low-volume breakouts
```

**Test**: ETH full dataset
**Gate 2 Criteria**:
- Win rate > 45% (cumulative improvement)
- Return > -50% (loss reduction)

**Action if FAIL**: ABORT Phase 13

---

### Phase 13C: Momentum Filter Addition (Day 3)

**File**: `scripts/12_ensemble_full.py`

**Changes from Phase 13B**:
```python
# Add to init():
def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

self.rsi = self.I(lambda: calculate_rsi(df_5m['Close']), name='RSI')

# Add to next() before entry:
current_rsi = self.rsi[-1]
if current_rsi < 40 or current_rsi > 60:
    return  # Block overbought/oversold entries
```

**Test**: ETH full dataset
**Gate 3 Criteria**:
- Win rate ≥ 50% (cumulative improvement)
- Return > -25% (significant loss reduction)

**Action if FAIL**: ABORT Phase 13

---

### Phase 13D: Cross-Asset Validation (Day 4)

**File**: `scripts/13_ensemble_cross_asset_validation.py`

**Test**: Full ensemble on ETH, BTC, SOL

**Success Criteria**:
- Win rate ≥ 55% on ≥1 asset
- Return > 0% on ≥1 asset
- Trades ≥ 30/year on tested asset

**Action if PASS**: Proceed to Phase 14 (regime filtering on ensemble)
**Action if FAIL**: ABANDON compression approach

---

## File Structure

```
user_strategies/research/compression_breakout_research/
├── scripts/
│   ├── 10_ensemble_trend_filter.py          # Phase 13A
│   ├── 11_ensemble_trend_volume.py          # Phase 13B
│   ├── 12_ensemble_full.py                  # Phase 13C
│   └── 13_ensemble_cross_asset_validation.py # Phase 13D
├── results/
│   └── phase_13_ensemble/
│       ├── phase_13a_trend_filter.csv
│       ├── phase_13b_trend_volume.csv
│       ├── phase_13c_full_ensemble.csv
│       ├── phase_13d_cross_asset.csv
│       └── PHASE_13_VALIDATION_REPORT.md
└── PHASE_13_ENSEMBLE_IMPLEMENTATION.md      # This file
```

---

## Code Reuse Policy

**Reuse from Phase 12A** (`09_mean_reversion_strategy.py`):
- ✅ Base class structure: `Strategy` inheritance
- ✅ ATR calculation: `calculate_atr()`
- ✅ Percentile rank: `calculate_percentile_rank()`
- ✅ Multi-timeframe ATR filter
- ✅ Breakout detection (20-period high/low)
- ✅ Data loading: `load_5m_data()`

**Revert from Phase 12A**:
- ❌ Entry direction: Use Phase 10D ORIGINAL (buy high, sell low)
- ✅ Exit logic: Keep Phase 10D (2ATR stop, 4ATR target)

**Add new**:
- SMA calculation (pandas built-in)
- Volume ratio (pandas built-in)
- RSI calculation (custom function, minimal)

---

## Error Handling Policy

**No silent failures. No defaults. No retries.**

### Filter Validation
```python
# Example: Assert filter integrity
if pd.isna(self.sma_50[-1]):
    raise RuntimeError(f"NaN SMA at bar {len(self.data)}")

if self.volume_avg[-1] == 0:
    raise RuntimeError(f"Zero volume average at bar {len(self.data)}")

if pd.isna(self.rsi[-1]):
    raise RuntimeError(f"NaN RSI at bar {len(self.data)}")
```

### Gate Failures
```python
# Example: Hard stop on gate failure
if win_rate <= 40:
    raise RuntimeError(f"GATE 1 FAIL: Win rate {win_rate:.1f}% ≤ 40%")
```

**All errors propagate to caller. User decides next action.**

---

## Decision Gates

### Gate 1: End of Phase 13A

**Metric**: Win rate with trend filter only

**Criteria**:
- Win rate > 40% (improvement from 36.3%)
- Trades ≥ 100 (sufficient sample)

**GO**: Proceed to Phase 13B
**NO-GO**: ABORT Phase 13, recommend proven strategies

---

### Gate 2: End of Phase 13B

**Metric**: Win rate with trend + volume filters

**Criteria**:
- Win rate > 45% (cumulative improvement)
- Return > -50% (vs -100% baseline)

**GO**: Proceed to Phase 13C
**NO-GO**: ABORT Phase 13

---

### Gate 3: End of Phase 13C

**Metric**: Win rate with full ensemble (trend + volume + momentum)

**Criteria**:
- Win rate ≥ 50% (approaching random)
- Return > -25% (significant improvement)

**GO**: Proceed to Phase 13D
**NO-GO**: ABORT Phase 13

---

### Gate 4: End of Phase 13D

**Metric**: Cross-asset validation

**Criteria**:
- Win rate ≥ 55% on ≥1 asset
- Return > 0% on ≥1 asset
- Trades ≥ 30/year

**GO**: Proceed to Phase 14 (regime filtering)
**NO-GO**: ABANDON compression approach

---

## Success Metrics

### Minimum Viable (Proceed to Phase 14)
- Win rate: ≥ 55% on best asset
- Return: > 0% on best asset
- Trades: ≥ 30/year
- Sharpe: > 0.0

### Production Ready (Skip to deployment)
- Win rate: ≥ 60% on ≥2 assets
- Return: > 10% annualized
- Sharpe: > 1.0
- Max drawdown: < 30%

---

## Dependencies

### Python Packages (from pyproject.toml)
- `backtesting==0.6.5`
- `pandas==2.3.2`
- `numpy==2.3.3`

### Data Sources
- `user_strategies/data/raw/crypto_5m/binance_spot_ETHUSDT-5m_20220101-20250930_v2.10.0.csv`
- `user_strategies/data/raw/crypto_5m/binance_spot_BTCUSDT-5m_20220101-20250930_v2.10.0.csv`
- `user_strategies/data/raw/crypto_5m/binance_spot_SOLUSDT-5m_20220101-20250930_v2.10.0.csv`

### Prior Research
- Phase 8-10: Compression detection validated
- Phase 11: Root cause analysis (regime lockout)
- Phase 12A: Mean reversion hypothesis rejected

---

## Version History

### v1.0.0 (2025-10-04)
- Initial implementation plan
- Multi-factor ensemble architecture defined
- 4-day timeline with decision gates
- SLOs defined (availability, correctness, observability, maintainability)

---

## References

**Supersedes**:
- Phase 12: Mean reversion approach (failed)

**Implements**:
- Ensemble pattern: Compression + Trend + Volume + Momentum

**Next Phase** (after failure):
- Phase 14: Proven Strategies Implementation (compression research ABANDONED)
- File: `user_strategies/research/proven_strategies/PHASE_14_PROVEN_STRATEGIES_IMPLEMENTATION.md`
- Strategy: Dual Moving Average Crossover (Trend Following)
