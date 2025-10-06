# Phase 12: Mean Reversion Pivot Implementation Plan

**Status**: FAILED at Gate 1
**Version**: 1.1.0 (updated with results)
**Date**: 2025-10-04
**Prerequisites**: Phase 11 diagnostics complete
**Outcome**: Mean reversion hypothesis rejected - compression approach non-viable

---

## Objective

Test hypothesis: Volatility compression → breakout failures can be exploited via mean reversion (fade breakouts).

**Hypothesis**: If 70% of breakouts fail (Phase 11 finding), then entering opposite direction should yield ~70% win rate.

---

## Service Level Objectives (SLOs)

### Availability
- Script execution success rate: ≥99% (allow 1 failure per 100 runs)
- Data loading success rate: 100% (hard requirement)

### Correctness
- Trade execution accuracy: 100% (no phantom trades)
- MFE/MAE calculation precision: ±0.01%
- Temporal integrity: Zero lookahead bias tolerance

### Observability
- Logging coverage: 100% of trade decisions
- Diagnostic output: CSV + markdown reports
- Trade-by-trade audit trail required

### Maintainability
- Code reuse from Phase 10D strategy: ≥80%
- Parameter externalization: All thresholds configurable
- Documentation: Inline comments for all logic inversions

**Excluded**: Performance optimization, security hardening (research code)

---

## Implementation Phases

### Phase 12A: Core Logic Inversion (Day 1)

**File**: `user_strategies/research/compression_breakout_research/scripts/09_mean_reversion_strategy.py`

**Changes from Phase 10D RollingWindowRegimeStrategy**:

```python
# BEFORE (Phase 10D - Breakout Following):
if current_price > prev_high:
    self.buy()   # Follow upside breakout
elif current_price < prev_low:
    self.sell()  # Follow downside breakout

# AFTER (Phase 12A - Mean Reversion):
if current_price > prev_high:
    self.sell()  # Fade upside breakout (expect reversion)
elif current_price < prev_low:
    self.buy()   # Fade downside breakout (expect reversion)
```

**Exit logic modifications**:
- Current: Stop at entry ± 2ATR, target at entry ± 4ATR
- New: Invert - stop at breakout continuation, target at range midpoint

**Dependencies**:
- Reuse: `calculate_atr()`, `calculate_percentile_rank()` from Phase 10D
- Reuse: Low volatility filter (multi-timeframe ATR < 10%)
- Reuse: Breakout detection (20-period high/low)

**Error Handling**:
- Data validation: Assert OHLCV columns present, raise ValueError if missing
- Indicator validation: Assert no NaN in ATR, raise RuntimeError if found
- Trade validation: Assert position size > 0, raise AssertionError if zero

---

### Phase 12B: Single-Asset Validation (Day 1-2)

**File**: `user_strategies/research/compression_breakout_research/scripts/10_mean_reversion_validation.py`

**Test Matrix**:

| Asset | Period | Baseline | Mean Reversion | Success Criteria |
|-------|--------|----------|----------------|------------------|
| ETH | 2022-2025 | -100% | Target: >-50% | Return improvement ≥50pp |
| ETH | 2024-2025 | -97.6% | Target: >0% | Profitability |
| ETH | 2022-2023 | -99.96% | Target: >-50% | Loss reduction |

**Metrics Captured**:
- Return [%]
- # Trades
- Win Rate [%] (target: ≥50%)
- Sharpe Ratio
- Max Drawdown [%]
- Avg Trade Duration (bars)
- MFE/MAE ratio distribution

**Kill Criteria** (hard stop):
1. Win rate < 35% (worse than breakout following)
2. Return < -75% (minimal improvement)
3. Trades < 10 (regime lockout persists)

**If ANY kill criterion met**: Abort Phase 12, escalate to user for decision.

---

### Phase 12C: Cross-Asset Validation (Day 2-3)

**File**: `user_strategies/research/compression_breakout_research/scripts/11_mean_reversion_cross_asset.py`

**Test**: Run mean reversion on BTC, SOL (same config as Phase 12B)

**Success Criteria**:
- ≥2 assets show return > -50%
- ≥1 asset shows return > 0%
- Win rate consistently ≥50% across assets

**Output**:
- `results/phase_12_mean_reversion/cross_asset_validation.csv`
- Comparative analysis vs Phase 11 baseline

---

### Phase 12D: Parameter Optimization (Day 3)

**Conditional**: Only execute if Phase 12C shows promise

**Parameters to sweep**:
- `stop_atr_multiple`: [1.0, 1.5, 2.0] (tighter than baseline)
- `target_atr_multiple`: [2.0, 3.0, 4.0] (range midpoint targets)
- `volatility_threshold`: [0.10, 0.15, 0.20] (current best known)

**Grid**: 3 × 3 × 3 = 27 combinations per asset

**Output**:
- Heatmap: Win rate vs (stop, target)
- Best configuration per asset
- Universal configuration (works across assets)

---

## File Structure

```
user_strategies/research/compression_breakout_research/
├── scripts/
│   ├── 09_mean_reversion_strategy.py          # Core strategy class
│   ├── 10_mean_reversion_validation.py        # Single-asset test
│   ├── 11_mean_reversion_cross_asset.py       # Multi-asset test
│   └── 12_mean_reversion_parameter_sweep.py   # Optimization (conditional)
├── results/
│   └── phase_12_mean_reversion/
│       ├── eth_baseline_vs_reversion.csv
│       ├── cross_asset_validation.csv
│       ├── parameter_sweep_results.csv (if Phase 12D runs)
│       └── PHASE_12_VALIDATION_REPORT.md
└── PHASE_12_MEAN_REVERSION_IMPLEMENTATION.md  # This file
```

---

## Code Reuse Policy

**Reuse from Phase 10D** (`08_comprehensive_parameter_sweep.py`):
- ✅ `calculate_atr()` function
- ✅ `calculate_percentile_rank()` function
- ✅ `load_5m_data()` function
- ✅ Multi-timeframe ATR filter logic
- ✅ Breakout detection (20-period high/low)

**Modify**:
- ❌ Entry direction (invert buy/sell)
- ❌ Exit targets (range midpoint vs continuation)
- ❌ Class name: `MeanReversionRegimeStrategy`

**Do NOT reuse**:
- ❌ Regime filtering (test baseline first)
- ❌ Parameter values (re-optimize for mean reversion)

---

## Decision Gates

### Gate 1: End of Day 1 (Phase 12B Complete)

**Check**: ETH single-asset results

**GO Criteria**:
- Return > -50%
- Win rate ≥ 50%
- Trades ≥ 50

**NO-GO**: If ANY criterion fails, escalate to user.

### Gate 2: End of Day 2 (Phase 12C Complete)

**Check**: Cross-asset results

**GO Criteria**:
- ≥1 asset profitable (return > 0%)
- All assets show improvement vs baseline
- Win rate ≥ 50% on ≥2 assets

**NO-GO**: If fails, recommend abandoning compression approach.

### Gate 3: End of Day 3 (Phase 12D Complete)

**Check**: Optimized configuration

**GO Criteria**:
- ≥1 asset shows return > 5% (annualized: >1.3%)
- Win rate ≥ 60%
- Sharpe ratio > 0.5

**NO-GO**: Strategy shows promise but needs regime filtering (Phase 13).

---

## Success Metrics

### Minimum Viable (Proceed to Phase 13)
- Return: > 0% on ≥1 asset
- Win rate: ≥ 50%
- Improvement vs baseline: ≥50pp

### Production Ready (Skip to deployment)
- Return: > 10% annualized (>37.5% over 3.75 years)
- Win rate: ≥ 60%
- Sharpe: > 1.0
- Works on ≥2 assets

---

## Error Handling Policy

**No silent failures. No defaults. No retries.**

### Data Errors
```python
# Example: Assert data integrity
if df.isnull().sum().sum() > 0:
    raise ValueError(f"NaN values in data: {df.isnull().sum()}")

if len(df) < 200:
    raise ValueError(f"Insufficient data: {len(df)} bars (need ≥200)")
```

### Execution Errors
```python
# Example: Validate trade execution
if position_size <= 0:
    raise AssertionError(f"Invalid position size: {position_size}")

if pd.isna(self.atr[-1]):
    raise RuntimeError(f"NaN ATR at bar {len(self.data)}")
```

### Validation Errors
```python
# Example: Check kill criteria
if stats['Win Rate [%]'] < 35:
    raise RuntimeError(f"KILL CRITERION: Win rate {stats['Win Rate [%]']:.1f}% < 35%")
```

**All errors propagate to caller. User must decide next action.**

---

## Dependencies

### Python Packages (from pyproject.toml)
- `backtesting==0.6.5` (framework)
- `pandas==2.3.2` (data manipulation)
- `numpy==2.3.3` (calculations)

### Data Sources
- `user_strategies/data/raw/crypto_5m/binance_spot_ETHUSDT-5m_20220101-20250930_v2.10.0.csv`
- `user_strategies/data/raw/crypto_5m/binance_spot_BTCUSDT-5m_20220101-20250930_v2.10.0.csv`
- `user_strategies/data/raw/crypto_5m/binance_spot_SOLUSDT-5m_20220101-20250930_v2.10.0.csv`

### Prior Research
- Phase 8: MAE/MFE compression analysis
- Phase 9: Streak entropy analysis
- Phase 10: Regime-aware trading (baseline strategy)
- Phase 11: Extended validation + root cause diagnostics

---

## Execution Results

### Phase 12A: FAILED at Gate 1

**ETH Results (2022-2025, 394k bars)**:

| Strategy | Return | Trades | Win Rate | Sharpe |
|----------|--------|--------|----------|--------|
| Breakout (Phase 10D) | -100.00% | 846 | 36.3% | -23.33 |
| Mean Reversion (Phase 12A) | -99.36% | 1,221 | **28.7%** | -1.75 |
| **Delta** | **+0.64pp** | **+375** | **-7.5pp** | **+21.58** |

**Gate 1 Evaluation**:
- Return > -50%: -99.36% ❌ FAIL
- Win Rate ≥ 50%: 28.7% ❌ FAIL
- Trades ≥ 50: 1,221 ✅ PASS

**Status**: ❌ 2/3 criteria failed → GATE 1 FAIL

**Key Finding**: Mean reversion performs WORSE than breakout following (28.7% vs 36.3% win rate), rejecting hypothesis that fading breakouts would improve performance.

**Root Cause**: Volatility compression zones do NOT predict profitable moves in EITHER direction. Both following and fading breakouts yield sub-random results.

---

## Version History

### v1.1.0 (2025-10-04)
- Added execution results
- Gate 1 failure documented
- Hypothesis rejected: Mean reversion worse than breakout
- Recommendation: Abandon compression approach

### v1.0.0 (2025-10-04)
- Initial implementation plan
- Based on Phase 11 diagnostics showing 70% breakout failure rate
- SLOs defined
- 3-day timeline with decision gates

---

## References

**Supersedes**:
- `/tmp/parameter_sensitivity/COMPREHENSIVE_REDESIGN_RECOMMENDATION.md` (archived to workspace)

**Implements**:
- Option B: Strategic Pivot - Mean Reversion from Compression

**Next Phase** (after failure):
- Phase 13: Ensemble Strategy (FAILED at Gate 1)
- Phase 14: Proven Strategies Implementation (compression research ABANDONED)
- File: `user_strategies/research/proven_strategies/PHASE_14_PROVEN_STRATEGIES_IMPLEMENTATION.md`
