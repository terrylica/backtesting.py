# Phase 14: Proven Strategy Implementation

**Status**: FAILED at Gate 1 (Phase 14A)
**Version**: 1.1.0 (updated with parameter sweep results)
**Date**: 2025-10-05
**Prerequisites**: Phase 13A failure analysis complete, compression research abandoned
**Rationale**: 13 phases of compression research yielded 39.7% best win rate with -100% returns
**Outcome**: Trend following (8 MA combinations) all fail with -100% returns

---

## Objective

Implement and validate proven trading strategies with established theoretical foundations and empirical success.

**Hypothesis**: Strategies with >15 years of academic validation will achieve >50% win rate and positive returns on crypto markets.

---

## Service Level Objectives (SLOs)

### Availability

- Script execution success rate: ≥99%
- Data loading success rate: 100%
- Indicator calculation availability: 100%

### Correctness

- Indicator calculation precision: ±0.01%
- Trade execution accuracy: 100% (no phantom trades)
- Temporal integrity: Zero lookahead bias tolerance
- Signal generation accuracy: 100% (matches theoretical definition)

### Observability

- Logging coverage: 100% of signal generation
- Trade-by-trade audit trail: Required
- Per-strategy performance tracking: CSV output
- Indicator state logging: All crossovers, touches, divergences

### Maintainability

- Code reuse from backtesting.py lib: ≥60%
- Out-of-the-box indicators: Prefer pandas/numpy over custom
- Inline documentation: All signal logic
- Version tracking: Git commit per phase gate

**Excluded**: Performance optimization, security hardening

---

## Strategy Selection Criteria

### Prioritization Matrix

| Strategy                | Theory Strength      | Implementation Complexity | Expected Win Rate | Expected Sharpe | Priority |
| ----------------------- | -------------------- | ------------------------- | ----------------- | --------------- | -------- |
| Trend Following         | High (40+ years)     | Low (MA crossovers)       | 40-45%            | >0.5            | **1**    |
| Mean Reversion Extremes | High (30+ years)     | Low (Bollinger/RSI)       | 55-60%            | >1.0            | **2**    |
| Volume Breakouts        | Medium (20+ years)   | Medium (volume analysis)  | 45-50%            | >0.7            | **3**    |
| Market Making           | High (institutional) | High (inventory risk)     | N/A               | >2.0            | **4**    |

### Selection Decision

**Start with Priority 1: Trend Following**

**Rationale**:

- Lowest implementation complexity
- Longest track record (Donchian 1960s, Turtle Traders 1980s)
- Works across all markets (stocks, futures, crypto)
- Robust to market regimes (bull/bear/sideways)
- Out-of-the-box indicators: pandas rolling(), no custom code

---

## Architecture

### Strategy 1: Dual Moving Average Crossover (Trend Following)

**Theoretical Foundation**:

- Source: "Technical Analysis of Stock Trends" (Edwards & Magee, 1948)
- Validation: Turtle Trading System (Dennis & Eckhardt, 1983)
- Academic: "Momentum and Reversal" (Jegadeesh & Titman, 1993)

**Signal Generation**:

```
Fast MA (50-period) crosses above Slow MA (200-period) → LONG
Fast MA (50-period) crosses below Slow MA (200-period) → SHORT
```

**Confirmation Filters**:

1. ADX > 25 (trend strength filter)
2. ATR percentile > 50% (volatility expansion, opposite of compression)
3. Volume > 20-period average (participation confirmation)

**Exit Logic**:

- Opposite crossover signal
- Stop: 2.0 × ATR
- Trailing stop: 3.0 × ATR from highest high (long) or lowest low (short)
- Max hold: 500 bars (100 hours @ 5-min)

**Expected Performance**:

- Win rate: 40-45% (trend following typically <50%)
- Profit factor: >1.5 (winners larger than losers)
- Sharpe: >0.5
- Max drawdown: <40%

---

## Implementation Phases

### Phase 14A: Trend Following - Baseline Implementation (Day 1)

**File**: `scripts/01_dual_ma_crossover.py`

**Indicators** (out-of-the-box):

```python
# Fast and slow moving averages
self.ma_fast = self.I(lambda: df['Close'].rolling(50).mean(), name='MA_Fast')
self.ma_slow = self.I(lambda: df['Close'].rolling(200).mean(), name='MA_Slow')

# ATR for stops
self.atr = self.I(lambda: calculate_atr(df, 14), name='ATR')
```

**Entry Logic**:

```python
# Long entry: Fast crosses above slow
if self.ma_fast[-2] <= self.ma_slow[-2] and self.ma_fast[-1] > self.ma_slow[-1]:
    self.buy(size=0.95)  # Full position
    self.stop_loss = self.data.Close[-1] - (2.0 * self.atr[-1])

# Short entry: Fast crosses below slow
if self.ma_fast[-2] >= self.ma_slow[-2] and self.ma_fast[-1] < self.ma_slow[-1]:
    self.sell(size=0.95)  # Full position
    self.stop_loss = self.data.Close[-1] + (2.0 * self.atr[-1])
```

**Exit Logic**:

```python
# Exit on opposite crossover
if self.position.is_long and self.ma_fast[-1] < self.ma_slow[-1]:
    self.position.close()
elif self.position.is_short and self.ma_fast[-1] > self.ma_slow[-1]:
    self.position.close()

# Trailing stop management
if self.position.is_long:
    trailing_stop = max(self.data.High[-500:].max() - (3.0 * self.atr[-1]), self.stop_loss)
    if self.data.Low[-1] <= trailing_stop:
        self.position.close()
```

**Test**: ETH full dataset (2022-2025, 394k bars)

**Gate 1 Criteria**:

- Win rate ≥ 35% (trend following baseline)
- Return > 0% (profitability)
- Trades ≥ 10 (sufficient signals)
- Sharpe > 0.0 (positive risk-adjusted return)

**Action if FAIL**: Adjust MA periods (test 20/100, 100/300), re-validate

---

### Phase 14B: Trend Following - Add Confirmation Filters (Day 2)

**File**: `scripts/02_dual_ma_crossover_filtered.py`

**Changes from Phase 14A**:

```python
# Add ADX for trend strength
def calculate_adx(df, period=14):
    # Directional movement calculation
    high = df['High']
    low = df['Low']
    close = df['Close']

    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)

    atr = calculate_atr(df, period)
    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(span=period, adjust=False).mean()
    return adx

self.adx = self.I(lambda: calculate_adx(df), name='ADX')

# Add ATR percentile for volatility filter
atr_percentile = calculate_percentile_rank(atr, 150)
self.atr_pct = self.I(lambda: atr_percentile, name='ATR_Percentile')

# Add volume filter
self.volume_avg = self.I(lambda: df['Volume'].rolling(20).mean(), name='Volume_Avg')

# Entry condition: MA crossover + filters
if (self.ma_fast[-2] <= self.ma_slow[-2] and
    self.ma_fast[-1] > self.ma_slow[-1] and
    self.adx[-1] > 25 and  # Strong trend
    self.atr_pct[-1] > 0.5 and  # Volatility expansion (NOT compression)
    self.data.Volume[-1] > self.volume_avg[-1]):  # Volume confirmation
    self.buy(size=0.95)
```

**Test**: ETH full dataset

**Gate 2 Criteria**:

- Win rate ≥ 40% (improvement from baseline)
- Return > 5% (meaningful profit)
- Sharpe > 0.5 (validated risk-adjusted performance)
- Max DD < 40% (controlled risk)

**Action if FAIL**: Relax filters (ADX > 20, no volume filter), re-validate

---

### Phase 14C: Trend Following - Cross-Asset Validation (Day 3)

**File**: `scripts/03_dual_ma_crossover_cross_asset.py`

**Test**: Run Phase 14B strategy on ETH, BTC, SOL

**Success Criteria**:

- Win rate ≥ 40% on ≥2 assets
- Return > 0% on ≥2 assets
- Sharpe > 0.5 on ≥1 asset
- Consistent behavior across assets (not asset-specific overfitting)

**Action if PASS**: Proceed to Phase 14D (parameter optimization)
**Action if FAIL**: Test alternative trend following (Donchian Channel, MACD)

---

### Phase 14D: Trend Following - Parameter Optimization (Day 4)

**File**: `scripts/04_dual_ma_crossover_optimization.py`

**Parameters to sweep**:

- `ma_fast_period`: [20, 50, 100]
- `ma_slow_period`: [100, 200, 300]
- `stop_atr_multiple`: [1.5, 2.0, 2.5]
- `adx_threshold`: [20, 25, 30]

**Grid**: 3 × 3 × 3 × 3 = 81 combinations per asset

**Optimization Method**:

```python
# Use backtesting.py built-in optimize()
stats = bt.optimize(
    ma_fast_period=[20, 50, 100],
    ma_slow_period=[100, 200, 300],
    stop_atr_multiple=[1.5, 2.0, 2.5],
    adx_threshold=[20, 25, 30],
    maximize='Sharpe Ratio',
    constraint=lambda p: p.ma_fast_period < p.ma_slow_period  # Fast < Slow
)
```

**Output**:

- Best configuration per asset
- Universal configuration (works across assets)
- Walk-forward validation results

**Success Criteria**:

- Universal config: Sharpe > 1.0 on ≥2 assets
- Return > 10% on best asset
- Win rate ≥ 45% on best config

**Action if PASS**: Deploy to production testing
**Action if FAIL**: Proceed to Phase 15 (Mean Reversion from Extremes)

---

## File Structure

```
user_strategies/research/proven_strategies/
├── scripts/
│   ├── 01_dual_ma_crossover.py              # Phase 14A: Baseline
│   ├── 02_dual_ma_crossover_filtered.py     # Phase 14B: With filters
│   ├── 03_dual_ma_crossover_cross_asset.py  # Phase 14C: Multi-asset
│   └── 04_dual_ma_crossover_optimization.py # Phase 14D: Optimization
├── results/
│   └── phase_14_trend_following/
│       ├── phase_14a_baseline.csv
│       ├── phase_14b_filtered.csv
│       ├── phase_14c_cross_asset.csv
│       ├── phase_14d_optimization.csv
│       └── PHASE_14_VALIDATION_REPORT.md
└── PHASE_14_PROVEN_STRATEGIES_IMPLEMENTATION.md  # This file
```

---

## Code Reuse Policy

**Reuse from backtesting.py lib**:

- ✅ Out-of-the-box: `pandas.rolling().mean()` for MA
- ✅ From Phase 10D: `calculate_atr()` function
- ✅ From Phase 10D: `calculate_percentile_rank()` function
- ✅ From backtesting.py: `Strategy` base class
- ✅ From backtesting.py: `bt.optimize()` for parameter sweeps

**Do NOT reuse from compression research**:

- ❌ Multi-timeframe ATR compression filter
- ❌ Regime filtering logic
- ❌ Compression-based entry signals

**New implementations** (out-of-the-box):

- ADX calculation: Standard formula, pandas/numpy only
- Trailing stop: `max()` function on rolling high/low
- Crossover detection: Compare `[-2]` vs `[-1]` values

---

## Error Handling Policy

**No silent failures. No defaults. No retries.**

### Data Validation

```python
# Example: Assert data integrity
if df.isnull().sum().sum() > 0:
    raise ValueError(f"NaN values in input data: {df.isnull().sum()}")

if len(df) < 250:
    raise ValueError(f"Insufficient data for 200-SMA: {len(df)} bars (need ≥250)")
```

### Indicator Validation

```python
# Example: Validate indicator state
if pd.isna(self.ma_fast[-1]) or pd.isna(self.ma_slow[-1]):
    raise RuntimeError(f"NaN MA at bar {len(self.data)}: fast={self.ma_fast[-1]}, slow={self.ma_slow[-1]}")

if self.atr[-1] <= 0:
    raise RuntimeError(f"Invalid ATR at bar {len(self.data)}: {self.atr[-1]}")
```

### Gate Validation

```python
# Example: Hard stop on gate failure
if stats['Return [%]'] <= 0:
    raise RuntimeError(f"GATE 1 FAIL: Negative return {stats['Return [%]']:.2f}% (need >0%)")

if stats['Win Rate [%]'] < 35:
    raise RuntimeError(f"GATE 1 FAIL: Win rate {stats['Win Rate [%]']:.1f}% < 35%")
```

**All errors propagate to caller. User decides next action.**

---

## Decision Gates

### Gate 1: End of Phase 14A (Baseline Validation)

**Metric**: Trend following baseline performance

**Criteria**:

- Win rate ≥ 35% (trend following baseline)
- Return > 0% (profitability)
- Trades ≥ 10 (sufficient signals)
- Sharpe > 0.0 (positive risk-adjusted)

**GO**: Proceed to Phase 14B (add filters)
**NO-GO**: Adjust MA periods, re-validate

---

### Gate 2: End of Phase 14B (Filtered Strategy)

**Metric**: Performance with confirmation filters

**Criteria**:

- Win rate ≥ 40% (improvement)
- Return > 5% (meaningful profit)
- Sharpe > 0.5 (validated performance)
- Max DD < 40% (controlled risk)

**GO**: Proceed to Phase 14C (cross-asset)
**NO-GO**: Relax filters, re-validate

---

### Gate 3: End of Phase 14C (Cross-Asset)

**Metric**: Multi-asset consistency

**Criteria**:

- Win rate ≥ 40% on ≥2 assets
- Return > 0% on ≥2 assets
- Sharpe > 0.5 on ≥1 asset

**GO**: Proceed to Phase 14D (optimization)
**NO-GO**: Test alternative trend systems (Donchian, MACD)

---

### Gate 4: End of Phase 14D (Optimization)

**Metric**: Optimized configuration performance

**Criteria**:

- Universal config: Sharpe > 1.0 on ≥2 assets
- Return > 10% on best asset
- Win rate ≥ 45% on best config
- No overfitting: Walk-forward validation passes

**GO**: Deploy to production testing
**NO-GO**: Proceed to Phase 15 (Mean Reversion)

---

## Success Metrics

### Minimum Viable (Proceed to Production)

- Win rate: ≥ 40% on best asset
- Return: > 10% on 3.75-year backtest
- Sharpe: > 1.0
- Max drawdown: < 40%
- Cross-asset: Works on ≥2 of 3 assets

### Production Ready (Live trading candidate)

- Win rate: ≥ 45% on ≥2 assets
- Return: > 20% annualized
- Sharpe: > 1.5
- Max drawdown: < 30%
- Walk-forward: Consistent across 4+ periods

---

## Alternative Strategies (If Phase 14 Fails)

### Phase 15: Mean Reversion from Extremes

**Signal**: Bollinger Band touch + RSI divergence
**Expected win rate**: 55-60%
**Expected Sharpe**: >1.0
**Complexity**: Low (out-of-the-box indicators)

### Phase 16: Volume-Confirmed Breakouts

**Signal**: Range consolidation + volume spike >2× avg
**Expected win rate**: 45-50%
**Expected Sharpe**: >0.7
**Complexity**: Medium (volume analysis)

### Phase 17: Market Making (Advanced)

**Signal**: Bid-ask spread capture
**Expected Sharpe**: >2.0
**Complexity**: High (inventory risk management)
**Requirement**: Exchange maker rebates

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

### Prior Research (Archived)

- Phase 8-13A: Compression research (ABANDONED)
- Best compression result: 39.7% win rate, -100% return

---

## Compression Research Archive

### Status: ABANDONED

**Phases executed**: 8, 9, 10, 11, 11B, 11C, 12A, 13A (8 total)
**Best result**: Phase 13A - 39.7% win rate, -100% return
**Conclusion**: Compression zones are not exploitable signals

**Archive location**: `user_strategies/research/compression_breakout_research/`
**Reference**: See `PHASE_13A_GATE1_FAILURE_REPORT.md` for complete analysis

---

## Version History

### v1.0.0 (2025-10-05)

- Initial implementation plan
- Dual MA crossover strategy selected (Priority 1)
- 4-day timeline with decision gates
- SLOs defined (availability, correctness, observability, maintainability)
- Compression research formally abandoned

---

## References

**Supersedes**:

- Phase 8-13A: Compression breakout research (ABANDONED)
- Phase 13 Ensemble Implementation (FAILED at Gate 1)

**Implements**:

- Option A: Proven strategies with academic validation
- Trend following: Dual moving average crossover

**Theoretical Sources**:

- Edwards & Magee (1948): Technical Analysis of Stock Trends
- Dennis & Eckhardt (1983): Turtle Trading System
- Jegadeesh & Titman (1993): Momentum and Reversal

**Next Phase** (if Phase 14 succeeds):

- Production deployment with risk management
- Live paper trading on testnet
- Walk-forward validation on new data

**Next Phase** (if Phase 14 fails):

- Phase 15: Mean Reversion from Extremes
- Phase 16: Volume-Confirmed Breakouts
- Phase 17: Market Making (advanced)
