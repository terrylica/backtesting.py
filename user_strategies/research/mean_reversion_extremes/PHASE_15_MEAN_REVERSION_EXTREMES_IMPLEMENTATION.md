# Phase 15: Mean Reversion from Extremes Implementation

**Status**: Active (FINAL TEST)
**Version**: 1.0.0
**Date**: 2025-10-05
**Prerequisites**: Phase 14A failure analysis complete, all trend strategies failed
**Rationale**: Final test of mean reversion hypothesis before abandoning crypto 5-minute trading

---

## Objective

Validate if crypto 5-minute markets are mean-reverting at price extremes using Bollinger Bands and RSI divergence.

**Hypothesis**: Crypto 5-minute markets exhibit strong mean reversion from statistical extremes (Bollinger Band touches), yielding >50% win rate and positive returns.

**Hard Stop Criteria**: If win rate <50% OR return <0%, ABANDON crypto 5-minute trading entirely.

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
- Bollinger Band calculation: Exact 2-sigma from 20-period SMA
- RSI calculation: Standard Wilder smoothing

### Observability
- Logging coverage: 100% of entry signals
- Trade-by-trade audit trail: Required
- BB touch classification: Upper/lower band identification
- RSI divergence logging: All divergence events captured

### Maintainability
- Code reuse: ≥70% from prior phases (ATR, data loading)
- Out-of-the-box indicators: pandas.rolling().std(), RSI standard formula
- Inline documentation: All signal logic
- Version tracking: Git commit on completion

**Excluded**: Performance optimization, security hardening

---

## Theoretical Foundation

### Bollinger Bands (John Bollinger, 1980s)

**Principle**: Price reverts to mean (moving average) after touching ±2σ bands

**Academic Support**:
- "Bollinger on Bollinger Bands" (Bollinger, 2001)
- Statistical basis: 95% of data within ±2σ under normal distribution
- Market application: Price extremes are unsustainable, revert to mean

**Signal Generation**:
- **Long entry**: Price touches lower band (oversold)
- **Short entry**: Price touches upper band (overbought)
- **Exit**: Price reverts to middle band (SMA)

### RSI Divergence (J. Welles Wilder, 1978)

**Principle**: Momentum divergence precedes price reversal

**Academic Support**:
- "New Concepts in Technical Trading Systems" (Wilder, 1978)
- Divergence types:
  - Bullish: Price makes lower low, RSI makes higher low
  - Bearish: Price makes higher high, RSI makes lower high

**Confirmation Filter**:
- Strengthens BB signal
- Reduces false entries at extremes

---

## Architecture

### Strategy: Bollinger Band Mean Reversion

**Indicators** (out-of-the-box):
```python
# Bollinger Bands (20-period, 2-sigma)
bb_period = 20
bb_std = 2.0

sma = df['Close'].rolling(bb_period).mean()
std = df['Close'].rolling(bb_period).std()
bb_upper = sma + (bb_std * std)
bb_lower = sma - (bb_std * std)
bb_middle = sma
```

**RSI** (standard Wilder smoothing):
```python
# RSI (14-period)
def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Wilder smoothing (EMA with alpha = 1/period)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
```

### Entry Logic

**LONG Entry** (oversold reversal):
```python
# Conditions (ALL must be true):
1. Price touches or breaks below lower BB
2. RSI < 30 (oversold confirmation)
3. Optional: RSI bullish divergence (price lower low, RSI higher low)

if (close <= bb_lower[-1] and
    rsi[-1] < 30):
    self.buy(size=0.95)
    entry_price = close
    target = bb_middle[-1]  # Revert to mean
    stop = close - (2.0 * atr)
```

**SHORT Entry** (overbought reversal):
```python
# Conditions (ALL must be true):
1. Price touches or breaks above upper BB
2. RSI > 70 (overbought confirmation)
3. Optional: RSI bearish divergence (price higher high, RSI lower high)

if (close >= bb_upper[-1] and
    rsi[-1] > 70):
    self.sell(size=0.95)
    entry_price = close
    target = bb_middle[-1]  # Revert to mean
    stop = close + (2.0 * atr)
```

### Exit Logic

**Exit Conditions**:
1. **Target**: Price reaches BB middle (mean reversion complete)
2. **Stop loss**: 2.0 × ATR from entry
3. **Time stop**: 100 bars (8.3 hours @ 5-min) without mean reversion
4. **Opposite signal**: New extreme in opposite direction

---

## Implementation Phases

### Phase 15A: Bollinger Band Baseline (Day 1)

**File**: `scripts/01_bollinger_mean_reversion.py`

**Implementation**:
- Bollinger Bands: 20-period SMA, 2σ
- RSI: 14-period Wilder smoothing
- Entry: BB touch + RSI <30 or >70
- Exit: BB middle, stop loss, or time stop

**Test**: ETH full dataset (2022-2025, 394k bars)

**HARD STOP Criteria** (Gate 1):
- Win rate ≥ 50% (random baseline)
- Return > 0% (profitability)
- Trades ≥ 20 (sufficient sample)
- Sharpe > 0.0 (positive risk-adjusted)

**Action if FAIL**: ABANDON crypto 5-minute trading, escalate with complete research summary

**Action if PASS**: Proceed to Phase 15B (add RSI divergence filter)

---

### Phase 15B: Add RSI Divergence Filter (Day 2) - CONDITIONAL

**Only execute if Phase 15A PASSES hard stop.**

**File**: `scripts/02_bollinger_rsi_divergence.py`

**Changes from Phase 15A**:
```python
# Divergence detection
def detect_bullish_divergence(price, rsi, lookback=5):
    # Price makes lower low
    price_ll = price[-1] < price[-lookback:-1].min()
    # RSI makes higher low
    rsi_hl = rsi[-1] > rsi[-lookback:-1].min()
    return price_ll and rsi_hl

# Entry condition: BB touch + RSI extreme + divergence
if (close <= bb_lower[-1] and
    rsi[-1] < 30 and
    detect_bullish_divergence(close_series, rsi_series)):
    self.buy(size=0.95)
```

**Test**: ETH full dataset

**Gate 2 Criteria**:
- Win rate ≥ 55% (improvement from baseline)
- Return > 5% (meaningful profit)
- Sharpe > 0.5 (validated performance)

**Action if PASS**: Proceed to Phase 15C (cross-asset)
**Action if FAIL**: Use Phase 15A baseline without divergence

---

### Phase 15C: Cross-Asset Validation (Day 3) - CONDITIONAL

**Only execute if Phase 15B PASSES Gate 2.**

**File**: `scripts/03_bollinger_cross_asset.py`

**Test**: Run best config on ETH, BTC, SOL

**Success Criteria**:
- Win rate ≥ 50% on ≥2 assets
- Return > 0% on ≥2 assets
- Consistent behavior (not asset-specific overfitting)

**Action if PASS**: Proceed to Phase 15D (optimization)
**Action if FAIL**: Deploy Phase 15B on ETH only (single-asset strategy)

---

### Phase 15D: Parameter Optimization (Day 4) - CONDITIONAL

**Only execute if Phase 15C PASSES.**

**File**: `scripts/04_bollinger_optimization.py`

**Parameters to sweep**:
- `bb_period`: [10, 20, 30]
- `bb_std`: [1.5, 2.0, 2.5]
- `rsi_period`: [10, 14, 20]
- `rsi_oversold`: [25, 30, 35]
- `rsi_overbought`: [65, 70, 75]

**Output**: Best configuration per asset, universal configuration

**Success Criteria**:
- Universal config: Win rate ≥55%, Sharpe >1.0 on ≥2 assets
- Return > 10% on best asset

**Action if PASS**: Deploy to production testing
**Action if FAIL**: Use Phase 15C baseline configuration

---

## File Structure

```
user_strategies/research/mean_reversion_extremes/
├── scripts/
│   ├── 01_bollinger_mean_reversion.py       # Phase 15A: Baseline
│   ├── 02_bollinger_rsi_divergence.py       # Phase 15B: Divergence filter
│   ├── 03_bollinger_cross_asset.py          # Phase 15C: Multi-asset
│   └── 04_bollinger_optimization.py         # Phase 15D: Optimization
├── results/
│   └── phase_15_mean_reversion/
│       ├── phase_15a_baseline.csv
│       ├── phase_15b_divergence.csv
│       ├── phase_15c_cross_asset.csv
│       ├── phase_15d_optimization.csv
│       └── PHASE_15_VALIDATION_REPORT.md
└── PHASE_15_MEAN_REVERSION_EXTREMES_IMPLEMENTATION.md  # This file
```

---

## Code Reuse Policy

**Reuse from prior phases**:
- ✅ `calculate_atr()` (Phase 10D)
- ✅ `load_5m_data()` (Phase 10D)
- ✅ Data validation patterns
- ✅ Position tracking logic

**Out-of-the-box implementations**:
- ✅ Bollinger Bands: `pandas.rolling().mean()`, `pandas.rolling().std()`
- ✅ RSI: Standard Wilder formula (EMA with alpha=1/14)
- ✅ Divergence: Simple lookback comparison

**Do NOT reuse**:
- ❌ Compression detection
- ❌ Regime filtering
- ❌ MA crossover logic

---

## Error Handling Policy

**No silent failures. No defaults. No retries.**

### Data Validation
```python
# Example: Assert data integrity
if df.isnull().sum().sum() > 0:
    raise ValueError(f"NaN values in input data: {df.isnull().sum()}")

if len(df) < 200:
    raise ValueError(f"Insufficient data: {len(df)} bars (need ≥200 for BB)")
```

### Indicator Validation
```python
# Example: Validate BB calculation
if pd.isna(bb_upper[-1]) or pd.isna(bb_lower[-1]):
    raise RuntimeError(f"NaN Bollinger Band at bar {len(self.data)}")

if bb_lower[-1] >= bb_upper[-1]:
    raise RuntimeError(f"Invalid BB: lower {bb_lower[-1]} >= upper {bb_upper[-1]}")
```

### Hard Stop Validation
```python
# Example: Enforce hard stop criteria
if stats['Win Rate [%]'] < 50:
    raise RuntimeError(
        f"HARD STOP TRIGGERED: Win rate {stats['Win Rate [%]']:.1f}% < 50% - "
        f"ABANDON crypto 5-minute trading"
    )

if stats['Return [%]'] <= 0:
    raise RuntimeError(
        f"HARD STOP TRIGGERED: Return {stats['Return [%]']:.2f}% ≤ 0% - "
        f"ABANDON crypto 5-minute trading"
    )
```

**All errors propagate to caller. Hard stop errors require user decision.**

---

## Decision Gates

### GATE 1 (HARD STOP): End of Phase 15A

**Metric**: Bollinger Band mean reversion baseline

**HARD STOP Criteria** (ALL must pass):
- Win rate ≥ 50% (random baseline)
- Return > 0% (profitability)
- Trades ≥ 20 (sufficient sample)
- Sharpe > 0.0 (positive risk-adjusted)

**PASS**: Proceed to Phase 15B (divergence filter)
**FAIL**: **ABANDON crypto 5-minute trading** - escalate with complete research summary

---

### Gate 2: End of Phase 15B (Conditional)

**Metric**: Performance with RSI divergence filter

**Criteria**:
- Win rate ≥ 55% (improvement)
- Return > 5% (meaningful profit)
- Sharpe > 0.5

**PASS**: Proceed to Phase 15C (cross-asset)
**FAIL**: Use Phase 15A baseline

---

### Gate 3: End of Phase 15C (Conditional)

**Metric**: Cross-asset consistency

**Criteria**:
- Win rate ≥ 50% on ≥2 assets
- Return > 0% on ≥2 assets

**PASS**: Proceed to Phase 15D (optimization)
**FAIL**: Deploy Phase 15B on ETH only

---

### Gate 4: End of Phase 15D (Conditional)

**Metric**: Optimized configuration

**Criteria**:
- Win rate ≥ 55% on ≥2 assets
- Return > 10% on best asset
- Sharpe > 1.0

**PASS**: Deploy to production
**FAIL**: Use Phase 15C baseline

---

## Success Metrics

### Minimum Viable (HARD STOP Criteria)
- Win rate: ≥ 50%
- Return: > 0%
- Trades: ≥ 20
- Sharpe: > 0.0

**If NOT met**: ABANDON crypto 5-minute trading

### Production Ready
- Win rate: ≥ 55%
- Return: > 10% (over 3.75 years)
- Sharpe: > 1.0
- Works on ≥2 assets (ETH, BTC, SOL)

---

## Alternative Actions (If Hard Stop Triggered)

### If Phase 15A Fails Hard Stop

**Evidence gathered across 10 phases**:
- Compression strategies: Best 39.7% win rate, -100% return
- Trend following: Best 40.3% win rate, -100% return
- Mean reversion extremes: <50% win rate or <0% return (if fails)

**Conclusion**: Crypto 5-minute markets fundamentally unsuitable for directional strategies

**Recommended Alternatives**:

1. **Change Timeframe**:
   - Test same strategies on 15-minute or 1-hour data
   - Hypothesis: Higher timeframe = better signal-to-noise
   - Timeline: 2 days
   - Probability: ~60%

2. **Change Asset Class**:
   - Test on traditional markets (S&P 500, forex)
   - Proven strategies validated on stocks/futures
   - Timeline: 3 days
   - Probability: ~80%

3. **Market Making Approach**:
   - Bid-ask spread capture
   - Inventory risk management
   - Requires exchange maker rebates
   - Timeline: 1 week
   - Probability: ~70%

4. **Abandon High-Frequency Trading**:
   - Focus on daily/weekly crypto strategies
   - Portfolio/macro approaches
   - Alternative: Traditional quant strategies
   - Timeline: Immediate

---

## Dependencies

### Python Packages (from pyproject.toml)
- `backtesting==0.6.5`
- `pandas==2.3.2`
- `numpy==2.3.3`

### Data Sources
- `user_strategies/data/raw/crypto_5m/binance_spot_ETHUSDT-5m_20220101-20250930_v2.10.0.csv`
- `user_strategies/data/raw/crypto_5m/binance_spot_BTCUSDT-5m_20220101-20250930_v2.10.0.csv` (Phase 15C)
- `user_strategies/data/raw/crypto_5m/binance_spot_SOLUSDT-5m_20220101-20250930_v2.10.0.csv` (Phase 15C)

### Prior Research (All Failed)
- Phase 8-13A: Compression research (best: 39.7% win rate, -100% return)
- Phase 14A: Trend following (best: 40.3% win rate, -100% return)
- Total: 9 phases, 12 strategies, 0 viable results

---

## Version History

### v1.0.0 (2025-10-05)
- Initial implementation plan
- Bollinger Band + RSI mean reversion strategy
- HARD STOP criteria defined: Win rate <50% OR return <0% → ABANDON
- SLOs defined (availability, correctness, observability, maintainability)
- Final test before abandoning crypto 5-minute trading

---

## References

**Supersedes**:
- Phase 8-14A: All prior crypto 5-minute research (FAILED)

**Implements**:
- Bollinger Band mean reversion (Bollinger, 2001)
- RSI divergence (Wilder, 1978)

**Theoretical Sources**:
- "Bollinger on Bollinger Bands" (John Bollinger, 2001)
- "New Concepts in Technical Trading Systems" (J. Welles Wilder, 1978)

**Next Phase** (if PASS):
- Phase 15B-15D: Iterative improvements
- Production deployment

**Next Action** (if FAIL):
- ABANDON crypto 5-minute trading
- Escalate decision: Change timeframe, asset class, or approach
- Complete research summary: 10 phases, 13+ strategies tested, all failed
