---
adr: 2025-12-20-clickhouse-triple-barrier-backtest
source: N/A (derived from conversation context)
implementation-status: in_progress
phase: preflight
last-updated: 2025-12-20
---

# Design Spec: ClickHouse-Powered Triple-Barrier Probabilistic Classification

**ADR**: [ClickHouse Triple-Barrier Backtest](/docs/adr/2025-12-20-clickhouse-triple-barrier-backtest.md)

## Overview

Implement a research-grade probabilistic classification system that:

1. Uses `gapless-crypto-clickhouse` for second-level data with zero-gap guarantees
2. Uses ClickHouse as an analytics engine for microstructure feature aggregation
3. Implements triple-barrier labeling with 3-class outputs (Y in {+, -, 0})
4. Trains probabilistic classifiers with purged/embargoed cross-validation
5. Integrates with backtesting.py for threshold-based strategy simulation

## Mathematical Framework

### Triple-Barrier Labels

At each event time t_0, define first-passage times:

```
tau^+ = inf{t > t_0 : X_t - X_{t_0} >= b}  (upper barrier hit)
tau^- = inf{t > t_0 : X_t - X_{t_0} <= -b} (lower barrier hit)
tau^0 = t_0 + H                             (horizon timeout)
tau = min(tau^+, tau^-, tau^0)
```

Label definition:

```
Y = +1 if tau = tau^+  (upper hit first)
Y = -1 if tau = tau^-  (lower hit first)
Y =  0 if tau = tau^0  (timeout)
```

Target probability: `pi_+(x) = P(Y = + | X_{t_0} = x)`

### Probabilistic Model

```
pi(x) = (pi_+, pi_-, pi_0) = softmax(f_theta(x))
```

Loss function: Cross-entropy (proper scoring rule)

```
L = -sum_k Y_k * log(pi_k(x))
```

## Implementation Tasks

### Phase 1: ClickHouse Integration Layer

- [ ] **Task 1.1**: Create `get_data_from_clickhouse()` adapter function
  - Query OHLCV data from ClickHouse via gapless-crypto-clickhouse
  - Map columns to backtesting.py format (Open, High, Low, Close, Volume)
  - Handle auto-ingest for missing data periods
  - Location: `user_strategies/strategies/clickhouse_adapter.py`

- [ ] **Task 1.2**: Create ClickHouse connection configuration
  - Environment variable support (CLICKHOUSE_HOST, etc.)
  - Fallback to local defaults
  - Location: `user_strategies/configs/clickhouse_config.py`

### Phase 2: Microstructure Feature SQL

- [ ] **Task 2.1**: Create SQL templates for microstructure aggregation
  - Taker buy ratio: `sum(taker_buy_base_asset_volume) / sum(volume)`
  - Order flow imbalance: `sum(taker_buy) - sum(volume - taker_buy)`
  - Trade intensity: `sum(number_of_trades)`, `avg(number_of_trades)`
  - VWAP deviation: `last(close) - sum(close * volume) / sum(volume)`
  - Hourly volatility: `stddevPop(log returns) * sqrt(3600)`
  - Location: `user_strategies/sql/microstructure_features.sql`

- [ ] **Task 2.2**: Create Python wrapper for SQL execution
  - Parameterized queries (symbol, interval, date range)
  - Return pandas DataFrame with feature columns
  - Location: `user_strategies/strategies/microstructure_features.py`

### Phase 3: Triple-Barrier Label Generation

- [ ] **Task 3.1**: Implement `compute_triple_barrier_labels()` function
  - Input: price series, barrier_pct (b), horizon_bars (H)
  - Output: pd.Series with Y in {+1, -1, 0}
  - Handle edge cases (insufficient forward data)
  - Location: `user_strategies/strategies/triple_barrier.py`

- [ ] **Task 3.2**: Create hyperparameter grid for b and H
  - Barrier percentages: [0.5%, 1%, 1.5%, 2%, 2.5%]
  - Horizon bars: [12, 24, 48, 72, 96] (at hourly granularity)
  - Location: `user_strategies/configs/barrier_config.py`

### Phase 4: Probabilistic Classifier with Purged CV

- [ ] **Task 4.1**: Implement `purged_kfold_split()` generator
  - TimeSeriesSplit base
  - Purge: remove training samples within horizon of test start
  - Embargo: add buffer between train and test
  - Location: `user_strategies/strategies/purged_cv.py`

- [ ] **Task 4.2**: Implement `TripleBarrierClassifier` class
  - 3-class softmax output
  - Cross-entropy loss (proper scoring rule)
  - Support for sklearn API (fit, predict, predict_proba)
  - Location: `user_strategies/strategies/classifier.py`

- [ ] **Task 4.3**: Implement hyperparameter optimization
  - Grid search over b, H using purged CV
  - Maximize log-likelihood (minimize cross-entropy)
  - Return best (b, H) configuration
  - Location: `user_strategies/strategies/hyperopt.py`

### Phase 5: backtesting.py Strategy Integration

- [ ] **Task 5.1**: Implement `TripleBarrierProbStrategy` class
  - Inherits from backtesting.Strategy
  - Uses pre-computed pi\_+(x) predictions
  - Entry when pi\_+(x) > threshold (optimizable parameter)
  - Location: `user_strategies/strategies/triple_barrier_strategy.py`

- [ ] **Task 5.2**: Create end-to-end pipeline script
  - Load data from ClickHouse
  - Compute microstructure features
  - Generate triple-barrier labels
  - Train classifier with purged CV
  - Run backtest with threshold optimization
  - Location: `user_strategies/scripts/run_triple_barrier_pipeline.py`

- [ ] **Task 5.3**: Add calibration diagnostics
  - Reliability diagram (predicted vs actual)
  - Brier score decomposition
  - ECE (Expected Calibration Error)
  - Location: `user_strategies/strategies/calibration.py`

### Phase 6: Documentation and Tests

- [ ] **Task 6.1**: Create comprehensive docstrings
  - Mathematical formulations in docstrings
  - Example usage patterns
  - Parameter descriptions

- [ ] **Task 6.2**: Create pytest test suite
  - Unit tests for each component
  - Integration test for full pipeline
  - Location: `user_strategies/tests/test_triple_barrier.py`

## Success Criteria

- [ ] ClickHouse connection works with gapless-crypto-clickhouse
- [ ] Microstructure features computed from 1-second data
- [ ] Triple-barrier labels generated correctly
- [ ] Classifier produces calibrated probabilities (reliability diagram)
- [ ] Purged CV prevents look-ahead bias
- [ ] backtesting.py strategy runs without errors
- [ ] Threshold optimization produces meaningful results
- [ ] All tests pass

## File Structure

```
user_strategies/
├── configs/
│   ├── clickhouse_config.py      # ClickHouse connection
│   └── barrier_config.py         # b, H hyperparameters
├── sql/
│   └── microstructure_features.sql
├── strategies/
│   ├── clickhouse_adapter.py     # Task 1.1
│   ├── microstructure_features.py # Task 2.2
│   ├── triple_barrier.py         # Task 3.1
│   ├── purged_cv.py              # Task 4.1
│   ├── classifier.py             # Task 4.2
│   ├── hyperopt.py               # Task 4.3
│   ├── triple_barrier_strategy.py # Task 5.1
│   └── calibration.py            # Task 5.3
├── scripts/
│   └── run_triple_barrier_pipeline.py # Task 5.2
└── tests/
    └── test_triple_barrier.py    # Task 6.2
```

## Dependencies

```toml
# Add to pyproject.toml
[project.optional-dependencies]
triple-barrier = [
    "gapless-crypto-clickhouse>=1.0.0",
    "clickhouse-driver>=0.2.0",
    "scikit-learn>=1.0.0",
    "torch>=2.0.0",  # Optional, for neural classifier
]
```

## Notes

- This is a **research-only** implementation, not for live trading
- ClickHouse must be running locally or accessible via cloud
- Requires substantial historical data for meaningful validation
- Hyperparameter optimization can be computationally expensive
