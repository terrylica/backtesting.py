# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a forked copy of backtesting.py (https://kernc.github.io/backtesting.py/), a Python framework for backtesting trading strategies. The current focus is on **testing and developing machine learning-based walk-forward optimization strategies** while maintaining clean separation between user-created code and the original backtesting.py codebase.

## Core Architecture

### Main Framework Components
- **`backtesting/backtesting.py`**: Core `Backtest` and `Strategy` classes that form the framework foundation
- **`backtesting/lib.py`**: Composable base strategies (`SignalStrategy`, `TrailingStrategy`) and utilities
- **`backtesting/_stats.py`**: Statistical computations (Sharpe ratio, drawdown, etc.)
- **`backtesting/_plotting.py`**: Interactive Bokeh-based visualization
- **`backtesting/_util.py`**: Internal utilities including cross-platform SharedMemory compatibility

### User Strategy Architecture (Separation of Concerns)

The project implements **clean separation** between original maintainer code and user development:

```
user_strategies/                    # User code (separate from original)
├── strategies/
│   ├── ml_strategy.py              # Production ML strategies
│   └── __init__.py
├── tests/
│   ├── test_ml_strategy.py         # Pytest-based tests
│   └── __init__.py
├── configs/                        # Strategy configurations
└── data/                          # Trading data
```

### ML Strategy Implementation Pattern

The **MLWalkForwardStrategy** is the most robust strategy implementation, featuring:
- **Walk-forward optimization**: Retrains ML model every N periods (default: 20)
- **Temporal integrity**: No look-ahead bias in feature engineering or model training
- **Production-ready architecture**: k-NN classification with proper risk management
- **Advanced features**: Multi-timeframe indicators, dynamic stop-losses, time-based trade management

## Development Environment

### Dependency Management
```bash
# Primary package manager - UV is used exclusively
uv add <package>                    # Add runtime dependency
uv add --dev <package>             # Add development dependency
uv run --active python <script>    # Run with UV environment
```

### Essential Dependencies
- **Core**: `numpy`, `pandas`, `scikit-learn`, `bokeh`
- **Testing**: `pytest`, `coverage`
- **Linting**: `ruff`, `flake8`, `mypy`
- **ML**: `sambo` (optimization), `tqdm` (progress bars)

## Common Development Commands

### Testing
```bash
# Run original framework tests
uv run --active python -m backtesting.test

# Run user strategy tests (pytest-based)
uv run --active python -m pytest user_strategies/tests/ -v

# Test specific ML strategy
uv run --active python -m pytest user_strategies/tests/test_ml_strategy.py::TestMLStrategies::test_ml_walk_forward_strategy_execution -v

# Run coverage analysis
uv run --active python -m coverage run --source=backtesting -m backtesting.test
uv run --active python -m coverage report --show-missing
uv run --active python -m coverage html
```

### Code Quality
```bash
# Linting (configured in pyproject.toml)
uv run --active python -m ruff check .
uv run --active python -m flake8
uv run --active python -m mypy backtesting/

# Auto-formatting
uv run --active python -m ruff format .
```

### ML Strategy Development
```bash
# Current working test file with persistent output
uv run --active python -c "
import sys; sys.path.append('user_strategies')
from strategies.ml_strategy import run_ml_strategy_with_persistence, MLWalkForwardStrategy
from backtesting.test import EURUSD

# Primary development pattern - saves to user_strategies/data/
data = EURUSD.iloc[:600]
stats, trades, files = run_ml_strategy_with_persistence(
    data, MLWalkForwardStrategy, n_train=200, retrain_frequency=20
)
print(f'Files: {[f.name for f in files if f]}')
"

# Legacy pattern (without persistence)
cd user_strategies
uv run --active python -c "
import sys; sys.path.append('..')
from strategies.ml_strategy import MLWalkForwardStrategy
from backtesting import Backtest
from backtesting.test import EURUSD

data = EURUSD.iloc[:600]  # Minimum 600+ data points required
bt = Backtest(data, MLWalkForwardStrategy, commission=0.0002, margin=0.05, cash=10000)
stats = bt.run(n_train=200, retrain_frequency=20)
print(f'Return: {stats[\"Return [%]\"]:,.2f}%')
"
```

## Code Standards

### Backtesting.py Framework Standards
- **Commission**: Use `0.0002` (2 basis points) for realistic trading costs
- **Cash**: Use `10_000_000` as universal cash amount (works for all assets)
- **Temporal Integrity**: ZERO TOLERANCE for look-ahead bias - never fit() on future data
- **Strategy Pattern**: Separate LONG/SHORT strategies, never unified implementations
- **Benchmark Comparison**: All strategies MUST include benchmark comparison

### ML Strategy Requirements
- **Data Minimums**: 600+ data points for reliable operation (200+ for training, 400+ for testing)
- **Feature Engineering**: All features normalized by current price for scale invariance
- **Retraining**: Walk-forward retraining every 10-20 periods for market adaptation
- **Risk Management**: Dynamic stop-losses, position sizing, time-based trade management

## Testing Philosophy

### Coverage Standards
- **Current baseline**: 98% test coverage (2,497 statements, 56 missing)
- **Target**: 100% line and branch coverage
- **Missing areas**: Edge cases in backtesting.py (12 lines), _util.py (17 lines), others (27 lines)

### Test Organization
- **Original tests**: `backtesting/test/_test.py` (comprehensive framework testing)
- **User tests**: `user_strategies/tests/` (pytest-based, focused on ML strategies)
- **Separation principle**: User tests completely separate from original maintainer tests

## Key Implementation Notes

### ML Strategy Data Flow
1. **Feature Engineering**: `create_features()` → Technical indicators + temporal features
2. **Data Preparation**: `get_clean_Xy()` → Remove NaN values, ensure temporal alignment
3. **Model Training**: Initial fit on first N samples, periodic retraining on rolling window
4. **Prediction**: Single-step-ahead classification (-1/0/1 for short/neutral/long)
5. **Risk Management**: Dynamic stop-loss/take-profit, time-based position management

### Critical Performance Metrics
**MLWalkForwardStrategy proven results**:
- Return: 14.14%, Sharpe: 3.42, Max Drawdown: -2.33%, Win Rate: 90.5%

### Optimization Guidelines
- **Parameter scanning**: Use `bt.optimize()` with proper constraints
- **Cross-validation**: Implement walk-forward optimization for ML models
- **Overfitting prevention**: Always use out-of-sample testing periods

## Current Focus: Gapless Crypto Data Integration

**Active Development**: Transitioning from EURUSD test data to `gapless-crypto-data` package for authentic market data integration.

**Working Test Pattern**: `run_ml_strategy_with_persistence()` function serves as primary development interface with automatic output to `user_strategies/data/`:
- **CSV**: Performance metrics and trades data
- **HTML**: Interactive Bokeh charts for visualization
- **Timestamped files**: Prevents overwrites during iterative development

**Next Integration Steps**:
1. Replace `backtesting.test.EURUSD` with `gapless-crypto-data`
2. Validate temporal integrity with authentic crypto market data
3. Adapt feature engineering for crypto-specific market microstructure
4. Optimize ML model parameters for crypto volatility patterns

## Project Intentions & Historical Context

This project focuses on **comprehensive testing and validation of ML walk-forward strategies** using the backtesting.py framework. Key historical achievements:

1. **Framework Analysis**: Completed 98% test coverage analysis, identified 56 missing lines across core modules
2. **Strategy Extraction**: Identified "Trading with Machine Learning" as most robust strategy implementation
3. **Clean Architecture**: Established separation of concerns between original codebase and user modifications
4. **ML Validation**: Verified MLWalkForwardStrategy functionality with production-ready performance metrics
5. **Testing Infrastructure**: Created comprehensive pytest-based testing for user strategies
6. **Persistent Output**: Implemented automatic file persistence system (CSV + HTML) in proper directory structure

The ultimate goal is to develop robust, temporally-sound ML trading strategies with authentic crypto market data while maintaining the integrity and testability of the underlying backtesting framework.