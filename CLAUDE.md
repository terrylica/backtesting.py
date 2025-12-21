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
user_strategies/                    # All user code (separate from original)
├── strategies/                     # Production strategies
│   ├── ml_strategy.py              # ML walk-forward strategy with crypto data support
│   └── __init__.py
├── tests/                          # Strategy tests
│   ├── test_ml_strategy.py         # ML strategy tests
│   └── __init__.py
├── configs/                        # Strategy configurations
├── data/                           # Organized persistent outputs
│   ├── trades/                     # Trade CSV outputs
│   ├── performance/                # Performance metrics
│   └── backtests/                  # HTML backtest visualizations
├── research/                       # Research and exploration
│   ├── data_pipeline/              # Pipeline studies
│   └── crypto_exploration/         # Crypto data exploration
├── logs/                           # Log files
├── outputs/                        # Temporary/test outputs
└── docs/                           # User strategy documentation
```

## Separation of Concerns Principle

### ABSOLUTE RULE: Never Modify Original backtesting.py Framework

This project maintains **complete isolation** between the original backtesting.py framework and user development work.

### Directory Ownership

**Maintainer-Owned (READ-ONLY - DO NOT TOUCH):**

- **`/backtesting/`** - Original framework code (backtesting.py, lib.py, \_stats.py, etc.)
- **`/doc/`** - Original documentation
- **`/setup.py`, `/setup.cfg`, `/MANIFEST.in`** - Original packaging configuration
- **`/README.md`, `/LICENSE.md`, `/CHANGELOG.md`, `/CONTRIBUTING.md`** - Original project documentation
- **`/.github/`** - Original GitHub workflows and configuration
- **`/.codecov.yml`** - Original code coverage configuration
- **`/requirements.txt`** - Original dependency specifications

**User-Owned (MODIFY FREELY):**

- **`/user_strategies/`** - All user strategies, tests, and configurations
- **`/CLAUDE.md`** - Claude-specific documentation (this file)
- **`/sessions/`** - Conversation history and session data
- **`/.claude/`** - Claude Code configuration

### Data Source Integration

The ML strategy now supports **dual data sources** with seamless switching:

```python
# EURUSD (traditional forex data)
data = get_data_source('EURUSD')

# Crypto data (authentic Binance data via gapless-crypto-data)
data = get_data_source('crypto', symbol='BTCUSDT', start='2024-01-01', end='2024-01-08')
```

Both data sources maintain identical OHLCV format and work seamlessly with all backtesting.py strategies.

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
- **Missing areas**: Edge cases in backtesting.py (12 lines), \_util.py (17 lines), others (27 lines)

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

## Current Focus: Market Regime Sensitivity Analysis

**Active Development**: Extended timeframe testing completed with 664 trades across 6 years (2019-2024) using authentic Binance data via `gapless-crypto-data`.

**Critical Discovery**: Temporal alignment effective short-term (+20.18% on 399 days) but degrades across market regimes (-4.96% on 6 years).

**Project Planning Files**:

- **`user_strategies/EXTENDED_TIMEFRAME_TESTING_PLAN.yml`**: Machine-readable execution specification with results
- **`user_strategies/EXTENDED_TIMEFRAME_ANALYSIS.md`**: Comprehensive analysis and next research directions
- **`user_strategies/TEMPORAL_ALIGNMENT_GUIDE.md`**: Optimal configuration documentation
- **`user_strategies/IN_SAMPLE_OUT_SAMPLE_ANALYSIS.md`**: Performance comparison analysis

**Next Research Direction**: Multi-timeframe validation with regime-specific configurations:

1. **14-day/14% configuration**: Medium-term stability testing
2. **3-day/3% configuration**: High-volatility periods
3. **Volatility-adaptive parameters**: Dynamic regime detection

## Project Intentions & Historical Context

This project focuses on **comprehensive testing and validation of ML walk-forward strategies** using the backtesting.py framework. Key historical achievements:

1. **Framework Analysis**: Completed 98% test coverage analysis, identified 56 missing lines across core modules
2. **Strategy Extraction**: Identified "Trading with Machine Learning" as most robust strategy implementation
3. **Clean Architecture**: Established separation of concerns between original codebase and user modifications
4. **Temporal Alignment Discovery**: Solved temporal mismatch (48-day predictions with 2% stops causing -99.94% losses)
5. **Extended Testing**: 664 trades across 6 years with statistical significance achieved
6. **Market Regime Analysis**: Identified regime sensitivity as key limitation for long-term deployment
7. **Session Protection**: Universal `.sessions/` protection system implemented for conversation history

The ultimate goal is to develop robust, temporally-sound ML trading strategies with regime-adaptive parameters for authentic crypto market deployment while maintaining the integrity and testability of the underlying backtesting framework.
