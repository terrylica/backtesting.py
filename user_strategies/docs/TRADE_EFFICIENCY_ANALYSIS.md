# Trade Efficiency Analysis Using MAE/MFE Metrics

**Date**: 2025-12-20
**ADR Reference**: 2025-12-20-trade-efficiency-mae-mfe
**Status**: Implemented

## Executive Summary

This document describes the implementation of MAE/MFE-based trade efficiency analysis for evaluating trading strategy performance. Unlike traditional Buy & Hold comparisons, this framework benchmarks each trade against its **maximum achievable profit during the holding period**, providing a fair and actionable measure of trade execution quality.

## Problem Statement

Traditional performance metrics (Return %, Sharpe Ratio) compare strategy returns to a passive benchmark. However, these metrics fail to answer a critical question:

> **Of the profit that was actually available during each trade, how much did we capture?**

A strategy can have negative returns but excellent execution (capturing most of available opportunities in a sideways market), or positive returns with poor execution (missing most upside in a trending market).

## Solution: MAE/MFE Efficiency Framework

### Key Concepts

| Metric                                | Definition                                               | Formula                                                                    |
| ------------------------------------- | -------------------------------------------------------- | -------------------------------------------------------------------------- |
| **MFE** (Maximum Favorable Excursion) | Best price reached during trade (in favorable direction) | Long: `(max(High) - Entry) / Entry`<br>Short: `(Entry - min(Low)) / Entry` |
| **MAE** (Maximum Adverse Excursion)   | Worst price reached during trade (adverse direction)     | Long: `(min(Low) - Entry) / Entry`<br>Short: `(Entry - max(High)) / Entry` |
| **Trade Efficiency**                  | Ratio of actual return to maximum possible return        | `Actual_Return / MFE`                                                      |
| **ETD** (End Trade Drawdown)          | Profit given back after reaching MFE                     | `MFE - Actual_Return`                                                      |

### Efficiency Interpretation

| Efficiency | Meaning                                             |
| ---------- | --------------------------------------------------- |
| 1.0        | Perfect capture - exited at the best possible price |
| 0.5        | Captured half of available profit                   |
| 0.0        | Broke even despite favorable price movement         |
| < 0        | Loss despite having profitable opportunity          |

## Mathematical Framework

For a trade entered at price P<sub>entry</sub> and exited at price P<sub>exit</sub>:

### Long Positions

```
MFE% = (max(High[entry:exit]) - P_entry) / P_entry × 100
MAE% = (min(Low[entry:exit]) - P_entry) / P_entry × 100
Actual% = (P_exit - P_entry) / P_entry × 100
Efficiency = Actual% / MFE%
```

### Short Positions

```
MFE% = (P_entry - min(Low[entry:exit])) / P_entry × 100
MAE% = (P_entry - max(High[entry:exit])) / P_entry × 100
Actual% = (P_entry - P_exit) / P_entry × 100
Efficiency = Actual% / MFE%
```

## Implementation

### Module Location

```
user_strategies/strategies/trade_efficiency.py
```

### Core Components

1. **`TradeEfficiencyMetrics`** - Dataclass for single trade metrics
2. **`TradeEfficiencyReport`** - Comprehensive analysis report with summary generation
3. **`calculate_mae_mfe()`** - Core calculation function
4. **`calculate_trade_efficiency()`** - Main analysis function
5. **`analyze_efficiency_by_direction()`** - Separate long/short analysis
6. **`print_comparison_table()`** - Long vs Short comparison output

### Usage Example

```python
from backtesting import Backtest
from user_strategies.strategies.trade_efficiency import (
    calculate_trade_efficiency,
    print_comparison_table,
)

# Run backtest
stats = bt.run()

# Calculate efficiency
report = calculate_trade_efficiency(stats['_trades'], ohlcv_data)
print(report.summary())

# Export to DataFrame
df = report.to_dataframe()
df.to_csv('trades_with_efficiency.csv', index=False)
```

### Separate Long/Short Analysis

Sequential blocking in backtesting can affect results when long and short trades compete. For fair comparison, run separate backtests:

```python
# Long-only backtest
stats_long = bt_long.run()
long_report = calculate_trade_efficiency(stats_long['_trades'], data)

# Short-only backtest
stats_short = bt_short.run()
short_report = calculate_trade_efficiency(stats_short['_trades'], data)

# Compare
print(print_comparison_table(long_report, short_report))
```

## Data Source: gapless-crypto-clickhouse

### Correct API Usage

The `gapless-crypto-clickhouse` package provides cached BTCUSDT data. **Critical**: use `index_type='datetime'` to get proper timestamps:

```python
import gapless_crypto_clickhouse as gcch

df = gcch.download(
    symbol='BTCUSDT',
    timeframe='5m',
    start='2024-01-01',
    end='2024-03-31',
    index_type='datetime'  # REQUIRED for proper timestamps
)

# Rename columns for backtesting.py
df = df.rename(columns={
    'open': 'Open', 'high': 'High',
    'low': 'Low', 'close': 'Close',
    'volume': 'Volume'
})
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
```

### Cache Behavior

The package uses ETag-based HTTP caching:

- First request downloads from Binance public data
- Subsequent requests show "Cache HIT" and load from local cache
- No network access required after initial cache population

## Generated Outputs

### HTML Backtest Visualizations

| File                                 | Description                       |
| ------------------------------------ | --------------------------------- |
| `efficiency_long_only_BTCUSDT.html`  | Long-only strategy visualization  |
| `efficiency_short_only_BTCUSDT.html` | Short-only strategy visualization |
| `efficiency_long_only_EURUSD.html`   | EURUSD long-only (sample data)    |
| `efficiency_short_only_EURUSD.html`  | EURUSD short-only (sample data)   |

### CSV Trade Data

| File                                        | Description                           |
| ------------------------------------------- | ------------------------------------- |
| `efficiency_long_only_BTCUSDT_trades.csv`   | Per-trade metrics for long positions  |
| `efficiency_short_only_BTCUSDT_trades.csv`  | Per-trade metrics for short positions |
| `efficiency_comparison_BTCUSDT_summary.csv` | Summary comparison statistics         |

### Sample Results (BTCUSDT 5m, Jan-Mar 2024)

```
============================================================
LONG vs SHORT EFFICIENCY COMPARISON
============================================================

Metric                          LONG-ONLY      SHORT-ONLY
------------------------------------------------------------
Trades                                522             361
Win Rate [%]                         38.9            34.1
Total MFE Available [%]            379.28          206.23
Total Captured [%]                  33.36          -10.94
Aggregate Efficiency [%]              8.8            -5.3
Mean Efficiency                    -0.611          -0.694
============================================================
```

**Interpretation**: Long positions captured 8.8% of available MFE (weak but positive), while short positions gave back 5.3% beyond their available MFE (indicating exits at worse than entry prices on average).

## Key Insights

1. **Direction Matters**: Long and short strategies behave differently in the same market conditions
2. **MFE as Benchmark**: Each trade is judged against its own achievable maximum, not market averages
3. **ETD for Exit Timing**: High ETD values indicate giving back too much profit before exit
4. **Sequential Blocking**: Run separate long-only and short-only backtests for fair comparison

## References

- Sweeney, John. "Maximum Adverse Excursion." _Technical Analysis of Stocks & Commodities_
- Lopez de Prado, M. "Advances in Financial Machine Learning" (Ch. 10: Bet Sizing)

## File Locations

```
user_strategies/
├── strategies/
│   └── trade_efficiency.py         # Core implementation
├── data/backtests/
│   ├── efficiency_*_BTCUSDT.html   # Crypto backtest outputs
│   ├── efficiency_*_EURUSD.html    # Forex backtest outputs
│   └── efficiency_*_trades.csv     # Per-trade efficiency data
└── docs/
    └── TRADE_EFFICIENCY_ANALYSIS.md # This documentation
```
