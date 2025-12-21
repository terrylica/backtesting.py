---
status: accepted
date: 2025-12-20
decision-maker: Terry Li
consulted: [PerformanceEngineer, UpstreamIntegration]
research-method: single-agent
clarification-iterations: 2
perspectives: [PerformanceOptimization, UpstreamIntegration]
---

# ADR: Multi-Timeframe Strategy with Second-Granularity Stop-Loss

**Design Spec**: [Implementation Spec](/docs/design/2025-12-20-multi-timeframe-second-granularity/spec.md)

## Context and Problem Statement

Current backtesting strategies operate on hourly (1H) data, which limits stop-loss precision. When price moves adversely within an hour, the stop-loss only triggers at the next bar close, potentially missing the optimal exit point and incurring larger-than-necessary losses.

The user requires:

1. **Entry/exit signals** based on hourly indicators (SMA crossovers, etc.)
2. **Stop-loss execution** checked every second for maximum precision
3. **Dynamic trailing stops** that automatically move up to protect profits

This requires a multi-timeframe architecture: run backtests on 1-second data while deriving signals from hourly resampled indicators.

### Before/After

```
 ⏮️ Before: Hourly Stop-Loss Checks

     ╭────────────────────────╮
     │     1H OHLCV Data      │
     ╰────────────────────────╯
       │
       │
       ∨
     ┌────────────────────────┐
     │    Strategy.next()     │
     │      24 calls/day      │
     └────────────────────────┘
       │
       │
       ∨
     ┌────────────────────────┐
     │    Stop-Loss Check     │
     │   (hourly precision)   │
     └────────────────────────┘
       │
       │ miss optimal exit
       ∨
     ╔════════════════════════╗
     ║ [!] Potential Slippage ║
     ║  (up to 59 min delay)  ║
     ╚════════════════════════╝
```

<details>
<summary>graph-easy source (Before)</summary>

```
graph { label: "⏮️ Before: Hourly Stop-Loss Checks"; flow: south; }
[ 1H OHLCV Data ] { shape: rounded; }
[ Strategy.next() ] { label: "Strategy.next()\n24 calls/day"; }
[ Stop-Loss Check ] { label: "Stop-Loss Check\n(hourly precision)"; }
[ Potential Slippage ] { label: "[!] Potential Slippage\n(up to 59 min delay)"; border: double; }

[ 1H OHLCV Data ] -> [ Strategy.next() ]
[ Strategy.next() ] -> [ Stop-Loss Check ]
[ Stop-Loss Check ] -- miss optimal exit --> [ Potential Slippage ]
```

</details>

```
⏭️ After: Second-Granularity Stop-Loss

         ╭─────────────────────╮
         │    1s OHLCV Data    │
         ╰─────────────────────╯
           │
           │
           ∨
         ┌─────────────────────┐
         │   Strategy.next()   │
         │  86,400 calls/day   │
         └─────────────────────┘
           │
           │
           ∨
         ┌─────────────────────┐
         │  TrailingStrategy   │
         │  (ATR-based stops)  │
         └─────────────────────┘
           │
           │ protect profits
           ∨
         ╔═════════════════════╗
         ║  [+] Precise Exit   ║
         ║ (1-second accuracy) ║
         ╚═════════════════════╝
```

<details>
<summary>graph-easy source (After)</summary>

```
graph { label: "⏭️ After: Second-Granularity Stop-Loss"; flow: south; }
[ 1s OHLCV Data ] { shape: rounded; }
[ Strategy.next() ] { label: "Strategy.next()\n86,400 calls/day"; }
[ TrailingStrategy ] { label: "TrailingStrategy\n(ATR-based stops)"; }
[ Precise Exit ] { label: "[+] Precise Exit\n(1-second accuracy)"; border: double; }

[ 1s OHLCV Data ] -> [ Strategy.next() ]
[ Strategy.next() ] -> [ TrailingStrategy ]
[ TrailingStrategy ] -- protect profits --> [ Precise Exit ]
```

</details>

## Research Summary

| Agent Perspective   | Key Finding                                                                 | Confidence |
| ------------------- | --------------------------------------------------------------------------- | ---------- |
| PerformanceEngineer | 1s data = 86,400 bars/day; memory ~16MB/day; acceptable for short periods   | High       |
| UpstreamIntegration | `TrailingStrategy` in backtesting.lib provides idiomatic trailing stop impl | High       |
| UpstreamIntegration | `resample_apply('1H', func, data)` creates hourly indicators from 1s data   | High       |
| PerformanceEngineer | gapless-crypto-clickhouse supports `timeframe='1s'` for BTCUSDT             | High       |

## Decision Log

| Decision Area      | Options Evaluated                 | Chosen           | Rationale                                      |
| ------------------ | --------------------------------- | ---------------- | ---------------------------------------------- |
| Data granularity   | 1m, 1s                            | 1s               | Maximum precision; user accepts slower runtime |
| Stop-loss approach | Manual tracking, TrailingStrategy | TrailingStrategy | Idiomatic backtesting.py pattern; ATR-based    |
| Signal timeframe   | Hourly from resample, native 1H   | Hourly resample  | Maintains single dataset; temporal alignment   |

### Trade-offs Accepted

| Trade-off              | Choice      | Accepted Cost                               |
| ---------------------- | ----------- | ------------------------------------------- |
| Speed vs Precision     | Precision   | ~45s download per day; slower backtest runs |
| Memory vs Granularity  | Granularity | ~16MB per day of 1s data in memory          |
| Complexity vs Accuracy | Accuracy    | Multi-timeframe code more complex           |

## Decision Drivers

- Stop-loss precision: 86,400 checks/day vs 24 checks/day
- Profit protection: Dynamic trailing stops lock in gains
- Existing infrastructure: TrailingStrategy already handles ATR-based stops
- Data availability: gapless-crypto-clickhouse supports 1s BTCUSDT data

## Considered Options

- **Option A: Run on 1-minute data** - 1,440 bars/day; reasonable precision but not maximum
- **Option B: Run on 1-second data with hourly signals** - Maximum precision; uses `resample_apply` for signals ← Selected
- **Option C: Dual backtest (hourly signals + 1s verification)** - Complex; synchronization issues

## Decision Outcome

Chosen option: **Option B**, because:

1. User explicitly requested maximum accuracy: "1s data is most accurate"
2. `TrailingStrategy` provides idiomatic implementation
3. `resample_apply` cleanly separates signal timeframe from execution timeframe
4. Single dataset simplifies temporal alignment

## Synthesis

**Convergent findings**: Both perspectives confirmed feasibility of 1s data backtesting with hourly signals.

**Divergent findings**: None - the approach is well-supported by existing backtesting.py infrastructure.

**Resolution**: Proceed with `TrailingStrategy` + `resample_apply` pattern.

## Consequences

### Positive

- Stop-loss checks 86,400×/day instead of 24×/day
- Trailing stops automatically protect profits
- Reuses proven backtesting.py patterns (no custom implementation)
- Clean separation: hourly decisions, second-level execution

### Negative

- Slower backtest execution (more bars to process)
- Higher memory usage (~16MB per day of data)
- Data download time (~45s per day from ClickHouse)

## Architecture

```
                                         🏗️ Multi-Timeframe Architecture

╭───────────────────╮                   ┌──────────────────┐     ┌─────────────────┐                 ╔════════════╗
│    ClickHouse     │                   │  resample_apply  │     │ Hourly Signals  │  entry signal   ║ Entry/Exit ║
│      1s Data      │ ────────────────> │ ('1H', SMA, ...) │ ──> │ (SMA crossover) │ ──────────────> ║ Decisions  ║
╰───────────────────╯                   └──────────────────┘     └─────────────────┘                 ╚════════════╝
  │                                                                                                    ∧
  │                                                                                                    │
  ∨                                                                                                    │
┏━━━━━━━━━━━━━━━━━━━┓                                                                                  │
┃ TrailingStrategy  ┃  stop triggered                                                                  │
┃ set_trailing_sl() ┃ ─────────────────────────────────────────────────────────────────────────────────┘
┗━━━━━━━━━━━━━━━━━━━┛
```

<details>
<summary>graph-easy source</summary>

```
graph { label: "🏗️ Multi-Timeframe Architecture"; flow: east; }

[ ClickHouse\n1s Data ] { shape: rounded; }
[ resample_apply ] { label: "resample_apply\n('1H', SMA, ...)"; }
[ Hourly Signals ] { label: "Hourly Signals\n(SMA crossover)"; }
[ TrailingStrategy ] { label: "TrailingStrategy\nset_trailing_sl()"; border: bold; }
[ Entry/Exit\nDecisions ] { border: double; }

[ ClickHouse\n1s Data ] -> [ resample_apply ]
[ resample_apply ] -> [ Hourly Signals ]
[ Hourly Signals ] -- entry signal --> [ Entry/Exit\nDecisions ]
[ ClickHouse\n1s Data ] -> [ TrailingStrategy ]
[ TrailingStrategy ] -- stop triggered --> [ Entry/Exit\nDecisions ]
```

</details>

## References

- [Trade Efficiency Analysis](/user_strategies/docs/TRADE_EFFICIENCY_ANALYSIS.md) - MAE/MFE metrics for evaluating stop-loss effectiveness
- [backtesting.lib TrailingStrategy](https://kernc.github.io/backtesting.py/doc/backtesting/lib.html#backtesting.lib.TrailingStrategy) - Upstream documentation
- [gapless-crypto-clickhouse](https://github.com/terrylica/gapless-crypto-clickhouse) - 1s data source
