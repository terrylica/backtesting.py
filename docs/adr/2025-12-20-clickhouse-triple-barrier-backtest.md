---
status: accepted
date: 2025-12-20
decision-maker: Terry Li
consulted:
  [
    gapless-crypto-clickhouse-explorer,
    backtesting-hf-limits-explorer,
    local-data-pipeline-explorer,
    gapless-ecosystem-explorer,
  ]
research-method: 9-agent-parallel-dctl
clarification-iterations: 4
perspectives: [DataEngineering, MLOps, QuantitativeFinance, SystemsIntegration]
---

# ADR: ClickHouse-Powered Triple-Barrier Probabilistic Classification for Backtesting

**Design Spec**: [Implementation Spec](/docs/design/2025-12-20-clickhouse-triple-barrier-backtest/spec.md)

## Context and Problem Statement

The backtesting.py framework excels at daily/hourly single-asset strategy simulation with simple signals. However, we have access to production-grade second-level crypto data via `gapless-crypto-clickhouse` with zero-gap guarantees and rich microstructure features (taker buy ratio, order flow, trade counts).

**The challenge**: How do we leverage this sophisticated data infrastructure for research-grade probabilistic classification while staying within backtesting.py's idiomatic usage patterns?

**Prior research (11 phases, 17 strategies)** proved that directional trading on crypto intraday (5-min to 1-hour) is non-viable. Higher timeframes performed WORSE (counter-intuitive). This ADR proposes a fundamentally different approach: **probabilistic classification with triple-barrier labels** rather than directional prediction.

### Before/After

<!-- graph-easy source:
[ gapless-crypto-clickhouse\n1-second data ] -> [ CSV files ] -> [ backtesting.py\nDaily strategies ] -> [ -4.96% returns\n17/17 failed ]
-->

```
+-------------------------+     +------------+     +----------------+     +----------------+
| gapless-crypto-         | --> | CSV files  | --> | backtesting.py | --> | -4.96% returns |
| clickhouse              |     |            |     | Daily          |     | 17/17 failed   |
| 1-second data           |     |            |     | strategies     |     |                |
+-------------------------+     +------------+     +----------------+     +----------------+
```

**AFTER:**

<!-- graph-easy source:
[ gapless-crypto-clickhouse\n1-second data ] -> [ ClickHouse\nAnalytics Engine ] -> [ Microstructure\nFeatures + Labels ] -> [ Probabilistic\nClassifier ] -> [ backtesting.py\nThreshold Strategy ] -> [ Calibrated P(+)\nResearch-grade ]
-->

```
+---------------+     +-----------+     +--------------+     +-------------+     +---------------+     +--------------+
| gapless-      | --> | ClickHouse| --> | Microstructure| --> | Probabilistic| --> | backtesting.py| --> | Calibrated   |
| crypto-       |     | Analytics |     | Features +   |     | Classifier  |     | Threshold    |     | P(+)         |
| clickhouse    |     | Engine    |     | Labels       |     |             |     | Strategy     |     | Research-    |
| 1-second data |     |           |     |              |     |             |     |              |     | grade        |
+---------------+     +-----------+     +--------------+     +-------------+     +---------------+     +--------------+
```

## Research Summary

| Agent Perspective         | Key Finding                                                                        | Confidence |
| ------------------------- | ---------------------------------------------------------------------------------- | ---------- |
| gapless-crypto-clickhouse | 1s-1mo timeframes, 11-column OHLCV+microstructure, zero-gap guarantee, auto-ingest | High       |
| backtesting-hf-limits     | 1 decision/bar, 10K plot limit, ~5M bar memory ceiling, fixed spread only          | High       |
| local-data-pipeline       | No ClickHouse integration present, only CSV, daily/hourly usage                    | High       |
| gapless-ecosystem         | Production-grade infrastructure far exceeds backtesting.py's design target         | High       |

## Decision Log

| Decision Area     | Options Evaluated                                                                       | Chosen                                | Rationale                                                    |
| ----------------- | --------------------------------------------------------------------------------------- | ------------------------------------- | ------------------------------------------------------------ |
| Primary Goal      | HFT, Intraday Directional, Feature Engineering, Data Validation                         | Feature Engineering + Data Validation | Prior research proved directional fails; research-only focus |
| Framework         | Adapt backtesting.py, Switch to hftbacktest, Build custom, Explore alternatives         | Adapt backtesting.py                  | Idiomatic usage, existing infrastructure                     |
| ClickHouse Depth  | Data source only, Analytics engine, Full pipeline                                       | Analytics engine                      | Complex aggregations before backtesting                      |
| Timeout Class Y=0 | Keep as 3rd class, Exclude, Weighted exclusion                                          | Keep as 3rd class                     | Most information-preserving                                  |
| Features          | Sequence view, Functional view, Microstructure aggregates, Hybrid                       | Hybrid                                | Combine sequence + ClickHouse aggregates                     |
| Event Trigger     | Candlestick patterns, Volatility compression, Microstructure anomaly, Regular intervals | Regular intervals                     | More events, simpler, less signal concentration              |
| Barriers b, H     | Adaptive ATR-based, Fixed percentage, Fixed ticks, Optimize via CV                      | Optimize via CV                       | Treat as hyperparameters with purged CV                      |
| Strategy Use      | Threshold entry, Kelly sizing, Probability-weighted, Research only                      | Threshold entry                       | Threshold optimized via backtesting.py                       |
| Backtest TF       | Hourly, Daily, 15-minute, Match event interval                                          | Match event interval                  | Exact alignment with event frequency                         |

### Trade-offs Accepted

| Trade-off                        | Choice                  | Accepted Cost                                               |
| -------------------------------- | ----------------------- | ----------------------------------------------------------- |
| Simplicity vs Accuracy           | 3-class softmax         | More complex than binary, but preserves timeout information |
| Flexibility vs Sample Efficiency | Hybrid features         | Requires more data than pure functional approach            |
| Throughput vs Resolution         | Event interval matching | May reduce trade frequency, but ensures temporal integrity  |

## Decision Drivers

- Prior research proved directional crypto intraday non-viable (0/17 strategies)
- gapless-crypto-clickhouse provides production-grade second-level data
- backtesting.py designed for simple signals, not HFT
- Research-grade probability calibration requires proper scoring rules
- Temporal integrity requires purged/embargoed cross-validation

## Considered Options

- **Option A: HFT with hftbacktest**: Purpose-built for tick data, L2/L3 orderbook. Rejected: steep learning curve, overkill for research
- **Option B: Custom event-driven framework**: Maximum flexibility. Rejected: high effort, reinventing the wheel
- **Option C: Adapt backtesting.py with ClickHouse analytics**: Use ClickHouse for aggregation, backtesting.py for simulation. **Selected**
- **Option D: Pure backtesting.py without ClickHouse**: Continue current approach. Rejected: already proven non-viable

## Decision Outcome

Chosen option: **Option C - Adapt backtesting.py with ClickHouse analytics**, because:

1. Leverages existing production-grade data infrastructure (gapless-crypto-clickhouse)
2. Uses ClickHouse for what it excels at (fast OLAP aggregations)
3. Uses backtesting.py idiomatically (strategy simulation at event interval)
4. Maintains research-grade ML hygiene (purged CV, proper scoring rules)
5. Avoids fighting framework limitations

## Synthesis

**Convergent findings**: All perspectives agreed that backtesting.py has hard limits for HFT (1 decision/bar, memory constraints), but works well for daily/hourly strategies.

**Divergent findings**: Whether to abandon backtesting.py entirely vs. adapt it. gapless-crypto-clickhouse perspective favored full custom solution; backtesting-hf-limits favored sticking with proven framework.

**Resolution**: Hybrid approach - use ClickHouse as analytics engine to pre-compute features and labels, then feed aggregated data to backtesting.py at the event interval granularity.

## Consequences

### Positive

- Leverages $50K+ data infrastructure investment (gapless-crypto-clickhouse)
- Research-grade probability calibration with proper scoring rules
- Maintains temporal integrity with purged/embargoed CV
- Uses backtesting.py idiomatically (no framework fighting)
- Clear separation: ClickHouse for analytics, backtesting.py for simulation

### Negative

- Two-system complexity (ClickHouse + backtesting.py)
- Requires ClickHouse running locally or cloud access
- Feature engineering happens outside backtesting.py
- Not suitable for live trading (research-only)

## Architecture

<!-- graph-easy source:
[ gapless-crypto-clickhouse ] { label: "1. Data Source"; } -> [ ClickHouse Analytics ] { label: "2. Aggregation"; }
[ ClickHouse Analytics ] -> [ Microstructure SQL\n(taker ratio, VWAP, trade count) ] { label: "2a. Features"; }
[ ClickHouse Analytics ] -> [ Triple-Barrier Labels\nY in {+,-,0} ] { label: "2b. Labels"; }
[ Microstructure SQL\n(taker ratio, VWAP, trade count) ] -> [ Probabilistic Classifier ] { label: "3. ML"; }
[ Triple-Barrier Labels\nY in {+,-,0} ] -> [ Probabilistic Classifier ]
[ Probabilistic Classifier ] -> [ Purged/Embargoed CV\nProper Scoring Rules ] { label: "3a. Validation"; }
[ Probabilistic Classifier ] -> [ pi(x) = (pi+, pi-, pi0) ] { label: "3b. Output"; }
[ pi(x) = (pi+, pi-, pi0) ] -> [ backtesting.py Strategy ] { label: "4. Backtest"; }
[ backtesting.py Strategy ] -> [ Threshold Optimization\nVisualization ] { label: "4a. Results"; }
-->

```
+---------------------------+
| gapless-crypto-clickhouse |
| 1. Data Source            |
+-------------+-------------+
              |
              v
+---------------------------+      +---------------------------+
| ClickHouse Analytics      | ---> | Microstructure SQL        |
| 2. Aggregation            |      | (taker ratio, VWAP,       |
+-------------+-------------+      | trade count)              |
              |                    | 2a. Features              |
              |                    +-------------+-------------+
              v                                  |
+---------------------------+                    |
| Triple-Barrier Labels     |                    |
| Y in {+,-,0}              |                    |
| 2b. Labels                |                    |
+-------------+-------------+                    |
              |                                  |
              +----------------+-----------------+
                               |
                               v
              +---------------------------+
              | Probabilistic Classifier  |
              | 3. ML                     |
              +-------------+-------------+
                            |
              +-------------+-------------+
              |                           |
              v                           v
+---------------------------+   +---------------------------+
| Purged/Embargoed CV       |   | pi(x) = (pi+, pi-, pi0)   |
| Proper Scoring Rules      |   | 3b. Output                |
| 3a. Validation            |   +-------------+-------------+
+---------------------------+                 |
                                              v
                            +---------------------------+
                            | backtesting.py Strategy   |
                            | 4. Backtest               |
                            +-------------+-------------+
                                          |
                                          v
                            +---------------------------+
                            | Threshold Optimization    |
                            | Visualization             |
                            | 4a. Results               |
                            +---------------------------+
```

## References

- [Crypto Intraday Research Termination](/user_strategies/research/CRYPTO_INTRADAY_RESEARCH_TERMINATION.md) - 11 phases, 17 strategies failed
- [Compression Breakout Research](/user_strategies/research/compression_breakout_research/README.md) - Phase 10 regime-aware trading
- [gapless-crypto-clickhouse](https://pypi.org/project/gapless-crypto-clickhouse/) - Data source package
- [Triple Barrier Labeling](https://www.newsletter.quantreo.com/p/the-triple-barrier-labeling-of-marco) - Marco Lopez de Prado's method
- [Proper Scoring Rules](https://en.wikipedia.org/wiki/Scoring_rule) - ML evaluation standard
