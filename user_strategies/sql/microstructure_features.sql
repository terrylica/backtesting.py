-- Microstructure Feature SQL Templates for Triple-Barrier Classification
-- ADR: 2025-12-20-clickhouse-triple-barrier-backtest
--
-- These SQL templates compute microstructure features from second-level data
-- aggregated to the event interval (e.g., hourly). Features are designed for
-- probabilistic classification with proper temporal alignment.
--
-- Usage: Templates use {placeholders} for parameterized queries via Python.

-- =============================================================================
-- FEATURE 1: Taker Buy Ratio (Order Flow Imbalance Proxy)
-- =============================================================================
-- Measures the proportion of volume initiated by buyers (takers lifting asks).
-- Values > 0.5 indicate net buying pressure, < 0.5 indicate net selling.
--
-- Range: [0, 1]
-- Interpretation: Higher values = more aggressive buying

SELECT
    toStartOfInterval(timestamp, INTERVAL {interval_seconds} SECOND) AS interval_start,
    sum(taker_buy_base_asset_volume) / nullIf(sum(volume), 0) AS taker_buy_ratio
FROM {table_name}
WHERE symbol = {symbol}
  AND timestamp >= {start_time}
  AND timestamp < {end_time}
GROUP BY interval_start
ORDER BY interval_start;


-- =============================================================================
-- FEATURE 2: Order Flow Imbalance (Signed Volume Delta)
-- =============================================================================
-- Net order flow: (taker buy volume) - (taker sell volume)
-- Taker sell volume = total volume - taker buy volume
--
-- Range: Unbounded, centered around 0
-- Interpretation: Positive = net buying, Negative = net selling

SELECT
    toStartOfInterval(timestamp, INTERVAL {interval_seconds} SECOND) AS interval_start,
    sum(taker_buy_base_asset_volume) - sum(volume - taker_buy_base_asset_volume) AS order_flow_imbalance,
    -- Normalized version (as ratio of total volume)
    (sum(taker_buy_base_asset_volume) - sum(volume - taker_buy_base_asset_volume))
        / nullIf(sum(volume), 0) AS order_flow_imbalance_ratio
FROM {table_name}
WHERE symbol = {symbol}
  AND timestamp >= {start_time}
  AND timestamp < {end_time}
GROUP BY interval_start
ORDER BY interval_start;


-- =============================================================================
-- FEATURE 3: Trade Intensity (Activity Metrics)
-- =============================================================================
-- Measures trading activity via trade count aggregations.
-- High trade counts may indicate increased volatility or interest.
--
-- Metrics:
--   - total_trades: Sum of trades in interval
--   - avg_trades_per_bar: Mean trades per source bar (e.g., per second)
--   - trade_intensity_zscore: Standardized relative to rolling window (computed in Python)

SELECT
    toStartOfInterval(timestamp, INTERVAL {interval_seconds} SECOND) AS interval_start,
    sum(number_of_trades) AS total_trades,
    avg(number_of_trades) AS avg_trades_per_bar,
    count(*) AS bar_count
FROM {table_name}
WHERE symbol = {symbol}
  AND timestamp >= {start_time}
  AND timestamp < {end_time}
GROUP BY interval_start
ORDER BY interval_start;


-- =============================================================================
-- FEATURE 4: VWAP Deviation (Price Location)
-- =============================================================================
-- Measures how far the closing price is from the volume-weighted average price.
-- Positive deviation = price above VWAP (potential overbought)
-- Negative deviation = price below VWAP (potential oversold)
--
-- Range: Unbounded, typically small percentages
-- Note: Normalized by VWAP for scale invariance

SELECT
    toStartOfInterval(timestamp, INTERVAL {interval_seconds} SECOND) AS interval_start,
    -- VWAP calculation
    sum(close * volume) / nullIf(sum(volume), 0) AS vwap,
    -- Last close in interval
    argMax(close, timestamp) AS last_close,
    -- VWAP deviation (absolute)
    argMax(close, timestamp) - sum(close * volume) / nullIf(sum(volume), 0) AS vwap_deviation,
    -- VWAP deviation (percentage)
    (argMax(close, timestamp) - sum(close * volume) / nullIf(sum(volume), 0))
        / nullIf(sum(close * volume) / nullIf(sum(volume), 0), 0) AS vwap_deviation_pct
FROM {table_name}
WHERE symbol = {symbol}
  AND timestamp >= {start_time}
  AND timestamp < {end_time}
GROUP BY interval_start
ORDER BY interval_start;


-- =============================================================================
-- FEATURE 5: Realized Volatility (Intrabar Price Dispersion)
-- =============================================================================
-- Computes realized volatility from high-frequency returns within each interval.
-- Uses log returns for better statistical properties.
--
-- Note: stddevPop for population std dev; annualization factor applied in Python
-- based on the source data frequency.

SELECT
    toStartOfInterval(timestamp, INTERVAL {interval_seconds} SECOND) AS interval_start,
    -- Standard deviation of log returns within interval
    stddevPop(log(close / lagInFrame(close, 1) OVER (ORDER BY timestamp))) AS realized_vol_raw,
    -- High-Low range as volatility proxy
    (max(high) - min(low)) / nullIf(avg(close), 0) AS range_volatility,
    -- Number of observations for volatility calculation
    count(*) AS obs_count
FROM {table_name}
WHERE symbol = {symbol}
  AND timestamp >= {start_time}
  AND timestamp < {end_time}
GROUP BY interval_start
ORDER BY interval_start;


-- =============================================================================
-- COMBINED FEATURE QUERY (All Features in One Pass)
-- =============================================================================
-- Efficient single-pass query to compute all microstructure features.
-- Use this for production feature generation.

SELECT
    toStartOfInterval(timestamp, INTERVAL {interval_seconds} SECOND) AS interval_start,

    -- OHLCV aggregates
    argMin(open, timestamp) AS open,
    max(high) AS high,
    min(low) AS low,
    argMax(close, timestamp) AS close,
    sum(volume) AS volume,

    -- Feature 1: Taker Buy Ratio
    sum(taker_buy_base_asset_volume) / nullIf(sum(volume), 0) AS taker_buy_ratio,

    -- Feature 2: Order Flow Imbalance
    (2 * sum(taker_buy_base_asset_volume) - sum(volume)) / nullIf(sum(volume), 0) AS order_flow_imbalance,

    -- Feature 3: Trade Intensity
    sum(number_of_trades) AS total_trades,
    avg(number_of_trades) AS avg_trades_per_bar,

    -- Feature 4: VWAP and Deviation
    sum(close * volume) / nullIf(sum(volume), 0) AS vwap,
    (argMax(close, timestamp) - sum(close * volume) / nullIf(sum(volume), 0))
        / nullIf(sum(close * volume) / nullIf(sum(volume), 0), 0) AS vwap_deviation_pct,

    -- Feature 5: Range Volatility
    (max(high) - min(low)) / nullIf(avg(close), 0) AS range_volatility,

    -- Metadata
    count(*) AS source_bar_count,
    sum(quote_asset_volume) AS quote_volume

FROM {table_name}
WHERE symbol = {symbol}
  AND timestamp >= {start_time}
  AND timestamp < {end_time}
GROUP BY interval_start
ORDER BY interval_start;
