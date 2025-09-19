# Data Pipeline Study

This directory contains comprehensive studies of the backtesting.py data pipeline flow using loguru for detailed debugging.

## Purpose

Understanding the complete data flow from input DataFrame through feature engineering to strategy execution before integrating gapless-crypto-data package.

## Files

### Study Scripts

- **`enhanced_debug_pipeline.py`** - Comprehensive pipeline analysis with deep instrumentation
  - Uses 600 data points for proper feature engineering
  - Ultra-detailed logging of every step in the data pipeline
  - Traces DataFrame operations, feature creation, and model training
  - **Primary script for understanding data flow**

- **`debug_data_pipeline.py`** - Initial pipeline analysis (basic version)
  - Original debugging attempt with 100 data points
  - Revealed the dropna() issue with insufficient data
  - Monkey-patches key functions for logging

- **`explore_gapless_crypto.py`** - Gapless-crypto-data package exploration
  - API discovery for the gapless-crypto-data package
  - Tests different function names and parameters
  - Prepares for crypto data integration

### Log Files

- **`debug_pipeline.log`** - Initial debug session logs
- **`enhanced_pipeline.log`** - (Generated when running enhanced script)

## Key Insights Discovered

### Data Pipeline Flow

1. **Input**: pandas DataFrame with OHLCV columns + DatetimeIndex
2. **Feature Engineering**:
   - Technical indicators (SMA10, SMA20, SMA50, SMA100)
   - Bollinger Bands
   - Normalized features (price ratios)
   - Delta features (moving average differences)
   - Temporal features (day, hour)
3. **Data Cleaning**: `dropna()` removes rows with NaN values
4. **Critical Issue**: SMA100 requires 100+ periods, dramatically reduces available data

### Feature Engineering Requirements

- **Minimum Data**: 100+ periods for SMA100 indicator
- **Recommended Data**: 200+ periods for reliable feature engineering
- **Data Loss**: Significant reduction after `dropna()` due to moving averages

### Strategy Initialization

- Model training requires clean (X, y) data after feature engineering
- Target variable (y) creation uses future price movements with threshold classification
- kNN classifier needs sufficient samples for training

## Usage

Run the enhanced pipeline study:

```bash
cd data_pipeline_study
uv run --active python enhanced_debug_pipeline.py
```

This will generate comprehensive logs showing exactly how data flows through the backtesting.py framework.

## Next Steps

1. **Complete Pipeline Understanding**: Run enhanced debug to trace every step
2. **Gapless-Crypto Integration**: Adapt pipeline for crypto data format
3. **Data Format Mapping**: Ensure crypto data matches expected OHLCV structure
4. **Temporal Integrity**: Validate no look-ahead bias in crypto data pipeline