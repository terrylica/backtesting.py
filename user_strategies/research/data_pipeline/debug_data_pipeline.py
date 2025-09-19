#!/usr/bin/env python3
"""
Comprehensive debug logging of backtesting.py data pipeline flow using loguru

This script traces how data flows through the framework from input to strategy execution,
helping us understand the pipeline before integrating gapless-crypto-data.
"""
import sys
import numpy as np
import pandas as pd
from loguru import logger
from pathlib import Path

# Configure loguru for detailed debugging
logger.remove()  # Remove default handler
logger.add(
    sys.stdout,
    level="TRACE",
    format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    colorize=True
)

# Also log to file for analysis
log_file = Path("debug_pipeline.log")
logger.add(
    log_file,
    level="TRACE",
    format="{time} | {level: <8} | {name}:{function}:{line} - {message}",
    rotation="10 MB"
)

logger.info("🚀 Starting backtesting.py data pipeline analysis")

# Import and patch backtesting framework with logging
sys.path.append('user_strategies')

try:
    from backtesting import Backtest, Strategy
    from backtesting.test import EURUSD, SMA
    from strategies.ml_strategy import MLWalkForwardStrategy, create_features, get_clean_Xy

    logger.success("✅ Successfully imported backtesting framework")

    # Analyze the test data structure
    logger.info("📊 Analyzing EURUSD test data structure")
    logger.debug(f"EURUSD type: {type(EURUSD)}")
    logger.debug(f"EURUSD shape: {EURUSD.shape}")
    logger.debug(f"EURUSD columns: {list(EURUSD.columns)}")
    logger.debug(f"EURUSD index type: {type(EURUSD.index)}")
    logger.debug(f"EURUSD date range: {EURUSD.index[0]} to {EURUSD.index[-1]}")
    logger.debug(f"EURUSD dtypes:\n{EURUSD.dtypes}")

    # Sample a subset for detailed analysis
    data_subset = EURUSD.iloc[:100].copy()
    logger.info(f"📈 Using data subset: {data_subset.shape} rows")

    # Log data preprocessing in create_features
    logger.info("🔧 Tracing feature engineering pipeline")

    # Monkey patch create_features to add logging
    original_create_features = create_features

    def logged_create_features(data):
        logger.trace(f"create_features() input: shape={data.shape}, columns={list(data.columns)}")
        logger.trace(f"create_features() input dtypes:\n{data.dtypes}")
        logger.trace(f"create_features() first 3 rows:\n{data.head(3)}")

        result = original_create_features(data)

        logger.trace(f"create_features() output: shape={result.shape}")
        logger.trace(f"create_features() feature columns: {[col for col in result.columns if col.startswith('X_')]}")
        logger.trace(f"create_features() NaN count after dropna: {result.isna().sum().sum()}")
        logger.trace(f"create_features() output dtypes:\n{result.dtypes}")

        return result

    # Patch the function
    import strategies.ml_strategy
    strategies.ml_strategy.create_features = logged_create_features

    # Monkey patch get_clean_Xy to add logging
    original_get_clean_Xy = get_clean_Xy

    def logged_get_clean_Xy(df):
        logger.trace(f"get_clean_Xy() input: shape={df.shape}")

        X, y = original_get_clean_Xy(df)

        logger.trace(f"get_clean_Xy() X output: shape={X.shape}, dtype={X.dtype}")
        logger.trace(f"get_clean_Xy() y output: shape={y.shape}, dtype={y.dtype}")
        logger.trace(f"get_clean_Xy() y unique values: {np.unique(y)}")
        logger.trace(f"get_clean_Xy() X sample (first row): {X[0] if len(X) > 0 else 'empty'}")

        return X, y

    strategies.ml_strategy.get_clean_Xy = logged_get_clean_Xy

    # Monkey patch Backtest class methods
    original_backtest_init = Backtest.__init__

    def logged_backtest_init(self, data, strategy, **kwargs):
        logger.info(f"🏗️  Backtest.__init__() called")
        logger.debug(f"Backtest data type: {type(data)}")
        logger.debug(f"Backtest data shape: {data.shape}")
        logger.debug(f"Backtest strategy: {strategy}")
        logger.debug(f"Backtest kwargs: {kwargs}")

        # Call original init
        result = original_backtest_init(self, data, strategy, **kwargs)

        logger.debug(f"Backtest._data type after init: {type(self._data)}")
        logger.debug(f"Backtest._strategy type: {type(self._strategy)}")

        return result

    Backtest.__init__ = logged_backtest_init

    # Monkey patch Strategy class
    original_strategy_init = Strategy.init

    def logged_strategy_init(self):
        logger.info(f"🎯 Strategy.init() called for {self.__class__.__name__}")
        logger.debug(f"Strategy data available: {hasattr(self, 'data')}")
        if hasattr(self, 'data'):
            logger.debug(f"Strategy data type: {type(self.data)}")
            logger.debug(f"Strategy data length: {len(self.data)}")
            logger.debug(f"Strategy data columns: {list(self.data.df.columns) if hasattr(self.data, 'df') else 'no df'}")

        # Call original init
        if hasattr(self, '__orig_init__'):
            return self.__orig_init__()
        else:
            return original_strategy_init(self)

    # Patch MLWalkForwardStrategy specifically
    original_ml_init = MLWalkForwardStrategy.init

    def logged_ml_init(self):
        logger.info(f"🤖 MLWalkForwardStrategy.init() starting")

        # Store original for calling
        self.__orig_init__ = original_ml_init

        logger.debug(f"ML strategy data shape: {self.data.df.shape}")
        logger.debug(f"ML strategy n_train parameter: {self.n_train}")
        logger.debug(f"ML strategy retrain_frequency: {self.retrain_frequency}")

        result = original_ml_init(self)

        logger.debug(f"ML strategy after init - clf type: {type(self.clf)}")
        logger.debug(f"ML strategy after init - df_features shape: {self.df_features.shape}")

        return result

    MLWalkForwardStrategy.init = logged_ml_init

    # Monkey patch Strategy.next
    original_strategy_next = Strategy.next

    def logged_strategy_next(self):
        logger.trace(f"📊 Strategy.next() called - bar {len(self.data)}")
        if hasattr(self, '_next_call_count'):
            self._next_call_count += 1
        else:
            self._next_call_count = 1

        if self._next_call_count <= 5 or self._next_call_count % 50 == 0:
            logger.trace(f"Strategy.next() call #{self._next_call_count}")
            logger.trace(f"Current bar data: OHLC={[self.data.Open[-1], self.data.High[-1], self.data.Low[-1], self.data.Close[-1]]}")

        return original_strategy_next(self)

    Strategy.next = logged_strategy_next

    # Now run the backtest with full logging
    logger.info("🚀 Starting logged backtest execution")

    bt = Backtest(
        data_subset,
        MLWalkForwardStrategy,
        commission=0.0002,
        margin=0.05,
        cash=10000
    )

    logger.info("📈 Running backtest with debug logging...")
    stats = bt.run(n_train=50, retrain_frequency=10)

    logger.success("✅ Backtest completed successfully")
    logger.info(f"📊 Results: Return={stats['Return [%]']:.2f}%, Sharpe={stats['Sharpe Ratio']:.2f}")

    logger.info(f"📝 Full debug log saved to: {log_file.absolute()}")

except Exception as e:
    logger.error(f"❌ Error during pipeline analysis: {e}")
    logger.exception("Full traceback:")

logger.info("🏁 Data pipeline analysis complete")