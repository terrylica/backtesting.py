#!/usr/bin/env python3
"""
Enhanced debug logging of backtesting.py data pipeline flow with deeper instrumentation

This version uses sufficient data and adds even more granular tracing to understand
the complete data flow from input DataFrame to Strategy execution.
"""
import sys
import numpy as np
import pandas as pd
from loguru import logger
from pathlib import Path
import inspect

# Configure loguru for maximum debugging detail
logger.remove()
logger.add(
    sys.stdout,
    level="TRACE",
    format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    colorize=True
)

log_file = Path("enhanced_pipeline.log")
logger.add(
    log_file,
    level="TRACE",
    format="{time} | {level: <8} | {name}:{function}:{line} - {message}",
    rotation="50 MB"
)

logger.info("🚀 Enhanced backtesting.py data pipeline analysis")

sys.path.append('..')  # Add parent directory to path
sys.path.append('../user_strategies')  # Add user_strategies to path

try:
    from backtesting import Backtest, Strategy
    from backtesting.test import EURUSD, SMA
    from strategies.ml_strategy import MLWalkForwardStrategy, create_features, get_clean_Xy, get_y, get_X

    logger.success("✅ Successfully imported backtesting framework")

    # Use sufficient data for proper analysis
    data_subset = EURUSD.iloc[:600].copy()  # Much larger sample
    logger.info(f"📈 Using enhanced data subset: {data_subset.shape} rows")

    # Deep analysis of data structure
    logger.info("🔍 Deep data structure analysis")
    logger.debug(f"Data memory usage:\n{data_subset.memory_usage(deep=True)}")
    logger.debug(f"Data info:\n{data_subset.info()}")
    logger.debug(f"Data describe:\n{data_subset.describe()}")

    # Patch pandas DataFrame methods to trace data flow
    original_df_getitem = pd.DataFrame.__getitem__
    original_df_loc = pd.DataFrame.loc.__getitem__
    original_df_iloc = pd.DataFrame.iloc.__getitem__

    def logged_df_getitem(self, key):
        result = original_df_getitem(self, key)
        if isinstance(key, str) and key in ['Open', 'High', 'Low', 'Close', 'Volume']:
            logger.trace(f"DataFrame['{key}'] accessed: shape={result.shape if hasattr(result, 'shape') else 'scalar'}")
        return result

    pd.DataFrame.__getitem__ = logged_df_getitem

    # Enhanced feature engineering tracing
    original_create_features = create_features

    def ultra_logged_create_features(data):
        logger.info(f"🔧 create_features() ENTRY")
        logger.debug(f"  Input DataFrame: shape={data.shape}, memory={data.memory_usage().sum()} bytes")
        logger.debug(f"  Input index range: {data.index[0]} to {data.index[-1]}")
        logger.debug(f"  Input columns: {list(data.columns)}")

        # Step by step feature creation
        df = data.copy()
        logger.trace(f"  After copy: shape={df.shape}")

        close = df.Close.values
        logger.trace(f"  Close array: shape={close.shape}, sample=[{close[0]:.5f}, {close[1]:.5f}, ...]")

        # Trace each technical indicator calculation
        logger.trace("  Calculating SMA indicators...")
        sma10 = SMA(df.Close, 10)
        logger.trace(f"    SMA10: NaN count={pd.isna(sma10).sum()}, first valid at index {pd.notna(sma10).idxmax()}")

        sma20 = SMA(df.Close, 20)
        logger.trace(f"    SMA20: NaN count={pd.isna(sma20).sum()}, first valid at index {pd.notna(sma20).idxmax()}")

        sma50 = SMA(df.Close, 50)
        logger.trace(f"    SMA50: NaN count={pd.isna(sma50).sum()}, first valid at index {pd.notna(sma50).idxmax()}")

        sma100 = SMA(df.Close, 100)
        logger.trace(f"    SMA100: NaN count={pd.isna(sma100).sum()}, first valid at index {pd.notna(sma100).idxmax()}")

        from strategies.ml_strategy import bbands
        upper, lower = bbands(df, 20, 2)
        logger.trace(f"    Bollinger Bands: upper NaN count={pd.isna(upper).sum()}, lower NaN count={pd.isna(lower).sum()}")

        # Add features one by one with detailed logging
        logger.trace("  Adding normalized price features...")
        df['X_SMA10'] = (close - sma10) / close
        df['X_SMA20'] = (close - sma20) / close
        df['X_SMA50'] = (close - sma50) / close
        df['X_SMA100'] = (close - sma100) / close

        logger.trace("  Adding delta features...")
        df['X_DELTA_SMA10'] = (sma10 - sma20) / close
        df['X_DELTA_SMA20'] = (sma20 - sma50) / close
        df['X_DELTA_SMA50'] = (sma50 - sma100) / close

        logger.trace("  Adding technical indicator features...")
        df['X_MOM'] = df.Close.pct_change(periods=2)
        df['X_BB_upper'] = (upper - close) / close
        df['X_BB_lower'] = (lower - close) / close
        df['X_BB_width'] = (upper - lower) / close

        logger.trace("  Adding temporal features...")
        df['X_day'] = df.index.dayofweek
        df['X_hour'] = df.index.hour
        df['X_Sentiment'] = 1.0

        logger.debug(f"  Before dropna: shape={df.shape}, total NaN={df.isna().sum().sum()}")

        # Detailed NaN analysis before dropna
        nan_counts = df.isna().sum()
        nan_features = nan_counts[nan_counts > 0]
        if len(nan_features) > 0:
            logger.debug(f"  NaN counts by column:\n{nan_features}")

        result = df.dropna().astype(float)
        logger.debug(f"  After dropna: shape={result.shape}, total NaN={result.isna().sum().sum()}")

        if len(result) > 0:
            feature_cols = [col for col in result.columns if col.startswith('X_')]
            logger.debug(f"  Feature columns: {feature_cols}")
            logger.trace(f"  First feature row sample:\n{result[feature_cols].iloc[0]}")

        logger.info(f"🔧 create_features() EXIT: {data.shape} → {result.shape}")
        return result

    # Enhanced get_y tracing
    original_get_y = get_y

    def ultra_logged_get_y(data, forecast_periods=48, threshold=0.004):
        logger.trace(f"get_y() ENTRY: forecast_periods={forecast_periods}, threshold={threshold}")
        logger.trace(f"  Input data shape: {data.shape}")

        y = data.Close.pct_change(forecast_periods).shift(-forecast_periods)
        logger.trace(f"  After pct_change+shift: NaN count={pd.isna(y).sum()}, valid count={pd.notna(y).sum()}")

        # Count values in each category before classification
        above_thresh = (y > threshold).sum()
        below_thresh = (y < -threshold).sum()
        between_thresh = y.between(-threshold, threshold).sum()

        logger.trace(f"  Value distribution: above_thresh={above_thresh}, below_thresh={below_thresh}, between_thresh={between_thresh}")

        y[y.between(-threshold, threshold)] = 0
        y[y > 0] = 1
        y[y < 0] = -1

        final_counts = y.value_counts().sort_index()
        logger.trace(f"  Final classification counts:\n{final_counts}")
        logger.trace(f"get_y() EXIT: shape={y.shape}")

        return y

    # Enhanced get_clean_Xy tracing
    original_get_clean_Xy = get_clean_Xy

    def ultra_logged_get_clean_Xy(df):
        logger.trace(f"get_clean_Xy() ENTRY: shape={df.shape}")

        X = get_X(df)
        logger.trace(f"  get_X() result: shape={X.shape}")

        y = ultra_logged_get_y(df).values
        logger.trace(f"  get_y() result: shape={y.shape}")

        isnan = np.isnan(y)
        nan_count = isnan.sum()
        valid_count = (~isnan).sum()

        logger.trace(f"  NaN analysis: {nan_count} NaN, {valid_count} valid")

        X_clean = X[~isnan]
        y_clean = y[~isnan]

        logger.trace(f"get_clean_Xy() EXIT: X={X_clean.shape}, y={y_clean.shape}")
        return X_clean, y_clean

    # Enhanced backtest tracing
    original_backtest_run = Backtest.run

    def ultra_logged_backtest_run(self, **kwargs):
        logger.info(f"🚀 Backtest.run() ENTRY with kwargs: {kwargs}")

        # Trace strategy instantiation
        logger.debug(f"  Strategy class: {self._strategy}")
        logger.debug(f"  Data shape: {self._data.shape}")

        return original_backtest_run(self, **kwargs)

    # Enhanced strategy init tracing
    original_strategy_init = MLWalkForwardStrategy.init

    def ultra_logged_ml_init(self):
        logger.info(f"🤖 MLWalkForwardStrategy.init() ENTRY")
        logger.debug(f"  Strategy parameters: n_train={self.n_train}, retrain_freq={self.retrain_frequency}")
        logger.debug(f"  Available data: shape={self.data.df.shape}")

        # Initialize kNN classifier
        from sklearn.neighbors import KNeighborsClassifier
        self.clf = KNeighborsClassifier(self.n_neighbors)
        logger.debug(f"  Initialized kNN classifier: {self.clf}")

        # Create features for entire dataset with detailed logging
        logger.info("  🔧 Creating features for entire dataset...")
        df_with_features = ultra_logged_create_features(self.data.df)
        logger.info(f"  ✅ Features created: {self.data.df.shape} → {df_with_features.shape}")

        # Train on first n_train samples
        logger.info(f"  🎯 Training on first {self.n_train} samples...")
        train_df = df_with_features.iloc[:self.n_train]
        logger.debug(f"    Train data shape: {train_df.shape}")

        X_train, y_train = ultra_logged_get_clean_Xy(train_df)
        logger.info(f"    Clean training data: X={X_train.shape}, y={y_train.shape}")

        if len(X_train) > 0 and len(y_train) > 0:
            self.clf.fit(X_train, y_train)
            logger.success(f"  ✅ Model trained successfully on {len(X_train)} samples")

            # Log training data characteristics
            unique_y = np.unique(y_train)
            y_counts = np.bincount(y_train.astype(int) + 1)  # Shift to positive indices
            logger.debug(f"    Training labels distribution: {dict(zip([-1, 0, 1], y_counts))}")
        else:
            logger.error(f"  ❌ No training data available after cleaning!")
            raise ValueError("Insufficient training data after feature engineering")

        # Store features for prediction
        self.df_features = df_with_features
        logger.debug(f"  Stored features DataFrame: shape={self.df_features.shape}")

        # Create indicators for plotting
        self.I(get_y, self.data.df, name='y_true')
        self.forecasts = self.I(lambda: np.repeat(np.nan, len(self.data)), name='forecast')
        logger.debug(f"  Created plotting indicators")

        logger.success(f"🤖 MLWalkForwardStrategy.init() COMPLETE")

    # Enhanced next() tracing
    original_strategy_next = MLWalkForwardStrategy.next

    def ultra_logged_ml_next(self):
        current_bar = len(self.data)

        if current_bar <= 60 or current_bar % 25 == 0:  # Log first 60 bars and every 25th
            logger.trace(f"📊 Strategy.next() bar {current_bar}")

        if current_bar < self.n_train:
            logger.trace(f"  Skipping training period: {current_bar} < {self.n_train}")
            return

        # Log retraining events
        if current_bar % self.retrain_frequency == 0:
            logger.info(f"🔄 Retraining model at bar {current_bar}")

        return original_strategy_next(self)

    # Apply all patches
    import strategies.ml_strategy
    strategies.ml_strategy.create_features = ultra_logged_create_features
    strategies.ml_strategy.get_clean_Xy = ultra_logged_get_clean_Xy
    strategies.ml_strategy.get_y = ultra_logged_get_y

    MLWalkForwardStrategy.init = ultra_logged_ml_init
    MLWalkForwardStrategy.next = ultra_logged_ml_next
    Backtest.run = ultra_logged_backtest_run

    # Run enhanced backtest
    logger.info("🚀 Starting enhanced logged backtest")

    bt = Backtest(
        data_subset,
        MLWalkForwardStrategy,
        commission=0.0002,
        margin=0.05,
        cash=10000
    )

    logger.info("📈 Running enhanced backtest with comprehensive logging...")
    stats = bt.run(n_train=200, retrain_frequency=20)

    logger.success("✅ Enhanced backtest completed successfully")
    logger.info(f"📊 Final results: Return={stats['Return [%]']:.2f}%, Sharpe={stats['Sharpe Ratio']:.2f}, Trades={stats['# Trades']}")

    logger.info(f"📝 Complete debug log saved to: {log_file.absolute()}")

except Exception as e:
    logger.error(f"❌ Error during enhanced pipeline analysis: {e}")
    logger.exception("Full traceback:")

logger.info("🏁 Enhanced data pipeline analysis complete")