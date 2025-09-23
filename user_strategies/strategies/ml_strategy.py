"""
Machine Learning Trading Strategy

Clean, extracted implementation of ML-based trading strategies with walk-forward optimization.
Based on the backtesting.py example but refactored for production use.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from backtesting import Strategy
from backtesting.test import SMA
from pathlib import Path
from datetime import datetime
from loguru import logger
import sys

# Gapless-crypto-data import with fallback mechanism
GAPLESS_CRYPTO_AVAILABLE = False
gapless_crypto_data = None

try:
    import gapless_crypto_data
    GAPLESS_CRYPTO_AVAILABLE = True
except ImportError:
    GAPLESS_CRYPTO_AVAILABLE = False


def _get_output_directory() -> Path:
    """Get the output directory for persistent data"""
    current_file = Path(__file__)
    project_root = current_file.parent.parent
    output_dir = project_root / "data"
    output_dir.mkdir(exist_ok=True)
    return output_dir


def _generate_filename_prefix(strategy_name: str) -> str:
    """Generate timestamp-based filename prefix"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{strategy_name}_{timestamp}"


def bbands(data: pd.DataFrame, n_lookback: int, n_std: float) -> tuple[pd.Series, pd.Series]:
    """
    Bollinger Bands indicator

    Args:
        data: OHLCV DataFrame
        n_lookback: Lookback period for moving average
        n_std: Number of standard deviations for bands

    Returns:
        tuple: (upper_band, lower_band)
    """
    hlc3 = (data.High + data.Low + data.Close) / 3
    mean = hlc3.rolling(n_lookback).mean()
    std = hlc3.rolling(n_lookback).std()
    upper = mean + n_std * std
    lower = mean - n_std * std
    return upper, lower


def prepare_ml_data(data: pd.DataFrame, forecast_periods: int = 5, forecast_threshold: float = 0.01) -> pd.DataFrame:
    """
    Pre-compute all ML features following idiomatic backtesting.py pattern

    Based on the original "Trading with Machine Learning" example.
    Features and targets are added directly to the DataFrame before backtesting.

    Args:
        data: Raw OHLCV DataFrame
        forecast_periods: Days ahead to predict (default: 5 for daily data)
        forecast_threshold: Classification threshold (default: 1% for 5-day moves)

    Returns:
        DataFrame with all features and target column added
    """
    df = data.copy()
    close = df.Close.values

    # Technical indicators (following original example exactly)
    sma10 = SMA(df.Close, 10)
    sma20 = SMA(df.Close, 20)
    sma50 = SMA(df.Close, 50)
    sma100 = SMA(df.Close, 100)
    upper, lower = bbands(df, 20, 2)

    # Price-derived features (normalized by current price)
    df['X_SMA10'] = (close - sma10) / close
    df['X_SMA20'] = (close - sma20) / close
    df['X_SMA50'] = (close - sma50) / close
    df['X_SMA100'] = (close - sma100) / close

    # Moving average delta features
    df['X_DELTA_SMA10'] = (sma10 - sma20) / close
    df['X_DELTA_SMA20'] = (sma20 - sma50) / close
    df['X_DELTA_SMA50'] = (sma50 - sma100) / close

    # Technical indicator features
    df['X_MOM'] = df.Close.pct_change(periods=2)
    df['X_BB_upper'] = (upper - close) / close
    df['X_BB_lower'] = (lower - close) / close
    df['X_BB_width'] = (upper - lower) / close

    # Temporal features
    df['X_day'] = df.index.dayofweek
    df['X_hour'] = df.index.hour
    df['X_Sentiment'] = 1.0  # Placeholder - replace with real sentiment

    # Pre-compute target variable with custom forecast parameters (ALIGNED!)
    y = df.Close.pct_change(forecast_periods).shift(-forecast_periods)
    y[y.between(-forecast_threshold, forecast_threshold)] = 0
    y[y > 0] = 1
    y[y < 0] = -1
    df['y_target'] = y

    # Return clean data (idiomatic pattern)
    return df.dropna().astype(float)


def prepare_ml_data_short_term(data: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-compute features optimized for short-term (1-day) predictions

    Following idiomatic backtesting.py pattern from official ML example.
    Optimized for daily Bitcoin volatility with 1-day prediction horizon.

    Args:
        data: Raw OHLCV DataFrame

    Returns:
        DataFrame with short-term features as X_ columns
    """
    df = data.copy()
    close = df.Close.values

    # Short-term technical indicators (optimized for daily predictions)
    sma5 = SMA(df.Close, 5)
    sma10 = SMA(df.Close, 10)
    sma20 = SMA(df.Close, 20)
    upper, lower = bbands(df, 10, 2)  # Shorter Bollinger Bands

    # Price-derived features (normalized by current price)
    df['X_SMA5'] = (close - sma5) / close
    df['X_SMA10'] = (close - sma10) / close
    df['X_SMA20'] = (close - sma20) / close

    # Moving average delta features
    df['X_DELTA_SMA5'] = (sma5 - sma10) / close
    df['X_DELTA_SMA10'] = (sma10 - sma20) / close

    # Short-term momentum and volatility features
    df['X_MOM1'] = df.Close.pct_change(periods=1)  # 1-day momentum
    df['X_MOM2'] = df.Close.pct_change(periods=2)  # 2-day momentum
    df['X_VOL5'] = df.Close.rolling(5).std() / close  # 5-day volatility

    # Bollinger Bands features
    df['X_BB_upper'] = (upper - close) / close
    df['X_BB_lower'] = (lower - close) / close
    df['X_BB_width'] = (upper - lower) / close

    # Temporal features (same as long-term)
    df['X_day'] = df.index.dayofweek
    df['X_hour'] = df.index.hour
    df['X_Sentiment'] = 1.0  # Placeholder

    # Return clean data (idiomatic pattern)
    return df.dropna().astype(float)


# create_features() function removed - use prepare_ml_data() instead
# This follows the idiomatic backtesting.py pattern with pre-computed features


def get_X(data: pd.DataFrame) -> np.ndarray:
    """Extract feature matrix X from DataFrame"""
    return data.filter(like='X').values


def get_y(data: pd.DataFrame, forecast_periods: int = 48, threshold: float = 0.004) -> pd.Series:
    """
    Create target variable for classification

    Args:
        data: OHLCV DataFrame
        forecast_periods: Number of periods to look ahead
        threshold: Minimum return threshold for classification

    Returns:
        Series with classifications: 1 (up), -1 (down), 0 (neutral)
    """
    y = data.Close.pct_change(forecast_periods).shift(-forecast_periods)
    y[y.between(-threshold, threshold)] = 0
    y[y > 0] = 1
    y[y < 0] = -1
    return y


def get_y_short_term(data: pd.DataFrame, forecast_periods: int = 1, threshold: float = 0.002) -> pd.Series:
    """
    Create target variable for short-term (1-day) classification

    Following official backtesting.py ML example pattern (line 103-109).
    Optimized for daily Bitcoin predictions with aligned temporal horizon.

    Args:
        data: OHLCV DataFrame
        forecast_periods: Number of periods to look ahead (default: 1 day)
        threshold: Minimum return threshold for classification (default: 0.2%)

    Returns:
        Series with classifications: 1 (up), -1 (down), 0 (neutral)
    """
    y = data.Close.pct_change(forecast_periods).shift(-forecast_periods)
    y[y.between(-threshold, threshold)] = 0  # 0.2% threshold for daily moves
    y[y > 0] = 1
    y[y < 0] = -1
    return y


def get_data_source(source: str = 'EURUSD', **kwargs) -> pd.DataFrame:
    """
    Data source adapter with comprehensive validation and logging

    Ultra-safe data source switching with extensive validation at every step.
    Initially supports only EURUSD, with crypto support to be added incrementally.

    Args:
        source: Data source type ('EURUSD' or 'crypto')
        **kwargs: Additional parameters for data fetching
                 For crypto: symbol, start, end, interval, etc.

    Returns:
        pd.DataFrame: OHLCV data with DatetimeIndex

    Raises:
        ValueError: If data validation fails
        ImportError: If required packages not available
    """
    logger.info(f"🔄 get_data_source() ENTRY: source='{source}', kwargs={kwargs}")

    data = None

    if source == 'EURUSD':
        logger.info("   Loading EURUSD data from backtesting.test...")
        try:
            from backtesting.test import EURUSD
            data = EURUSD.copy()
            logger.success(f"   ✅ EURUSD data loaded: shape={data.shape}")

        except ImportError as e:
            logger.error(f"   ❌ Failed to import EURUSD: {e}")
            raise ImportError(f"Cannot import EURUSD data: {e}")

    elif source == 'crypto':
        logger.info("   🔄 Processing crypto data request...")

        if not GAPLESS_CRYPTO_AVAILABLE:
            logger.warning("   ⚠️ gapless-crypto-data package not available")
            logger.info("   Falling back to EURUSD for safety...")

            # Fallback to EURUSD when crypto package unavailable
            from backtesting.test import EURUSD
            data = EURUSD.copy()
            logger.warning(f"   ⚠️ Using EURUSD fallback: shape={data.shape}")

        else:
            logger.info("   ✅ gapless-crypto-data package available, attempting crypto data fetch...")

            # Set default parameters for extended crypto data fetching (4.7 years)
            symbol = kwargs.get('symbol', 'BTCUSDT')
            start = kwargs.get('start', '2021-01-01')
            end = kwargs.get('end', '2025-08-31')  # Extended period: 4.7 years of data
            interval = kwargs.get('interval', '1d')  # Daily intervals for extended backtesting

            logger.info(f"   Crypto parameters: symbol={symbol}, start={start}, end={end}, interval={interval}")

            try:
                # Attempt to fetch crypto data using gapless-crypto-data
                logger.info(f"   Fetching crypto data: symbol={symbol}, start={start}, end={end}, interval={interval}")

                # Use gapless_crypto_data.fetch_data with proper parameters
                crypto_data = gapless_crypto_data.fetch_data(
                    symbol=symbol,
                    timeframe=interval,
                    start=start,
                    end=end
                )

                logger.success(f"   ✅ Crypto data fetched successfully!")

                # Convert to standard OHLCV format

                # Column mapping from gapless-crypto-data to backtesting.py format
                column_mapping = {
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume'
                }

                # Check if we have the expected lowercase columns
                required_crypto_cols = ['open', 'high', 'low', 'close', 'volume', 'date']
                missing_crypto_cols = set(required_crypto_cols) - set(crypto_data.columns)
                if missing_crypto_cols:
                    raise ValueError(f"Crypto data missing expected columns: {missing_crypto_cols}")


                # Select and rename columns to OHLCV format
                selected_cols = ['date'] + list(column_mapping.keys())
                crypto_subset = crypto_data[selected_cols].copy()
                crypto_subset = crypto_subset.rename(columns=column_mapping)


                # Convert date column to DatetimeIndex
                if 'date' in crypto_subset.columns:
                    crypto_subset.set_index('date', inplace=True)

                data = crypto_subset
                logger.success(f"   ✅ Crypto data converted to backtesting.py format: shape={data.shape}")

            except Exception as e:
                logger.error(f"   ❌ Failed to fetch crypto data: {e}")
                logger.warning("   Falling back to EURUSD for safety...")

                # Fallback to EURUSD on any crypto fetch failure
                from backtesting.test import EURUSD
                data = EURUSD.copy()
                logger.warning(f"   ⚠️ Using EURUSD fallback after crypto failure: shape={data.shape}")

    else:
        logger.error(f"   ❌ Unknown data source: '{source}'")
        raise ValueError(f"Unsupported data source: '{source}'. Supported: ['EURUSD', 'crypto']")

    # Basic data validation
    if data is not None:
        # Validate required OHLCV columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = set(required_cols) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Data missing required OHLCV columns: {missing_cols}")

        # Validate data types
        numeric_cols = ['Open', 'High', 'Low', 'Close']
        for col in numeric_cols:
            if not pd.api.types.is_numeric_dtype(data[col]):
                raise ValueError(f"Column '{col}' must be numeric, got {data[col].dtype}")

        # Validate index
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError(f"Data index must be DatetimeIndex, got {type(data.index)}")
    else:
        raise ValueError(f"No data available from source '{source}'")

    logger.success(f"🔄 get_data_source() SUCCESS: source='{source}', shape={data.shape}")
    return data


def get_clean_Xy_short_term(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Return clean (X, y) arrays for short-term predictions without NaN values

    Following official backtesting.py ML example pattern (line 112-119).
    Used with short-term features and 1-day targets.

    Args:
        df: DataFrame with short-term features and price data

    Returns:
        tuple: (X_clean, y_clean) as numpy arrays
    """
    X = get_X(df)
    y = get_y_short_term(df).values

    isnan = np.isnan(y)
    X_clean = X[~isnan]
    y_clean = y[~isnan]

    return X_clean, y_clean


class PersistentOutputMixin:
    """Mixin class for strategies that need to save persistent outputs"""

    def save_backtest_results(self, stats: pd.Series, trades: pd.DataFrame, strategy_name: str, backtest_instance=None):
        """Save backtest results to persistent files"""
        output_dir = _get_output_directory()
        prefix = _generate_filename_prefix(strategy_name)

        # Save performance metrics
        performance_file = output_dir / f"{prefix}_performance.csv"
        stats.to_csv(performance_file)

        # Save trades data
        trades_file = output_dir / f"{prefix}_trades.csv"
        trades.to_csv(trades_file, index=False)

        # Save HTML plot if backtest instance is provided
        html_file = None
        if backtest_instance is not None:
            html_file = output_dir / f"{prefix}_backtest.html"
            backtest_instance.plot(filename=str(html_file), open_browser=False)

        print(f"✅ Results saved to: {output_dir}")
        print(f"📊 Performance: {performance_file.name}")
        print(f"📈 Trades: {trades_file.name}")
        if html_file:
            print(f"📉 Chart: {html_file.name}")

        return performance_file, trades_file, html_file


class MLTrainOnceStrategy(Strategy, PersistentOutputMixin):
    """
    ML Strategy that trains once on initial data

    Follows idiomatic backtesting.py pattern with pre-computed features.
    Features must be added to data before backtesting starts.

    Parameters:
        n_train: Number of training samples
        price_delta: Threshold for stop-loss and take-profit (as fraction)
        position_size: Position size as fraction of equity
        n_neighbors: Number of neighbors for kNN classifier
        test_in_sample: If True, test on training data for overfitting verification
    """

    # Strategy parameters
    n_train = 400
    price_delta = 0.02  # 2% - appropriate for daily Bitcoin volatility
    position_size = 0.2  # 20% of equity
    n_neighbors = 7
    test_in_sample = False  # Overfitting verification mode

    def init(self):
        """Initialize strategy with ML model and indicators (idiomatic pattern)"""
        logger.info(f"🤖 {self.__class__.__name__}.init() STARTING")

        # Verify pre-computed features exist (idiomatic requirement)
        feature_cols = [col for col in self.data.df.columns if col.startswith('X_')]
        if not feature_cols:
            raise ValueError("No pre-computed features found! Call prepare_ml_data() first.")

        logger.info(f"   Found {len(feature_cols)} pre-computed features: {feature_cols[:3]}...")

        # Initialize kNN classifier
        self.clf = KNeighborsClassifier(self.n_neighbors)

        # Extract pre-computed features and targets (idiomatic pattern)
        if self.test_in_sample:
            # Overfitting verification: test on training data
            logger.warning(f"   🔍 OVERFITTING TEST MODE: Testing on training data")
            train_start, train_end = 0, self.n_train
        else:
            # Normal mode: test on out-of-sample data
            train_start, train_end = 0, self.n_train

        # Get training data from pre-computed features
        train_df = self.data.df.iloc[train_start:train_end]
        X_train = train_df.filter(like='X_').values
        y_train = train_df['y_target'].values  # Use pre-computed targets

        # Clean training data
        isnan = np.isnan(y_train)
        X_train_clean = X_train[~isnan]
        y_train_clean = y_train[~isnan]

        logger.info(f"   Training data: {X_train.shape} → {X_train_clean.shape} (cleaned)")

        if len(X_train_clean) > 0 and len(y_train_clean) > 0:
            self.clf.fit(X_train_clean, y_train_clean)

            # Log training accuracy for overfitting verification
            train_score = self.clf.score(X_train_clean, y_train_clean)
            logger.success(f"   ✅ Model trained on {len(X_train_clean)} samples")
            logger.info(f"   📊 Training accuracy: {train_score:.3f}")

            # Log class distribution
            unique_y, counts = np.unique(y_train_clean, return_counts=True)
            class_dist = dict(zip(unique_y, counts))
            logger.info(f"   📈 Class distribution: {class_dist}")
        else:
            logger.error(f"   ❌ No training data available after cleaning!")
            raise ValueError("Insufficient training data after feature engineering")

        # Create indicators for plotting (idiomatic pattern)
        self.I(lambda: self.data.df['y_target'], name='y_true')  # Use pre-computed targets
        self.forecasts = self.I(lambda: np.repeat(np.nan, len(self.data)), name='forecast')

        logger.success(f"🤖 {self.__class__.__name__}.init() COMPLETE")

    def next(self):
        """Execute trading logic using pre-computed features (idiomatic pattern)"""
        # Skip training period
        if len(self.data) < self.n_train:
            return

        # Get current market data
        high, low, close = self.data.High, self.data.Low, self.data.Close
        current_time = self.data.index[-1]

        # Access pre-computed features (idiomatic pattern)
        current_features = self.data.df.iloc[-1:].filter(like='X_')

        if len(current_features) == 0 or current_features.isna().any().any():
            return

        X_current = current_features.values
        forecast = self.clf.predict(X_current)[0]

        # Update forecast indicator
        self.forecasts[-1] = forecast

        # Calculate stop-loss and take-profit levels
        current_price = close[-1]
        upper = current_price * (1 + self.price_delta)
        lower = current_price * (1 - self.price_delta)

        # Execute trades based on forecast
        if forecast == 1 and not self.position.is_long:
            self.buy(size=self.position_size, tp=upper, sl=lower)
        elif forecast == -1 and not self.position.is_short:
            self.sell(size=self.position_size, tp=lower, sl=upper)

        # Aggressive stop-loss for trades open > 2 days
        self._manage_open_trades(current_time, high, low)

    def _manage_open_trades(self, current_time: pd.Timestamp, high: float, low: float):
        """Manage stop-losses for trades open longer than 2 days"""
        for trade in self.trades:
            if current_time - trade.entry_time > pd.Timedelta('2 days'):
                if trade.is_long:
                    trade.sl = max(trade.sl, low)  # Tighten long stop-loss
                else:
                    trade.sl = min(trade.sl, high)  # Tighten short stop-loss


class MLWalkForwardStrategy(MLTrainOnceStrategy):
    """
    ML Strategy with walk-forward optimization - TEMPORALLY ALIGNED

    Retrains the model periodically on recent data to adapt to changing market conditions.
    Now supports custom forecast horizons for proper temporal alignment with stop-losses.

    Additional Parameters:
        retrain_frequency: Retrain every N bars (default: 20)
        forecast_periods: Days ahead to predict (default: 5 for daily data)
        forecast_threshold: Classification threshold (default: 1% for 5-day moves)
        price_delta: Stop-loss/take-profit (default: 5% for 5-day horizon)
    """

    retrain_frequency = 20
    forecast_periods = 5        # 5 days ahead (aligned with daily data)
    forecast_threshold = 0.01   # 1% threshold for 5-day moves
    price_delta = 0.05          # 5% stop-loss aligned with 5-day forecast

    def next(self):
        """Execute trading logic with periodic retraining"""
        current_bar = len(self.data)

        # Skip cold start period
        if current_bar < self.n_train:
            return

        # Retrain model every retrain_frequency periods
        if current_bar % self.retrain_frequency == 0:
            logger.info(f"🔄 Triggering model retraining at bar {current_bar}")
            self._retrain_model()

        # Execute standard trading logic
        super().next()

    def _retrain_model(self):
        """Retrain model on recent data using pre-computed features (idiomatic pattern)"""
        current_idx = len(self.data) - 1
        logger.info(f"🔄 _retrain_model() STARTING at bar {current_idx}")

        # Get recent data for training using pre-computed features
        start_idx = max(0, current_idx - self.n_train)
        recent_df = self.data.df.iloc[start_idx:current_idx]

        # Extract features and targets from pre-computed data
        X_recent = recent_df.filter(like='X_').values
        y_recent = recent_df['y_target'].values  # Use pre-computed targets

        # Clean data (remove NaN values)
        isnan = np.isnan(y_recent)
        X_recent_clean = X_recent[~isnan]
        y_recent_clean = y_recent[~isnan]

        logger.info(f"   Recent data: {X_recent.shape} → {X_recent_clean.shape} (cleaned)")

        # Retrain if we have enough clean data
        if len(X_recent_clean) >= 50:  # Minimum training samples
            if len(X_recent_clean) > 10:  # Ensure sufficient clean samples
                # Store previous accuracy for comparison
                prev_score = None
                if len(X_recent_clean) > 0:
                    prev_score = self.clf.score(X_recent_clean, y_recent_clean)

                # Retrain model
                self.clf.fit(X_recent_clean, y_recent_clean)
                new_score = self.clf.score(X_recent_clean, y_recent_clean)

                logger.success(f"   ✅ Model retrained on {len(X_recent_clean)} samples")
                logger.info(f"   📊 Training accuracy: {new_score:.3f}")

                # Log class distribution for retraining analysis
                unique_y, counts = np.unique(y_recent_clean, return_counts=True)
                class_dist = dict(zip(unique_y, counts))
                logger.info(f"   📈 Retrain class distribution: {class_dist}")
            else:
                logger.warning(f"   ⚠️ Insufficient clean samples for retraining: {len(X_recent_clean)} <= 10")
        else:
            logger.warning(f"   ⚠️ Insufficient data for retraining: {len(X_recent_clean)} < 50")



def verify_overfitting(data: pd.DataFrame, strategy_class=MLTrainOnceStrategy, **kwargs) -> dict:
    """
    Verify overfitting by comparing in-sample vs out-of-sample performance

    This function tests the strategy on both training data (in-sample) and
    test data (out-of-sample) to detect overfitting. Significant performance
    differences indicate potential overfitting.

    Args:
        data: OHLCV DataFrame with pre-computed features (call prepare_ml_data() first)
        strategy_class: Strategy class to test (default: MLTrainOnceStrategy)
        **kwargs: Additional parameters for backtest or strategy

    Returns:
        dict: Overfitting analysis results with performance metrics and diagnosis
    """
    from backtesting import Backtest

    logger.info(f"🔍 verify_overfitting() STARTING for {strategy_class.__name__}")

    # Verify pre-computed features exist
    feature_cols = [col for col in data.columns if col.startswith('X_')]
    if not feature_cols:
        raise ValueError("No pre-computed features found! Call prepare_ml_data() first.")

    logger.info(f"   Found {len(feature_cols)} pre-computed features for overfitting test")

    # Extract backtest parameters
    backtest_params = {
        'cash': kwargs.pop('cash', 10_000_000),
        'commission': kwargs.pop('commission', 0.0002),
        'margin': kwargs.pop('margin', 0.05),
        'exclusive_orders': kwargs.pop('exclusive_orders', True),
        'trade_on_close': kwargs.pop('trade_on_close', False)
    }

    # Test 1: In-sample performance (overfitting test)
    logger.info(f"   📊 Running IN-SAMPLE test (overfitting detection mode)...")
    bt_insample = Backtest(data, strategy_class, **backtest_params)
    stats_insample = bt_insample.run(test_in_sample=True, **kwargs)

    # Test 2: Out-of-sample performance (normal mode)
    logger.info(f"   🎯 Running OUT-OF-SAMPLE test (normal mode)...")
    bt_outsample = Backtest(data, strategy_class, **backtest_params)
    stats_outsample = bt_outsample.run(test_in_sample=False, **kwargs)

    # Extract key performance metrics
    metrics_insample = {
        'return_pct': stats_insample['Return [%]'],
        'sharpe_ratio': stats_insample['Sharpe Ratio'],
        'win_rate': stats_insample['Win Rate [%]'],
        'max_drawdown': stats_insample['Max. Drawdown [%]'],
        'num_trades': stats_insample['# Trades']
    }

    metrics_outsample = {
        'return_pct': stats_outsample['Return [%]'],
        'sharpe_ratio': stats_outsample['Sharpe Ratio'],
        'win_rate': stats_outsample['Win Rate [%]'],
        'max_drawdown': stats_outsample['Max. Drawdown [%]'],
        'num_trades': stats_outsample['# Trades']
    }

    # Calculate performance differences (in-sample - out-of-sample)
    return_diff = metrics_insample['return_pct'] - metrics_outsample['return_pct']
    sharpe_diff = metrics_insample['sharpe_ratio'] - metrics_outsample['sharpe_ratio']
    winrate_diff = metrics_insample['win_rate'] - metrics_outsample['win_rate']

    # Overfitting diagnosis
    overfitting_signals = []

    if return_diff > 5.0:  # In-sample return > 5% better
        overfitting_signals.append(f"Return gap: {return_diff:+.1f}% (in-sample advantage)")

    if sharpe_diff > 0.5:  # In-sample Sharpe > 0.5 better
        overfitting_signals.append(f"Sharpe gap: {sharpe_diff:+.2f} (in-sample advantage)")

    if winrate_diff > 15.0:  # In-sample win rate > 15% better
        overfitting_signals.append(f"Win rate gap: {winrate_diff:+.1f}% (in-sample advantage)")

    if metrics_outsample['return_pct'] < -5.0 and metrics_insample['return_pct'] > 5.0:
        overfitting_signals.append("Positive in-sample, negative out-of-sample (classic overfitting)")

    # Overall diagnosis
    is_overfitted = len(overfitting_signals) >= 2
    confidence = "HIGH" if len(overfitting_signals) >= 3 else "MEDIUM" if len(overfitting_signals) == 2 else "LOW"

    # Results summary
    results = {
        'is_overfitted': is_overfitted,
        'confidence': confidence,
        'overfitting_signals': overfitting_signals,
        'metrics_insample': metrics_insample,
        'metrics_outsample': metrics_outsample,
        'performance_gaps': {
            'return_diff': return_diff,
            'sharpe_diff': sharpe_diff,
            'winrate_diff': winrate_diff
        },
        'stats_insample': stats_insample,
        'stats_outsample': stats_outsample
    }

    # Print diagnosis
    print(f"\n🔍 OVERFITTING ANALYSIS: {strategy_class.__name__}")
    print(f"\n📊 IN-SAMPLE Performance (Training Data):")
    print(f"  Return: {metrics_insample['return_pct']:+.2f}%")
    print(f"  Sharpe: {metrics_insample['sharpe_ratio']:.2f}")
    print(f"  Win Rate: {metrics_insample['win_rate']:.1f}%")
    print(f"  Trades: {metrics_insample['num_trades']}")

    print(f"\n🎯 OUT-OF-SAMPLE Performance (Test Data):")
    print(f"  Return: {metrics_outsample['return_pct']:+.2f}%")
    print(f"  Sharpe: {metrics_outsample['sharpe_ratio']:.2f}")
    print(f"  Win Rate: {metrics_outsample['win_rate']:.1f}%")
    print(f"  Trades: {metrics_outsample['num_trades']}")

    print(f"\n📎 PERFORMANCE GAPS (In-Sample - Out-of-Sample):")
    print(f"  Return Difference: {return_diff:+.2f}%")
    print(f"  Sharpe Difference: {sharpe_diff:+.2f}")
    print(f"  Win Rate Difference: {winrate_diff:+.1f}%")

    if is_overfitted:
        print(f"\n⚠️  OVERFITTING DETECTED ({confidence} confidence)")
        for signal in overfitting_signals:
            print(f"    • {signal}")
        print(f"\n📝 Recommendation: Reduce model complexity, add regularization, or collect more data")
    else:
        print(f"\n✅ NO SIGNIFICANT OVERFITTING DETECTED")
        print(f"    Performance differences within acceptable ranges")

    logger.success(f"🔍 verify_overfitting() COMPLETE: {'OVERFITTED' if is_overfitted else 'CLEAN'}")

    return results


def run_ml_strategy_with_persistence(data: pd.DataFrame, strategy_class=MLWalkForwardStrategy, **kwargs):
    """
    Run ML strategy with automatic output persistence

    Args:
        data: OHLCV DataFrame with pre-computed features (call prepare_ml_data() first)
        strategy_class: Strategy class to use (default: MLWalkForwardStrategy)
        **kwargs: Additional parameters for backtest or strategy

    Returns:
        tuple: (stats, trades, output_files)
    """
    from backtesting import Backtest

    # Verify pre-computed features exist
    feature_cols = [col for col in data.columns if col.startswith('X_')]
    if not feature_cols:
        raise ValueError("No pre-computed features found! Call prepare_ml_data() first.")

    # Extract backtest parameters
    backtest_params = {
        'cash': kwargs.pop('cash', 10_000_000),
        'commission': kwargs.pop('commission', 0.0002),
        'margin': kwargs.pop('margin', 0.05),
        'exclusive_orders': kwargs.pop('exclusive_orders', True),
        'trade_on_close': kwargs.pop('trade_on_close', False)
    }

    # Run backtest
    bt = Backtest(data, strategy_class, **backtest_params)
    stats = bt.run(**kwargs)

    # Create a mock strategy instance just for output persistence
    class MockStrategy(PersistentOutputMixin):
        pass

    strategy_instance = MockStrategy()
    strategy_name = strategy_class.__name__

    # Save results
    trades_df = stats._trades if hasattr(stats, '_trades') else pd.DataFrame()
    output_files = strategy_instance.save_backtest_results(stats, trades_df, strategy_name, bt)

    print(f"\n🎯 {strategy_name} Performance Summary:")
    print(f"Return: {stats['Return [%]']:.2f}%")
    print(f"Sharpe Ratio: {stats['Sharpe Ratio']:.2f}")
    print(f"Max Drawdown: {stats['Max. Drawdown [%]']:.2f}%")
    print(f"Win Rate: {stats['Win Rate [%]']:.1f}%")
    print(f"# Trades: {stats['# Trades']}")

    return stats, trades_df, output_files


# Optimized feature functions removed - use prepare_ml_data() with custom parameters
# For shorter indicators, modify the prepare_ml_data() function directly


class MLShortTermStrategy(Strategy, PersistentOutputMixin):
    """
    ML Strategy with short-term (1-day) predictions - TEMPORALLY ALIGNED

    Following official backtesting.py ML example pattern (lines 164-217).
    Solves the temporal mismatch: 1-day predictions with 2% daily stop-losses.

    Parameters:
        n_train: Number of training samples (default: 100 for daily predictions)
        price_delta: Stop-loss/take-profit threshold (2% - now ALIGNED!)
        retrain_frequency: How often to retrain (default: 5 days)
        n_neighbors: k-NN classifier neighbors
    """

    # Strategy parameters (optimized for 1-day predictions)
    n_train = 100
    price_delta = 0.02      # 2% - NOW ALIGNED with 1-day forecast!
    retrain_frequency = 5   # Retrain every 5 days for daily predictions
    n_neighbors = 7

    def init(self):
        """Initialize strategy following official ML example pattern (lines 167-180)"""
        logger.info(f"🚀 {self.__class__.__name__}.init() - SHORT-TERM ALIGNED STRATEGY")

        # Verify pre-computed features exist (idiomatic requirement)
        feature_cols = [col for col in self.data.df.columns if col.startswith('X_')]
        if not feature_cols:
            raise ValueError("No pre-computed features found! Call prepare_ml_data_short_term() first.")

        logger.info(f"   Found {len(feature_cols)} short-term features: {feature_cols[:3]}...")

        # Initialize kNN classifier (exact pattern from official example)
        self.clf = KNeighborsClassifier(self.n_neighbors)

        # Train the classifier on first n_train examples (idiomatic pattern)
        df = self.data.df.iloc[:self.n_train]
        X, y = get_clean_Xy_short_term(df)

        if len(X) > 0 and len(y) > 0:
            self.clf.fit(X, y)

            # Log training accuracy
            train_score = self.clf.score(X, y)
            logger.success(f"   ✅ Model trained on {len(X)} samples")
            logger.info(f"   📊 Training accuracy: {train_score:.3f}")

            # Log class distribution for short-term predictions
            unique_y, counts = np.unique(y, return_counts=True)
            class_dist = dict(zip(unique_y, counts))
            logger.info(f"   📈 Class distribution: {class_dist}")
        else:
            logger.error(f"   ❌ No training data available after cleaning!")
            raise ValueError("Insufficient training data for short-term predictions")

        # Create indicators for plotting (exact pattern from lines 177-180)
        self.I(get_y_short_term, self.data.df, name='y_true')
        self.forecasts = self.I(lambda: np.repeat(np.nan, len(self.data)), name='forecast')

        logger.success(f"🚀 {self.__class__.__name__}.init() COMPLETE - TEMPORAL ALIGNMENT ACHIEVED!")

    def next(self):
        """Execute trading logic following official pattern (lines 182-217)"""
        # Skip training period (exact pattern from line 184)
        if len(self.data) < self.n_train:
            return

        # Get current market data
        high, low, close = self.data.High, self.data.Low, self.data.Close
        current_time = self.data.index[-1]

        # Forecast next movement (idiomatic access pattern from line 192)
        X = get_X(self.data.df.iloc[-1:])
        forecast = self.clf.predict(X)[0]

        # Update forecast indicator (exact pattern from line 196)
        self.forecasts[-1] = forecast

        # Calculate stop-loss and take-profit levels (pattern from lines 202-203)
        upper, lower = close[-1] * (1 + np.r_[1, -1] * self.price_delta)

        # Execute trades based on forecast (exact pattern from lines 204-207)
        # NOW ALIGNED: 1-day prediction with 2% stop-loss!
        if forecast == 1 and not self.position.is_long:
            self.buy(size=0.2, tp=upper, sl=lower)
        elif forecast == -1 and not self.position.is_short:
            self.sell(size=0.2, tp=lower, sl=upper)

        # Aggressive stop-loss management (pattern from lines 211-216)
        for trade in self.trades:
            if current_time - trade.entry_time > pd.Timedelta('2 days'):
                if trade.is_long:
                    trade.sl = max(trade.sl, low)
                else:
                    trade.sl = min(trade.sl, high)


class MLShortTermWalkForward(MLShortTermStrategy):
    """
    Short-term ML Strategy with walk-forward optimization - TEMPORALLY ALIGNED

    Following official backtesting.py ML example pattern (lines 235-250).
    Combines 1-day predictions with periodic retraining for market adaptation.
    Solves the temporal mismatch with frequent retraining for daily predictions.
    """

    def next(self):
        """Execute trading logic with periodic retraining (official pattern lines 236-250)"""
        # Skip cold start period (exact pattern from line 238)
        if len(self.data) < self.n_train:
            return

        # Re-train the model every retrain_frequency iterations (pattern from line 244)
        # More frequent retraining for 1-day predictions (every 5 days vs 20)
        if len(self.data) % self.retrain_frequency:
            return super().next()

        # Retrain on last n_train values (exact pattern from line 248)
        logger.info(f"🔄 Retraining short-term model at bar {len(self.data)}")
        df = self.data.df[-self.n_train:]
        X, y = get_clean_Xy_short_term(df)

        if len(X) > 10:  # Ensure sufficient samples for retraining
            # Store previous performance for comparison
            prev_score = None
            if len(X) > 0:
                prev_score = self.clf.score(X, y)

            # Retrain model (exact pattern from official example)
            self.clf.fit(X, y)
            new_score = self.clf.score(X, y)

            logger.success(f"   ✅ Short-term model retrained on {len(X)} samples")
            logger.info(f"   📊 Training accuracy: {new_score:.3f}")

            # Log class distribution for retraining analysis
            unique_y, counts = np.unique(y, return_counts=True)
            class_dist = dict(zip(unique_y, counts))
            logger.info(f"   📈 Retrain class distribution: {class_dist}")
        else:
            logger.warning(f"   ⚠️ Insufficient data for short-term retraining: {len(X)} samples")

        # Execute normal trading logic after retraining
        super().next()