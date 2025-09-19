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


def create_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create feature matrix for ML model

    Args:
        data: OHLCV DataFrame

    Returns:
        DataFrame with engineered features
    """
    df = data.copy()
    close = df.Close.values

    # Calculate technical indicators
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

    # Example sentiment feature (replace with real sentiment data)
    df['X_Sentiment'] = 1.0  # Placeholder - replace with actual sentiment

    return df.dropna().astype(float)


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


def get_clean_Xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Return clean (X, y) arrays without NaN values

    Args:
        df: DataFrame with features and price data

    Returns:
        tuple: (X_clean, y_clean) as numpy arrays
    """
    X = get_X(df)
    y = get_y(df).values
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

    Parameters:
        n_train: Number of training samples
        price_delta: Threshold for stop-loss and take-profit (as fraction)
        position_size: Position size as fraction of equity
        n_neighbors: Number of neighbors for kNN classifier
    """

    # Strategy parameters
    n_train = 400
    price_delta = 0.004  # 0.4%
    position_size = 0.2  # 20% of equity
    n_neighbors = 7

    def init(self):
        """Initialize strategy with ML model and indicators"""
        # Initialize kNN classifier
        self.clf = KNeighborsClassifier(self.n_neighbors)

        # Create features for entire dataset
        df_with_features = create_features(self.data.df)

        # Train on first n_train samples
        train_df = df_with_features.iloc[:self.n_train]
        X_train, y_train = get_clean_Xy(train_df)
        self.clf.fit(X_train, y_train)

        # Store features for prediction
        self.df_features = df_with_features

        # Create indicators for plotting
        self.I(get_y, self.data.df, name='y_true')
        self.forecasts = self.I(lambda: np.repeat(np.nan, len(self.data)), name='forecast')

    def next(self):
        """Execute trading logic for each time step"""
        # Skip training period
        if len(self.data) < self.n_train:
            return

        # Get current market data
        high, low, close = self.data.High, self.data.Low, self.data.Close
        current_time = self.data.index[-1]

        # Get features for current observation
        current_idx = len(self.data) - 1
        if current_idx >= len(self.df_features):
            return

        X_current = get_X(self.df_features.iloc[current_idx:current_idx+1])
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
    ML Strategy with walk-forward optimization

    Retrains the model periodically on recent data to adapt to changing market conditions.

    Additional Parameters:
        retrain_frequency: Retrain every N bars (default: 20)
    """

    retrain_frequency = 20

    def next(self):
        """Execute trading logic with periodic retraining"""
        # Skip cold start period
        if len(self.data) < self.n_train:
            return

        # Retrain model every retrain_frequency periods
        if len(self.data) % self.retrain_frequency == 0:
            self._retrain_model()

        # Execute standard trading logic
        super().next()

    def _retrain_model(self):
        """Retrain model on recent data"""
        current_idx = len(self.data) - 1

        # Get recent data for training
        start_idx = max(0, current_idx - self.n_train)
        recent_df = self.df_features.iloc[start_idx:current_idx]

        # Retrain if we have enough data
        if len(recent_df) >= 50:  # Minimum training samples
            X_recent, y_recent = get_clean_Xy(recent_df)
            if len(X_recent) > 10:  # Ensure sufficient clean samples
                self.clf.fit(X_recent, y_recent)


def run_ml_strategy_with_persistence(data: pd.DataFrame, strategy_class=MLWalkForwardStrategy, **kwargs):
    """
    Run ML strategy with automatic output persistence

    Args:
        data: OHLCV DataFrame
        strategy_class: Strategy class to use (default: MLWalkForwardStrategy)
        **kwargs: Additional parameters for backtest or strategy

    Returns:
        tuple: (stats, trades, output_files)
    """
    from backtesting import Backtest

    # Extract backtest parameters
    backtest_params = {
        'cash': kwargs.pop('cash', 10000),
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