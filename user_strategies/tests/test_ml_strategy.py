"""
Pytest-based tests for ML trading strategies

Verifies that ML strategies run without errors and produce valid results.
Follows Python testing best practices with proper fixtures and assertions.
"""

import pytest
import numpy as np
import pandas as pd
from backtesting import Backtest
from backtesting.test import EURUSD

# Import our user strategies (clean separation from original codebase)
from user_strategies.strategies.ml_strategy import (
    MLTrainOnceStrategy,
    MLWalkForwardStrategy,
    create_features,
    get_X,
    get_y,
    get_clean_Xy,
    bbands
)


@pytest.fixture
def sample_data():
    """Provide sample OHLCV data for testing"""
    # Use a subset of EURUSD data for faster testing
    return EURUSD.iloc[:500].copy()


@pytest.fixture
def sample_data_with_features(sample_data):
    """Provide sample data with engineered features"""
    return create_features(sample_data)


class TestFeatureEngineering:
    """Test feature engineering functions"""

    def test_bbands_calculation(self, sample_data):
        """Test Bollinger Bands calculation"""
        upper, lower = bbands(sample_data, n_lookback=20, n_std=2)

        # Verify output types and shapes
        assert isinstance(upper, pd.Series)
        assert isinstance(lower, pd.Series)
        assert len(upper) == len(sample_data)
        assert len(lower) == len(sample_data)

        # Verify upper > lower (after warm-up period)
        valid_idx = ~(upper.isna() | lower.isna())
        assert (upper[valid_idx] >= lower[valid_idx]).all()

    def test_create_features(self, sample_data):
        """Test feature matrix creation"""
        df_features = create_features(sample_data)

        # Verify output structure
        assert isinstance(df_features, pd.DataFrame)
        assert len(df_features) <= len(sample_data)  # Some rows dropped due to NaN

        # Verify feature columns exist
        feature_cols = [col for col in df_features.columns if col.startswith('X_')]
        assert len(feature_cols) >= 10  # Should have multiple features

        # Verify no NaN values in output
        assert not df_features.isna().any().any()

    def test_get_X_extraction(self, sample_data_with_features):
        """Test feature matrix extraction"""
        X = get_X(sample_data_with_features)

        # Verify output
        assert isinstance(X, np.ndarray)
        assert X.ndim == 2
        assert X.shape[0] == len(sample_data_with_features)
        assert X.shape[1] > 0  # Should have features

    def test_get_y_target_creation(self, sample_data):
        """Test target variable creation"""
        y = get_y(sample_data, forecast_periods=48, threshold=0.004)

        # Verify output
        assert isinstance(y, pd.Series)
        assert len(y) == len(sample_data)

        # Verify classification values
        unique_vals = y.dropna().unique()
        valid_vals = {-1, 0, 1}
        assert set(unique_vals).issubset(valid_vals)

    def test_get_clean_Xy(self, sample_data_with_features):
        """Test clean feature/target extraction"""
        X, y = get_clean_Xy(sample_data_with_features)

        # Verify outputs
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape[0] == y.shape[0]  # Same number of samples
        assert X.ndim == 2
        assert y.ndim == 1

        # Verify no NaN values
        assert not np.isnan(X).any()
        assert not np.isnan(y).any()


class TestMLStrategies:
    """Test ML trading strategy implementations"""

    def test_ml_train_once_strategy_initialization(self, sample_data):
        """Test MLTrainOnceStrategy initializes without errors"""
        # Create backtest instance
        bt = Backtest(
            sample_data,
            MLTrainOnceStrategy,
            commission=0.0002,
            margin=0.05,
            cash=10000
        )

        # Verify backtest created successfully
        assert bt is not None
        assert bt._strategy == MLTrainOnceStrategy

    def test_ml_train_once_strategy_execution(self, sample_data):
        """Test MLTrainOnceStrategy runs without errors"""
        bt = Backtest(
            sample_data,
            MLTrainOnceStrategy,
            commission=0.0002,
            margin=0.05,
            cash=10000
        )

        # Run backtest - should not raise exceptions
        stats = bt.run()

        # Verify stats returned
        assert stats is not None
        assert isinstance(stats, pd.Series)

        # Verify key statistics exist
        expected_stats = ['Return [%]', 'Sharpe Ratio', 'Max Drawdown [%]']
        for stat in expected_stats:
            assert stat in stats.index

    def test_ml_walk_forward_strategy_execution(self, sample_data):
        """Test MLWalkForwardStrategy runs without errors"""
        bt = Backtest(
            sample_data,
            MLWalkForwardStrategy,
            commission=0.0002,
            margin=0.05,
            cash=10000
        )

        # Run backtest - should not raise exceptions
        stats = bt.run()

        # Verify stats returned
        assert stats is not None
        assert isinstance(stats, pd.Series)

        # Verify strategy completed execution
        assert 'Return [%]' in stats.index
        assert not pd.isna(stats['Return [%]'])

    def test_strategy_parameters_customization(self, sample_data):
        """Test strategy parameter customization"""
        # Test with custom parameters
        custom_params = {
            'n_train': 200,
            'price_delta': 0.005,
            'position_size': 0.1,
            'n_neighbors': 5
        }

        bt = Backtest(
            sample_data,
            MLTrainOnceStrategy,
            commission=0.0002,
            margin=0.05,
            cash=10000
        )

        # Run with custom parameters
        stats = bt.run(**custom_params)

        # Verify execution completed
        assert stats is not None
        assert 'Return [%]' in stats.index

    def test_walk_forward_retraining(self, sample_data):
        """Test walk-forward retraining functionality"""
        # Use minimal retrain frequency for testing
        custom_params = {
            'retrain_frequency': 10,
            'n_train': 100
        }

        bt = Backtest(
            sample_data,
            MLWalkForwardStrategy,
            commission=0.0002,
            margin=0.05,
            cash=10000
        )

        # Should run without errors even with frequent retraining
        stats = bt.run(**custom_params)

        # Verify successful execution
        assert stats is not None
        assert isinstance(stats, pd.Series)


class TestStrategyRobustness:
    """Test strategy robustness and edge cases"""

    def test_insufficient_data_handling(self):
        """Test strategy behavior with minimal data"""
        # Create minimal dataset
        minimal_data = EURUSD.iloc[:50].copy()

        bt = Backtest(
            minimal_data,
            MLWalkForwardStrategy,
            commission=0.0002,
            margin=0.05,
            cash=10000
        )

        # Should handle gracefully without crashing
        stats = bt.run(n_train=30)
        assert stats is not None

    def test_strategy_with_different_timeframes(self):
        """Test strategy with different data frequencies"""
        # Test with daily data (resampled)
        daily_data = EURUSD.resample('D').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()

        if len(daily_data) >= 100:  # Ensure sufficient data
            bt = Backtest(
                daily_data.iloc[:100],
                MLWalkForwardStrategy,
                commission=0.0002,
                margin=0.05,
                cash=10000
            )

            stats = bt.run(n_train=50)
            assert stats is not None

    @pytest.mark.parametrize("strategy_class", [MLTrainOnceStrategy, MLWalkForwardStrategy])
    def test_strategy_consistency(self, sample_data, strategy_class):
        """Test both strategies produce consistent results"""
        bt = Backtest(
            sample_data,
            strategy_class,
            commission=0.0002,
            margin=0.05,
            cash=10000
        )

        # Run multiple times with same random seed (if applicable)
        stats1 = bt.run(n_train=200)
        stats2 = bt.run(n_train=200)

        # Results should be identical for deterministic strategies
        assert stats1['Return [%]'] == stats2['Return [%]']


class TestIntegration:
    """Integration tests for complete workflow"""

    def test_complete_ml_workflow(self, sample_data):
        """Test complete ML trading workflow"""
        # 1. Feature engineering
        df_features = create_features(sample_data)
        assert df_features is not None

        # 2. Data preparation
        X, y = get_clean_Xy(df_features)
        assert len(X) > 0 and len(y) > 0

        # 3. Strategy execution
        bt = Backtest(
            sample_data,
            MLWalkForwardStrategy,
            commission=0.0002,
            margin=0.05,
            cash=10000
        )

        stats = bt.run()

        # 4. Verify complete workflow
        assert stats is not None
        assert 'Sharpe Ratio' in stats.index

        # 5. Verify no major issues in execution
        # (Strategy should complete without exceptions)
        assert not pd.isna(stats['Return [%]'])


if __name__ == "__main__":
    """Allow running tests directly with python -m pytest"""
    pytest.main([__file__, "-v"])