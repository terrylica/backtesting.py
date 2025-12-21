"""Tests for triple-barrier classification system.

ADR: 2025-12-20-clickhouse-triple-barrier-backtest

Comprehensive test suite covering:
- Triple-barrier label generation
- Purged cross-validation
- Classifier training and prediction
- Calibration metrics
- Strategy integration
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_prices() -> pd.Series:
    """Generate sample price series for testing."""
    np.random.seed(42)
    n = 500
    # Random walk with drift
    returns = 0.0001 + 0.02 * np.random.randn(n)
    prices = 100 * np.exp(np.cumsum(returns))
    index = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.Series(prices, index=index, name="Close")


@pytest.fixture
def sample_ohlcv(sample_prices: pd.Series) -> pd.DataFrame:
    """Generate sample OHLCV data."""
    close = sample_prices.values
    # Generate realistic OHLC from close
    high = close * (1 + 0.01 * np.abs(np.random.randn(len(close))))
    low = close * (1 - 0.01 * np.abs(np.random.randn(len(close))))
    open_price = close * (1 + 0.005 * np.random.randn(len(close)))

    # Ensure OHLC constraints
    high = np.maximum(high, np.maximum(open_price, close))
    low = np.minimum(low, np.minimum(open_price, close))

    return pd.DataFrame(
        {
            "Open": open_price,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": np.random.poisson(1000, len(close)),
        },
        index=sample_prices.index,
    )


@pytest.fixture
def sample_features(sample_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Generate sample feature matrix."""
    df = sample_ohlcv
    features = pd.DataFrame(index=df.index)
    features["returns"] = df["Close"].pct_change()
    features["volatility"] = features["returns"].rolling(10).std()
    features["momentum"] = df["Close"] / df["Close"].shift(10) - 1
    features["range"] = (df["High"] - df["Low"]) / df["Close"]
    # Fill NaN with 0 for simplicity
    return features.fillna(0)


# =============================================================================
# Triple-Barrier Label Tests
# =============================================================================


class TestTripleBarrierLabels:
    """Tests for triple_barrier.py label generation."""

    def test_compute_labels_basic(self, sample_prices: pd.Series) -> None:
        """Test basic label computation."""
        from user_strategies.strategies.triple_barrier import (
            compute_triple_barrier_labels,
        )

        labels = compute_triple_barrier_labels(
            sample_prices, barrier_pct=0.02, horizon_bars=24
        )

        assert len(labels) == len(sample_prices)
        assert labels.name == "label"

        # Check valid labels are in {-1, 0, 1}
        valid_labels = labels.dropna()
        assert set(valid_labels.unique()).issubset({-1, 0, 1})

    def test_labels_have_nan_at_end(self, sample_prices: pd.Series) -> None:
        """Test that labels are NaN where insufficient forward data."""
        from user_strategies.strategies.triple_barrier import (
            compute_triple_barrier_labels,
        )

        horizon = 24
        labels = compute_triple_barrier_labels(
            sample_prices, barrier_pct=0.02, horizon_bars=horizon
        )

        # Last `horizon` labels should be NaN
        assert labels.iloc[-horizon:].isna().all()
        # Earlier labels should not all be NaN
        assert not labels.iloc[:-horizon].isna().all()

    def test_labels_with_trending_prices(self) -> None:
        """Test labels with strongly trending prices."""
        from user_strategies.strategies.triple_barrier import (
            LABEL_UPPER,
            compute_triple_barrier_labels,
        )

        # Strongly upward trending prices
        n = 100
        prices = pd.Series(
            100 * np.exp(np.cumsum(np.full(n, 0.01))),
            index=pd.date_range("2024-01-01", periods=n, freq="h"),
        )

        labels = compute_triple_barrier_labels(
            prices, barrier_pct=0.02, horizon_bars=10
        )

        # Most labels should be +1 (upper hit) for uptrend
        valid = labels.dropna()
        upper_ratio = (valid == LABEL_UPPER).mean()
        assert upper_ratio > 0.5, f"Expected >50% upper labels, got {upper_ratio:.1%}"

    def test_labels_with_ranging_prices(self) -> None:
        """Test labels with ranging (sideways) prices."""
        from user_strategies.strategies.triple_barrier import (
            LABEL_TIMEOUT,
            compute_triple_barrier_labels,
        )

        # Tight ranging prices
        n = 100
        prices = pd.Series(
            100 + 0.1 * np.sin(np.linspace(0, 4 * np.pi, n)),
            index=pd.date_range("2024-01-01", periods=n, freq="h"),
        )

        labels = compute_triple_barrier_labels(
            prices, barrier_pct=0.05, horizon_bars=10  # Wide barriers
        )

        # Most labels should be 0 (timeout) for ranging market
        valid = labels.dropna()
        timeout_ratio = (valid == LABEL_TIMEOUT).mean()
        assert timeout_ratio > 0.5, f"Expected >50% timeout labels, got {timeout_ratio:.1%}"

    def test_label_statistics(self, sample_prices: pd.Series) -> None:
        """Test label statistics computation."""
        from user_strategies.strategies.triple_barrier import (
            compute_triple_barrier_labels,
            get_label_statistics,
        )

        labels = compute_triple_barrier_labels(
            sample_prices, barrier_pct=0.02, horizon_bars=24
        )
        stats = get_label_statistics(labels)

        assert "n_total" in stats
        assert "n_upper" in stats
        assert "n_lower" in stats
        assert "n_timeout" in stats
        assert stats["n_upper"] + stats["n_lower"] + stats["n_timeout"] == stats["n_total"]

    def test_invalid_inputs(self, sample_prices: pd.Series) -> None:
        """Test error handling for invalid inputs."""
        from user_strategies.strategies.triple_barrier import (
            compute_triple_barrier_labels,
        )

        with pytest.raises(ValueError, match="barrier_pct must be positive"):
            compute_triple_barrier_labels(sample_prices, barrier_pct=-0.01, horizon_bars=24)

        with pytest.raises(ValueError, match="horizon_bars must be positive"):
            compute_triple_barrier_labels(sample_prices, barrier_pct=0.01, horizon_bars=0)


# =============================================================================
# Purged CV Tests
# =============================================================================


class TestPurgedCV:
    """Tests for purged_cv.py cross-validation."""

    def test_purged_kfold_basic(self) -> None:
        """Test basic PurgedKFold functionality."""
        from user_strategies.strategies.purged_cv import PurgedKFold

        X = np.random.randn(100, 5)
        y = np.random.randint(0, 3, 100)

        cv = PurgedKFold(n_splits=5, horizon=10, embargo=0)

        folds = list(cv.split(X, y))
        assert len(folds) == 5

        for train_idx, test_idx in folds:
            # No overlap between train and test
            assert len(set(train_idx) & set(test_idx)) == 0
            # Indices are valid
            assert train_idx.max() < len(X)
            assert test_idx.max() < len(X)

    def test_purged_kfold_purges_correctly(self) -> None:
        """Test that purging removes samples near test boundary."""
        from user_strategies.strategies.purged_cv import PurgedKFold

        X = np.random.randn(100, 5)
        horizon = 10

        cv = PurgedKFold(n_splits=5, horizon=horizon, embargo=0)

        for train_idx, test_idx in cv.split(X):
            test_start = test_idx.min()
            # No training samples within `horizon` of test start
            train_near_test = train_idx[train_idx >= test_start - horizon]
            train_near_test = train_near_test[train_near_test < test_start]
            assert len(train_near_test) == 0, "Purging failed"

    def test_purged_time_series_split(self) -> None:
        """Test PurgedTimeSeriesSplit."""
        from user_strategies.strategies.purged_cv import PurgedTimeSeriesSplit

        X = np.random.randn(200, 5)
        cv = PurgedTimeSeriesSplit(n_splits=5, horizon=10, embargo=5)

        folds = list(cv.split(X))
        assert len(folds) <= 5

        # Training set should expand
        prev_train_size = 0
        for train_idx, test_idx in folds:
            assert len(train_idx) >= prev_train_size
            prev_train_size = len(train_idx)

    def test_train_test_split(self) -> None:
        """Test purged_train_test_split function."""
        from user_strategies.strategies.purged_cv import purged_train_test_split

        X = np.random.randn(100, 5)
        y = np.random.randint(0, 3, 100)

        X_train, X_test, y_train, y_test = purged_train_test_split(
            X, y, test_size=0.2, horizon=10, embargo=5
        )

        assert len(X_train) + len(X_test) < len(X)  # Some samples purged
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)


# =============================================================================
# Classifier Tests
# =============================================================================


class TestClassifier:
    """Tests for classifier.py."""

    def test_classifier_fit_predict(
        self, sample_features: pd.DataFrame, sample_prices: pd.Series
    ) -> None:
        """Test classifier training and prediction."""
        from user_strategies.strategies.classifier import TripleBarrierClassifier
        from user_strategies.strategies.triple_barrier import (
            compute_triple_barrier_labels,
        )

        labels = compute_triple_barrier_labels(
            sample_prices, barrier_pct=0.02, horizon_bars=24
        )

        # Align and remove NaN
        valid_mask = ~labels.isna()
        X = sample_features.loc[valid_mask].values
        y = labels.loc[valid_mask].values

        # Train/test split
        split = int(len(X) * 0.7)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        clf = TripleBarrierClassifier(model_type="logistic", calibrated=False)
        clf.fit(X_train, y_train)

        # Test predictions
        y_pred = clf.predict(X_test)
        assert len(y_pred) == len(X_test)
        assert set(y_pred).issubset({-1, 0, 1})

        # Test probabilities
        y_proba = clf.predict_proba(X_test)
        assert y_proba.shape == (len(X_test), 3)
        assert np.allclose(y_proba.sum(axis=1), 1.0)

    def test_classifier_model_types(self, sample_features: pd.DataFrame) -> None:
        """Test different classifier model types."""
        from user_strategies.strategies.classifier import TripleBarrierClassifier

        X = sample_features.iloc[:100].values
        y = np.random.choice([-1, 0, 1], 100)

        for model_type in ["logistic", "rf", "gbm"]:
            clf = TripleBarrierClassifier(model_type=model_type, calibrated=False)
            clf.fit(X, y)
            proba = clf.predict_proba(X)
            assert proba.shape == (100, 3)

    def test_predict_proba_directional(self, sample_features: pd.DataFrame) -> None:
        """Test directional probability prediction."""
        from user_strategies.strategies.classifier import TripleBarrierClassifier

        X = sample_features.iloc[:100].values
        y = np.random.choice([-1, 0, 1], 100)

        clf = TripleBarrierClassifier(model_type="logistic")
        clf.fit(X, y)

        direction = clf.predict_proba_directional(X)
        assert len(direction) == 100
        assert direction.min() >= -1
        assert direction.max() <= 1


# =============================================================================
# Calibration Tests
# =============================================================================


class TestCalibration:
    """Tests for calibration.py diagnostics."""

    def test_brier_score(self) -> None:
        """Test Brier score computation."""
        from user_strategies.strategies.calibration import compute_brier_score

        # Perfect predictions
        y_true = np.array([1, -1, 0])
        y_proba = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])

        brier = compute_brier_score(y_true, y_proba)
        assert brier == 0.0

        # Random predictions should have higher Brier score
        y_proba_random = np.full((3, 3), 1 / 3)
        brier_random = compute_brier_score(y_true, y_proba_random)
        assert brier_random > 0

    def test_ece_computation(self) -> None:
        """Test Expected Calibration Error."""
        from user_strategies.strategies.calibration import compute_ece

        np.random.seed(42)
        n = 1000
        y_true = np.random.choice([-1, 0, 1], n)
        y_proba = np.random.dirichlet([1, 1, 1], n)

        ece = compute_ece(y_true, y_proba, n_bins=10)
        assert 0 <= ece <= 1

    def test_reliability_diagram_data(self) -> None:
        """Test reliability diagram data generation."""
        from user_strategies.strategies.calibration import (
            compute_reliability_diagram_data,
        )

        np.random.seed(42)
        n = 500
        y_true = np.random.choice([-1, 0, 1], n)
        y_proba = np.random.dirichlet([1, 1, 1], n)

        data = compute_reliability_diagram_data(y_true, y_proba, n_bins=10)

        assert len(data) == 10
        assert "bin_center" in data.columns
        assert "predicted_prob" in data.columns
        assert "observed_freq" in data.columns
        assert "count" in data.columns

    def test_calibration_metrics(self) -> None:
        """Test comprehensive calibration metrics."""
        from user_strategies.strategies.calibration import compute_calibration_metrics

        np.random.seed(42)
        n = 500
        y_true = np.random.choice([-1, 0, 1], n)
        y_proba = np.random.dirichlet([1, 1, 1], n)

        metrics = compute_calibration_metrics(y_true, y_proba)

        assert metrics.brier_score >= 0
        assert 0 <= metrics.ece <= 1
        assert 0 <= metrics.mce <= 1
        assert len(metrics.class_brier_scores) == 3


# =============================================================================
# Barrier Config Tests
# =============================================================================


class TestBarrierConfig:
    """Tests for barrier_config.py."""

    def test_barrier_config_creation(self) -> None:
        """Test BarrierConfig dataclass."""
        from user_strategies.configs.barrier_config import BarrierConfig

        config = BarrierConfig(barrier_pct=0.01, horizon_bars=24)
        assert config.barrier_pct == 0.01
        assert config.horizon_bars == 24
        assert config.symmetric is True

    def test_barrier_config_validation(self) -> None:
        """Test BarrierConfig validation."""
        from user_strategies.configs.barrier_config import BarrierConfig

        with pytest.raises(ValueError, match="barrier_pct must be positive"):
            BarrierConfig(barrier_pct=-0.01, horizon_bars=24)

        with pytest.raises(ValueError, match="horizon_bars must be positive"):
            BarrierConfig(barrier_pct=0.01, horizon_bars=0)

    def test_config_grid_generation(self) -> None:
        """Test configuration grid generation."""
        from user_strategies.configs.barrier_config import (
            BARRIER_PCT_GRID,
            HORIZON_BARS_GRID,
            generate_config_grid,
        )

        configs = generate_config_grid()
        expected_count = len(BARRIER_PCT_GRID) * len(HORIZON_BARS_GRID)
        assert len(configs) == expected_count

    def test_timeframe_config(self) -> None:
        """Test timeframe-specific config generation."""
        from user_strategies.configs.barrier_config import get_config_for_timeframe

        config_1m = get_config_for_timeframe("1m")
        config_1h = get_config_for_timeframe("1h")
        config_1d = get_config_for_timeframe("1d")

        # Higher timeframes should have wider barriers
        assert config_1m.barrier_pct < config_1h.barrier_pct < config_1d.barrier_pct


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for the complete pipeline."""

    def test_full_pipeline_with_sample_data(
        self, sample_ohlcv: pd.DataFrame, sample_prices: pd.Series
    ) -> None:
        """Test complete pipeline with sample data."""
        from user_strategies.strategies.calibration import compute_calibration_metrics
        from user_strategies.strategies.classifier import TripleBarrierClassifier
        from user_strategies.strategies.purged_cv import purged_train_test_split
        from user_strategies.strategies.triple_barrier import (
            compute_triple_barrier_labels,
        )

        # Compute features
        features = pd.DataFrame(index=sample_ohlcv.index)
        features["returns"] = sample_ohlcv["Close"].pct_change().fillna(0)
        features["volatility"] = features["returns"].rolling(10).std().fillna(0)

        # Generate labels
        labels = compute_triple_barrier_labels(
            sample_prices, barrier_pct=0.02, horizon_bars=24
        )

        # Align and split
        valid_mask = ~labels.isna()
        X = features.loc[valid_mask].values
        y = labels.loc[valid_mask].values

        X_train, X_test, y_train, y_test = purged_train_test_split(
            X, y, test_size=0.3, horizon=24
        )

        # Train classifier
        clf = TripleBarrierClassifier(model_type="logistic")
        clf.fit(X_train, y_train)

        # Predict and evaluate
        y_proba = clf.predict_proba(X_test)
        metrics = compute_calibration_metrics(y_test, y_proba)

        # Basic sanity checks
        assert metrics.brier_score < 1.0
        assert metrics.ece < 0.5
