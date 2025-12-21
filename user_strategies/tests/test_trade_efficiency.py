"""Unit tests for trade efficiency MAE/MFE analysis module.

Tests the core calculation functions and report generation for trade efficiency
metrics based on Maximum Adverse Excursion (MAE) and Maximum Favorable Excursion (MFE).
"""

import numpy as np
import pandas as pd
import pytest

from user_strategies.strategies.trade_efficiency import (
    TradeEfficiencyMetrics,
    TradeEfficiencyReport,
    calculate_mae_mfe,
    calculate_trade_efficiency,
    print_comparison_table,
)


@pytest.fixture
def sample_ohlcv():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range("2024-01-01", periods=20, freq="h")
    return pd.DataFrame(
        {
            "Open": [100, 101, 102, 103, 104, 105, 104, 103, 102, 101,
                     100, 99, 98, 97, 96, 97, 98, 99, 100, 101],
            "High": [102, 103, 104, 105, 106, 107, 106, 105, 104, 103,
                     102, 101, 100, 99, 98, 99, 100, 101, 102, 103],
            "Low": [99, 100, 101, 102, 103, 104, 103, 102, 101, 100,
                    99, 98, 97, 96, 95, 96, 97, 98, 99, 100],
            "Close": [101, 102, 103, 104, 105, 104, 103, 102, 101, 100,
                      99, 98, 97, 96, 97, 98, 99, 100, 101, 102],
            "Volume": [1000] * 20,
        },
        index=dates,
    )


@pytest.fixture
def sample_long_trade():
    """Create sample long trade DataFrame."""
    return pd.DataFrame(
        {
            "EntryBar": [0],
            "ExitBar": [5],
            "EntryPrice": [100.0],
            "ExitPrice": [104.0],
            "Size": [1.0],  # Positive = long
            "EntryTime": pd.Timestamp("2024-01-01 00:00"),
            "ExitTime": pd.Timestamp("2024-01-01 05:00"),
        }
    )


@pytest.fixture
def sample_short_trade():
    """Create sample short trade DataFrame."""
    return pd.DataFrame(
        {
            "EntryBar": [5],
            "ExitBar": [14],
            "EntryPrice": [105.0],
            "ExitPrice": [96.0],
            "Size": [-1.0],  # Negative = short
            "EntryTime": pd.Timestamp("2024-01-01 05:00"),
            "ExitTime": pd.Timestamp("2024-01-01 14:00"),
        }
    )


class TestCalculateMaeMfe:
    """Tests for calculate_mae_mfe function."""

    def test_long_trade_mfe_mae(self, sample_ohlcv):
        """Test MFE/MAE calculation for long position."""
        # Long trade from bar 0 to bar 5 (entry at 100, exit at 104)
        # High range: 102, 103, 104, 105, 106, 107 -> max = 107
        # Low range: 99, 100, 101, 102, 103, 104 -> min = 99
        mfe_pct, mae_pct, actual_pct = calculate_mae_mfe(
            sample_ohlcv, entry_bar=0, exit_bar=5, entry_price=100.0, is_long=True
        )

        # MFE = (107 - 100) / 100 * 100 = 7%
        assert mfe_pct == pytest.approx(7.0, rel=1e-2)

        # MAE = (99 - 100) / 100 * 100 = -1%
        assert mae_pct == pytest.approx(-1.0, rel=1e-2)

        # Actual = (104 - 100) / 100 * 100 = 4%
        assert actual_pct == pytest.approx(4.0, rel=1e-2)

    def test_short_trade_mfe_mae(self, sample_ohlcv):
        """Test MFE/MAE calculation for short position."""
        # Short trade from bar 5 to bar 14 (entry at 105)
        # For short: MFE = entry - min(Low), MAE = entry - max(High)
        # Low range from bar 5-14: 104, 103, 102, 101, 100, 99, 98, 97, 96, 95 -> min = 95
        # High range from bar 5-14: 107, 106, 105, 104, 103, 102, 101, 100, 99, 98 -> max = 107
        # Close at bar 14 = 97 (from fixture)
        mfe_pct, mae_pct, actual_pct = calculate_mae_mfe(
            sample_ohlcv, entry_bar=5, exit_bar=14, entry_price=105.0, is_long=False
        )

        # MFE = (105 - 95) / 105 * 100 = 9.52%
        assert mfe_pct == pytest.approx(9.52, rel=1e-2)

        # MAE = (105 - 107) / 105 * 100 = -1.90%
        assert mae_pct == pytest.approx(-1.90, rel=1e-2)

        # Actual = (105 - 97) / 105 * 100 = 7.62% (Close[14] = 97)
        assert actual_pct == pytest.approx(7.62, rel=1e-2)

    def test_single_bar_trade(self, sample_ohlcv):
        """Test calculation for single-bar trade."""
        mfe_pct, mae_pct, actual_pct = calculate_mae_mfe(
            sample_ohlcv, entry_bar=0, exit_bar=0, entry_price=100.0, is_long=True
        )

        # MFE = (102 - 100) / 100 * 100 = 2%
        assert mfe_pct == pytest.approx(2.0, rel=1e-2)

        # MAE = (99 - 100) / 100 * 100 = -1%
        assert mae_pct == pytest.approx(-1.0, rel=1e-2)


class TestCalculateTradeEfficiency:
    """Tests for calculate_trade_efficiency function."""

    def test_single_long_trade(self, sample_ohlcv, sample_long_trade):
        """Test efficiency calculation for single long trade."""
        report = calculate_trade_efficiency(sample_long_trade, sample_ohlcv)

        assert report.n_trades == 1
        assert report.n_long == 1
        assert report.n_short == 0
        assert len(report.trades) == 1

        # Efficiency = actual / MFE = 4 / 7 ≈ 0.571
        trade = report.trades[0]
        assert trade.efficiency == pytest.approx(0.571, rel=1e-2)
        assert trade.direction == "LONG"

    def test_single_short_trade(self, sample_ohlcv, sample_short_trade):
        """Test efficiency calculation for single short trade."""
        report = calculate_trade_efficiency(sample_short_trade, sample_ohlcv)

        assert report.n_trades == 1
        assert report.n_long == 0
        assert report.n_short == 1
        assert len(report.trades) == 1

        # Efficiency = actual / MFE = 7.62 / 9.52 ≈ 0.80
        trade = report.trades[0]
        assert trade.efficiency == pytest.approx(0.80, rel=1e-2)
        assert trade.direction == "SHORT"

    def test_empty_trades(self, sample_ohlcv):
        """Test handling of empty trades DataFrame."""
        empty_trades = pd.DataFrame()
        report = calculate_trade_efficiency(empty_trades, sample_ohlcv)

        assert report.n_trades == 0
        assert len(report.trades) == 0
        assert report.aggregate_efficiency == 0

    def test_none_trades(self, sample_ohlcv):
        """Test handling of None trades input."""
        report = calculate_trade_efficiency(None, sample_ohlcv)

        assert report.n_trades == 0
        assert len(report.trades) == 0

    def test_aggregate_efficiency(self, sample_ohlcv, sample_long_trade):
        """Test aggregate efficiency calculation."""
        report = calculate_trade_efficiency(sample_long_trade, sample_ohlcv)

        # Aggregate = total_captured / total_mfe * 100
        expected_agg = (report.total_captured / report.total_mfe_available) * 100
        assert report.aggregate_efficiency == pytest.approx(expected_agg, rel=1e-2)


class TestTradeEfficiencyReport:
    """Tests for TradeEfficiencyReport class."""

    def test_summary_output(self, sample_ohlcv, sample_long_trade):
        """Test summary string generation."""
        report = calculate_trade_efficiency(sample_long_trade, sample_ohlcv)
        summary = report.summary()

        assert "TRADE EFFICIENCY REPORT" in summary
        assert "Total Trades: 1" in summary
        assert "Long: 1" in summary
        assert "MFE" in summary

    def test_summary_with_trades(self, sample_ohlcv, sample_long_trade):
        """Test summary with per-trade details."""
        report = calculate_trade_efficiency(sample_long_trade, sample_ohlcv)
        summary = report.summary(include_trades=True)

        assert "Per-Trade Details" in summary
        assert "LONG" in summary

    def test_to_dataframe(self, sample_ohlcv, sample_long_trade):
        """Test DataFrame export."""
        report = calculate_trade_efficiency(sample_long_trade, sample_ohlcv)
        df = report.to_dataframe()

        assert len(df) == 1
        assert "EntryTime" in df.columns
        assert "Efficiency" in df.columns
        assert "MFE%" in df.columns
        assert "MAE%" in df.columns

    def test_empty_report_to_dataframe(self):
        """Test DataFrame export for empty report."""
        report = TradeEfficiencyReport()
        df = report.to_dataframe()

        assert df.empty


class TestPrintComparisonTable:
    """Tests for print_comparison_table function."""

    def test_comparison_output(self, sample_ohlcv, sample_long_trade, sample_short_trade):
        """Test comparison table generation."""
        long_report = calculate_trade_efficiency(sample_long_trade, sample_ohlcv)
        short_report = calculate_trade_efficiency(sample_short_trade, sample_ohlcv)

        table = print_comparison_table(long_report, short_report)

        assert "LONG vs SHORT EFFICIENCY COMPARISON" in table
        assert "LONG-ONLY" in table
        assert "SHORT-ONLY" in table
        assert "Win Rate" in table
        assert "Aggregate Efficiency" in table

    def test_comparison_with_none(self):
        """Test comparison table with None reports."""
        table = print_comparison_table(None, None)

        assert "LONG vs SHORT EFFICIENCY COMPARISON" in table
        assert "0" in table  # Zero trades


class TestTradeEfficiencyMetrics:
    """Tests for TradeEfficiencyMetrics dataclass."""

    def test_metrics_creation(self):
        """Test creating metrics dataclass."""
        metrics = TradeEfficiencyMetrics(
            entry_time=pd.Timestamp("2024-01-01"),
            exit_time=pd.Timestamp("2024-01-02"),
            direction="LONG",
            entry_price=100.0,
            exit_price=105.0,
            actual_pct=5.0,
            mfe_pct=7.0,
            mae_pct=-2.0,
            efficiency=0.714,
            etd_pct=2.0,
            bars_held=24,
        )

        assert metrics.direction == "LONG"
        assert metrics.efficiency == 0.714
        assert metrics.bars_held == 24


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_mfe(self, sample_ohlcv):
        """Test handling of zero MFE (no favorable movement)."""
        # Create a trade where price never moves favorably
        flat_ohlcv = pd.DataFrame(
            {
                "Open": [100] * 5,
                "High": [100] * 5,  # Never goes above entry
                "Low": [98] * 5,
                "Close": [99] * 5,
                "Volume": [1000] * 5,
            },
            index=pd.date_range("2024-01-01", periods=5, freq="h"),
        )

        trades = pd.DataFrame(
            {
                "EntryBar": [0],
                "ExitBar": [4],
                "EntryPrice": [100.0],
                "ExitPrice": [99.0],
                "Size": [1.0],
                "EntryTime": pd.Timestamp("2024-01-01"),
                "ExitTime": pd.Timestamp("2024-01-01 04:00"),
            }
        )

        report = calculate_trade_efficiency(trades, flat_ohlcv)

        # When MFE is 0, efficiency should be NaN
        assert len(report.trades) == 1
        assert np.isnan(report.trades[0].efficiency)

    def test_negative_return_with_positive_mfe(self, sample_ohlcv):
        """Test trade with loss despite favorable movement (negative efficiency)."""
        # Trade that loses money despite price moving favorably at some point
        trades = pd.DataFrame(
            {
                "EntryBar": [0],
                "ExitBar": [10],
                "EntryPrice": [100.0],
                "ExitPrice": [99.0],  # Exit at loss
                "Size": [1.0],
                "EntryTime": pd.Timestamp("2024-01-01"),
                "ExitTime": pd.Timestamp("2024-01-01 10:00"),
            }
        )

        report = calculate_trade_efficiency(trades, sample_ohlcv)

        trade = report.trades[0]
        assert trade.actual_pct < 0  # Loss
        assert trade.mfe_pct > 0  # Had opportunity
        assert trade.efficiency < 0  # Negative efficiency

    def test_win_rate_calculation(self, sample_ohlcv):
        """Test win rate is correctly calculated."""
        # Create mix of winning and losing trades
        trades = pd.DataFrame(
            {
                "EntryBar": [0, 5],
                "ExitBar": [5, 10],
                "EntryPrice": [100.0, 105.0],
                "ExitPrice": [105.0, 100.0],  # Win, then loss
                "Size": [1.0, 1.0],
                "EntryTime": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01 05:00")],
                "ExitTime": [pd.Timestamp("2024-01-01 05:00"), pd.Timestamp("2024-01-01 10:00")],
            }
        )

        report = calculate_trade_efficiency(trades, sample_ohlcv)

        # 1 win out of 2 trades = 50% win rate
        assert report.win_rate == pytest.approx(50.0, rel=1e-2)
