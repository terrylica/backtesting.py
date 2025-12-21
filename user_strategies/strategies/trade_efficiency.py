"""Trade Efficiency Analysis using MAE/MFE metrics.

ADR: 2025-12-20-trade-efficiency-mae-mfe

This module provides trade efficiency analysis using Maximum Adverse Excursion (MAE)
and Maximum Favorable Excursion (MFE) metrics. These metrics benchmark each trade
against its maximum possible outcome during the holding period.

Mathematical Framework:
----------------------
For a trade entered at price P_entry and exited at price P_exit:

    MFE (Maximum Favorable Excursion):
        Long:  MFE = (max(High[entry:exit]) - P_entry) / P_entry
        Short: MFE = (P_entry - min(Low[entry:exit])) / P_entry

    MAE (Maximum Adverse Excursion):
        Long:  MAE = (min(Low[entry:exit]) - P_entry) / P_entry
        Short: MAE = (P_entry - max(High[entry:exit])) / P_entry

    Trade Efficiency:
        Efficiency = Actual_Return / MFE

    Where:
        Efficiency = 1.0  -> Perfect capture (exited at best possible price)
        Efficiency = 0.5  -> Captured half of available profit
        Efficiency < 0    -> Loss despite favorable price movement

Key Insight:
-----------
Unlike Buy & Hold comparison, efficiency benchmarks each trade against what was
ACTUALLY ACHIEVABLE during that specific trade. A trade in a sideways market with
50% efficiency may be better executed than a trade in a trending market with 20%
efficiency.

References:
----------
- Sweeney, John. "Maximum Adverse Excursion." Technical Analysis of Stocks & Commodities
- Lopez de Prado, M. "Advances in Financial Machine Learning" (Ch. 10: Bet Sizing)

Example Usage:
-------------
    >>> from backtesting import Backtest
    >>> from user_strategies.strategies.trade_efficiency import (
    ...     calculate_trade_efficiency,
    ...     TradeEfficiencyReport,
    ... )
    >>> stats = bt.run()
    >>> report = calculate_trade_efficiency(stats['_trades'], ohlcv_data)
    >>> print(report.summary())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class TradeEfficiencyMetrics:
    """Efficiency metrics for a single trade.

    Attributes:
        entry_time: Entry timestamp
        exit_time: Exit timestamp
        direction: 'LONG' or 'SHORT'
        entry_price: Trade entry price
        exit_price: Trade exit price
        actual_pct: Actual return percentage
        mfe_pct: Maximum Favorable Excursion percentage
        mae_pct: Maximum Adverse Excursion percentage
        efficiency: Trade efficiency ratio (actual/MFE)
        etd_pct: End Trade Drawdown (MFE - actual, profit given back)
        bars_held: Number of bars held
    """

    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: str
    entry_price: float
    exit_price: float
    actual_pct: float
    mfe_pct: float
    mae_pct: float
    efficiency: float
    etd_pct: float
    bars_held: int


@dataclass
class TradeEfficiencyReport:
    """Comprehensive trade efficiency analysis report.

    Attributes:
        trades: List of individual trade efficiency metrics
        n_trades: Total number of trades
        n_long: Number of long trades
        n_short: Number of short trades
        mean_efficiency: Mean efficiency across all trades
        median_efficiency: Median efficiency
        aggregate_efficiency: Total actual / Total MFE
        total_mfe_available: Sum of all MFE percentages
        total_captured: Sum of all actual returns
        win_rate: Percentage of profitable trades
    """

    trades: list[TradeEfficiencyMetrics] = field(default_factory=list)
    n_trades: int = 0
    n_long: int = 0
    n_short: int = 0
    mean_efficiency: float = 0.0
    median_efficiency: float = 0.0
    aggregate_efficiency: float = 0.0
    total_mfe_available: float = 0.0
    total_captured: float = 0.0
    win_rate: float = 0.0

    def summary(self, include_trades: bool = False) -> str:
        """Generate human-readable summary of efficiency analysis.

        Args:
            include_trades: If True, include per-trade details

        Returns:
            Formatted string summary
        """
        lines = [
            "=" * 70,
            "TRADE EFFICIENCY REPORT (MAE/MFE Analysis)",
            "=" * 70,
            "",
            f"Total Trades: {self.n_trades}",
            f"  Long: {self.n_long}",
            f"  Short: {self.n_short}",
            f"  Win Rate: {self.win_rate:.1f}%",
            "",
            "MFE (Maximum Favorable Excursion):",
            f"  Total Available: {self.total_mfe_available:.2f}%",
            "",
            "Actual Returns:",
            f"  Total Captured: {self.total_captured:.2f}%",
            "",
            "*** TRADE EFFICIENCY ***",
            f"  Aggregate: {self.aggregate_efficiency:.1f}%",
            f"  Mean: {self.mean_efficiency:.3f}",
            f"  Median: {self.median_efficiency:.3f}",
            "",
        ]

        # Efficiency distribution
        if self.trades:
            eff_values = [t.efficiency for t in self.trades if not np.isnan(t.efficiency)]
            eff_capped = np.clip(eff_values, -2, 1)

            lines.extend([
                "Distribution:",
                f"  Perfect (>0.9): {sum(1 for e in eff_capped if e > 0.9)}",
                f"  Good (0.5-0.9): {sum(1 for e in eff_capped if 0.5 < e <= 0.9)}",
                f"  Moderate (0.25-0.5): {sum(1 for e in eff_capped if 0.25 < e <= 0.5)}",
                f"  Weak (0-0.25): {sum(1 for e in eff_capped if 0 <= e <= 0.25)}",
                f"  Loss (<0): {sum(1 for e in eff_capped if e < 0)}",
                "",
            ])

        # Interpretation
        lines.append("Interpretation:")
        if self.aggregate_efficiency >= 50:
            lines.append(f"  GOOD: Capturing {self.aggregate_efficiency:.0f}% of available profit")
        elif self.aggregate_efficiency >= 25:
            lines.append(f"  MODERATE: Capturing {self.aggregate_efficiency:.0f}% of available profit")
            lines.append("  Opportunity: Exit timing improvements could increase capture")
        elif self.aggregate_efficiency >= 0:
            lines.append(f"  WEAK: Only capturing {self.aggregate_efficiency:.0f}% of available profit")
        else:
            lines.append(f"  NEGATIVE: Giving back profit and more ({self.aggregate_efficiency:.0f}%)")
            lines.append("  Issue: Exits are consistently worse than entry")

        if include_trades and self.trades:
            lines.extend([
                "",
                "-" * 70,
                "Per-Trade Details:",
                "-" * 70,
                f"{'#':>3} {'Dir':>5} {'Actual%':>9} {'MFE%':>8} {'Eff':>8} {'Bars':>5}",
                "-" * 70,
            ])
            for i, t in enumerate(self.trades[:20]):  # Limit to first 20
                eff_str = f"{t.efficiency:.2f}" if not np.isnan(t.efficiency) else "N/A"
                lines.append(
                    f"{i+1:>3} {t.direction:>5} {t.actual_pct:>9.3f} "
                    f"{t.mfe_pct:>8.3f} {eff_str:>8} {t.bars_held:>5}"
                )
            if len(self.trades) > 20:
                lines.append(f"... and {len(self.trades) - 20} more trades")

        lines.append("=" * 70)
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert trade metrics to DataFrame for further analysis.

        Returns:
            DataFrame with one row per trade
        """
        if not self.trades:
            return pd.DataFrame()

        return pd.DataFrame([
            {
                "EntryTime": t.entry_time,
                "ExitTime": t.exit_time,
                "Direction": t.direction,
                "EntryPrice": t.entry_price,
                "ExitPrice": t.exit_price,
                "Actual%": t.actual_pct,
                "MFE%": t.mfe_pct,
                "MAE%": t.mae_pct,
                "Efficiency": t.efficiency,
                "ETD%": t.etd_pct,
                "Bars": t.bars_held,
            }
            for t in self.trades
        ])


def calculate_mae_mfe(
    ohlcv: pd.DataFrame,
    entry_bar: int,
    exit_bar: int,
    entry_price: float,
    is_long: bool,
) -> tuple[float, float, float]:
    """Calculate MAE, MFE, and actual return for a trade.

    Args:
        ohlcv: OHLCV DataFrame with High, Low, Close columns
        entry_bar: Bar index at entry
        exit_bar: Bar index at exit
        entry_price: Trade entry price
        is_long: True for long positions, False for short

    Returns:
        Tuple of (mfe_pct, mae_pct, actual_pct) as percentages
    """
    # Get price range during trade
    trade_highs = ohlcv["High"].iloc[entry_bar : exit_bar + 1]
    trade_lows = ohlcv["Low"].iloc[entry_bar : exit_bar + 1]
    exit_price = ohlcv["Close"].iloc[exit_bar]

    if is_long:
        # Long: MFE = best high, MAE = worst low
        mfe_price = trade_highs.max()
        mae_price = trade_lows.min()
        mfe_pct = (mfe_price - entry_price) / entry_price * 100
        mae_pct = (mae_price - entry_price) / entry_price * 100  # Negative for loss
        actual_pct = (exit_price - entry_price) / entry_price * 100
    else:
        # Short: MFE = best low (profit), MAE = worst high (loss)
        mfe_price = trade_lows.min()
        mae_price = trade_highs.max()
        mfe_pct = (entry_price - mfe_price) / entry_price * 100  # Positive
        mae_pct = (entry_price - mae_price) / entry_price * 100  # Negative for loss
        actual_pct = (entry_price - exit_price) / entry_price * 100

    return mfe_pct, mae_pct, actual_pct


def calculate_trade_efficiency(
    trades_df: pd.DataFrame,
    ohlcv: pd.DataFrame,
) -> TradeEfficiencyReport:
    """Calculate efficiency metrics for all trades.

    This function analyzes each trade from a backtesting.py backtest and computes
    MAE/MFE efficiency metrics.

    Args:
        trades_df: DataFrame from stats['_trades'] with columns:
            EntryBar, ExitBar, EntryPrice, ExitPrice, Size, EntryTime, ExitTime
        ohlcv: OHLCV DataFrame used in the backtest

    Returns:
        TradeEfficiencyReport with comprehensive analysis

    Example:
        >>> stats = bt.run()
        >>> report = calculate_trade_efficiency(stats['_trades'], data)
        >>> print(report.summary())
    """
    if trades_df is None or len(trades_df) == 0:
        logger.warning("No trades to analyze")
        return TradeEfficiencyReport()

    trade_metrics = []

    for _, trade in trades_df.iterrows():
        entry_bar = int(trade["EntryBar"])
        exit_bar = int(trade["ExitBar"])
        entry_price = trade["EntryPrice"]
        exit_price = trade["ExitPrice"]
        is_long = trade["Size"] > 0

        # Calculate MAE/MFE
        mfe_pct, mae_pct, actual_pct = calculate_mae_mfe(
            ohlcv, entry_bar, exit_bar, entry_price, is_long
        )

        # Calculate efficiency
        if mfe_pct > 0:
            efficiency = actual_pct / mfe_pct
        else:
            efficiency = np.nan

        # ETD: End Trade Drawdown (profit given back)
        etd_pct = mfe_pct - actual_pct

        trade_metrics.append(
            TradeEfficiencyMetrics(
                entry_time=trade.get("EntryTime", pd.NaT),
                exit_time=trade.get("ExitTime", pd.NaT),
                direction="LONG" if is_long else "SHORT",
                entry_price=entry_price,
                exit_price=exit_price,
                actual_pct=actual_pct,
                mfe_pct=mfe_pct,
                mae_pct=mae_pct,
                efficiency=efficiency,
                etd_pct=etd_pct,
                bars_held=exit_bar - entry_bar,
            )
        )

    # Aggregate statistics
    n_trades = len(trade_metrics)
    n_long = sum(1 for t in trade_metrics if t.direction == "LONG")
    n_short = n_trades - n_long

    actual_values = [t.actual_pct for t in trade_metrics]
    mfe_values = [t.mfe_pct for t in trade_metrics]
    eff_values = [t.efficiency for t in trade_metrics if not np.isnan(t.efficiency)]

    total_mfe = sum(mfe_values)
    total_actual = sum(actual_values)
    aggregate_eff = (total_actual / total_mfe * 100) if total_mfe != 0 else 0

    # Cap efficiencies for mean/median to avoid outlier distortion
    if len(eff_values) > 0:
        eff_capped = np.clip(eff_values, -2, 1)
        mean_eff = float(np.mean(eff_capped))
        median_eff = float(np.median(eff_capped))
    else:
        mean_eff = 0.0
        median_eff = 0.0

    return TradeEfficiencyReport(
        trades=trade_metrics,
        n_trades=n_trades,
        n_long=n_long,
        n_short=n_short,
        mean_efficiency=mean_eff,
        median_efficiency=median_eff,
        aggregate_efficiency=aggregate_eff,
        total_mfe_available=total_mfe,
        total_captured=total_actual,
        win_rate=(sum(1 for a in actual_values if a > 0) / n_trades * 100) if n_trades > 0 else 0.0,
    )


def analyze_efficiency_by_direction(
    long_trades_df: pd.DataFrame | None,
    short_trades_df: pd.DataFrame | None,
    ohlcv: pd.DataFrame,
) -> dict[str, TradeEfficiencyReport]:
    """Analyze efficiency separately for long and short trades.

    This provides fair comparison when sequential blocking may affect results.
    Run separate long-only and short-only backtests, then analyze each.

    Args:
        long_trades_df: Trades from long-only backtest (stats['_trades'])
        short_trades_df: Trades from short-only backtest (stats['_trades'])
        ohlcv: OHLCV DataFrame used in backtests

    Returns:
        Dict with 'long' and 'short' TradeEfficiencyReport objects
    """
    results = {}

    if long_trades_df is not None and len(long_trades_df) > 0:
        results["long"] = calculate_trade_efficiency(long_trades_df, ohlcv)

    if short_trades_df is not None and len(short_trades_df) > 0:
        results["short"] = calculate_trade_efficiency(short_trades_df, ohlcv)

    return results


def print_comparison_table(
    long_report: TradeEfficiencyReport | None,
    short_report: TradeEfficiencyReport | None,
) -> str:
    """Generate comparison table for long vs short efficiency.

    Args:
        long_report: Efficiency report from long-only backtest
        short_report: Efficiency report from short-only backtest

    Returns:
        Formatted comparison table string
    """
    lines = [
        "=" * 60,
        "LONG vs SHORT EFFICIENCY COMPARISON",
        "=" * 60,
        "",
        f"{'Metric':<25} {'LONG-ONLY':>15} {'SHORT-ONLY':>15}",
        "-" * 60,
    ]

    long_trades = long_report.n_trades if long_report else 0
    short_trades = short_report.n_trades if short_report else 0
    lines.append(f"{'Trades':<25} {long_trades:>15} {short_trades:>15}")

    if long_report and short_report:
        lines.append(
            f"{'Win Rate [%]':<25} {long_report.win_rate:>15.1f} {short_report.win_rate:>15.1f}"
        )
        lines.append(
            f"{'Total MFE Available [%]':<25} {long_report.total_mfe_available:>15.2f} "
            f"{short_report.total_mfe_available:>15.2f}"
        )
        lines.append(
            f"{'Total Captured [%]':<25} {long_report.total_captured:>15.2f} "
            f"{short_report.total_captured:>15.2f}"
        )
        lines.append(
            f"{'Aggregate Efficiency [%]':<25} {long_report.aggregate_efficiency:>15.1f} "
            f"{short_report.aggregate_efficiency:>15.1f}"
        )
        lines.append(
            f"{'Mean Efficiency':<25} {long_report.mean_efficiency:>15.3f} "
            f"{short_report.mean_efficiency:>15.3f}"
        )

    lines.extend(["", "=" * 60])
    return "\n".join(lines)
