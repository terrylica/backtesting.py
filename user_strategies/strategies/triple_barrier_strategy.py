"""Triple-barrier probabilistic strategy for backtesting.py.

ADR: 2025-12-20-clickhouse-triple-barrier-backtest

Implements a backtesting.py Strategy that uses pre-computed probability
predictions from a triple-barrier classifier. Entry signals are generated
when P(upper) exceeds an optimizable threshold.

This strategy bridges the ML classification system with backtesting.py's
event-driven simulation framework.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from backtesting import Strategy
from backtesting.lib import crossover

if TYPE_CHECKING:
    from backtesting._typing import Array


class TripleBarrierProbStrategy(Strategy):
    """Strategy based on triple-barrier probability predictions.

    Uses pre-computed P(upper) predictions to generate entry signals.
    Long positions are entered when P(upper) > threshold.
    Short positions are entered when P(lower) > threshold (optional).

    Parameters (optimizable via bt.optimize()):
        entry_threshold: Minimum P(upper) for long entry (default: 0.5)
        exit_threshold: P(upper) below which to exit longs (default: 0.4)
        use_shorts: Enable short positions when P(lower) > threshold
        position_size: Fraction of equity per trade (default: 1.0)
    """

    # Optimizable parameters
    entry_threshold = 0.55
    exit_threshold = 0.45
    use_shorts = False
    position_size = 1.0

    def init(self) -> None:
        """Initialize strategy indicators.

        Expects the data to have 'prob_upper' column with P(Y=+1|X) predictions.
        Optionally 'prob_lower' for short signals.
        """
        # Validate required columns
        if not hasattr(self.data, "prob_upper"):
            msg = (
                "Data must have 'prob_upper' column with probability predictions. "
                "Use TripleBarrierClassifier.predict_proba_upper() to generate."
            )
            raise ValueError(msg)

        # Create indicator wrappers for plotting
        self.prob_upper = self.I(lambda: self.data.prob_upper, name="P(upper)")

        if hasattr(self.data, "prob_lower"):
            self.prob_lower = self.I(lambda: self.data.prob_lower, name="P(lower)")
        else:
            self.prob_lower = None

        if hasattr(self.data, "prob_timeout"):
            self.prob_timeout = self.I(
                lambda: self.data.prob_timeout, name="P(timeout)"
            )

    def next(self) -> None:
        """Execute trading logic for current bar."""
        prob_up = self.prob_upper[-1]

        # Handle NaN probabilities (e.g., before model warmup)
        if np.isnan(prob_up):
            return

        # Long entry logic
        if not self.position:
            if prob_up > self.entry_threshold:
                self.buy(size=self.position_size)
            elif self.use_shorts and self.prob_lower is not None:
                prob_down = self.prob_lower[-1]
                if not np.isnan(prob_down) and prob_down > self.entry_threshold:
                    self.sell(size=self.position_size)

        # Exit logic
        elif self.position.is_long:
            if prob_up < self.exit_threshold:
                self.position.close()
        elif self.position.is_short and self.prob_lower is not None:
            prob_down = self.prob_lower[-1]
            if np.isnan(prob_down) or prob_down < self.exit_threshold:
                self.position.close()


class TripleBarrierDirectionalStrategy(Strategy):
    """Strategy using directional probability difference.

    Uses P(upper) - P(lower) as a directional signal:
    - Positive values indicate bullish bias
    - Negative values indicate bearish bias

    This approach may be more robust than absolute thresholds.

    Parameters:
        direction_threshold: Minimum |P(up) - P(down)| for entry
        neutral_zone: Range around 0 where no action is taken
    """

    direction_threshold = 0.15
    neutral_zone = 0.05
    position_size = 1.0

    def init(self) -> None:
        """Initialize directional signal."""
        if not hasattr(self.data, "prob_upper") or not hasattr(self.data, "prob_lower"):
            msg = "Data must have 'prob_upper' and 'prob_lower' columns"
            raise ValueError(msg)

        # Directional signal: P(up) - P(down)
        self.direction = self.I(
            lambda: self.data.prob_upper - self.data.prob_lower,
            name="Direction",
        )

    def next(self) -> None:
        """Execute directional trading logic."""
        direction = self.direction[-1]

        if np.isnan(direction):
            return

        if not self.position:
            # Enter long if strongly bullish
            if direction > self.direction_threshold:
                self.buy(size=self.position_size)
            # Enter short if strongly bearish
            elif direction < -self.direction_threshold:
                self.sell(size=self.position_size)
        else:
            # Exit if direction reverses into neutral zone
            if self.position.is_long and direction < self.neutral_zone:
                self.position.close()
            elif self.position.is_short and direction > -self.neutral_zone:
                self.position.close()


def prepare_data_for_strategy(
    ohlcv: pd.DataFrame,
    prob_upper: np.ndarray | pd.Series,
    prob_lower: np.ndarray | pd.Series | None = None,
    prob_timeout: np.ndarray | pd.Series | None = None,
) -> pd.DataFrame:
    """Prepare data DataFrame for triple-barrier strategy.

    Combines OHLCV data with probability predictions into a format
    suitable for backtesting.py.

    Args:
        ohlcv: DataFrame with Open, High, Low, Close, Volume columns
        prob_upper: P(Y=+1|X) predictions aligned with ohlcv index
        prob_lower: Optional P(Y=-1|X) predictions
        prob_timeout: Optional P(Y=0|X) predictions

    Returns:
        DataFrame ready for Backtest(data, TripleBarrierProbStrategy)
    """
    # Validate OHLCV columns
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = set(required) - set(ohlcv.columns)
    if missing:
        msg = f"Missing required columns: {missing}"
        raise ValueError(msg)

    # Create copy with probability columns
    data = ohlcv.copy()
    data["prob_upper"] = prob_upper

    if prob_lower is not None:
        data["prob_lower"] = prob_lower

    if prob_timeout is not None:
        data["prob_timeout"] = prob_timeout

    return data


def run_backtest_with_optimization(
    data: pd.DataFrame,
    strategy_class: type[Strategy] = TripleBarrierProbStrategy,
    cash: float = 10_000_000,
    commission: float = 0.0002,
    margin: float = 1.0,
    optimize_params: dict | None = None,
    constraint: callable | None = None,
) -> tuple:
    """Run backtest with optional parameter optimization.

    Args:
        data: DataFrame with OHLCV and probability columns
        strategy_class: Strategy class to use
        cash: Starting cash
        commission: Commission per trade (fraction)
        margin: Margin requirement
        optimize_params: Dict of parameter ranges for optimization
        constraint: Optional constraint function for optimization

    Returns:
        Tuple of (stats, backtest_instance)
    """
    from backtesting import Backtest

    bt = Backtest(
        data,
        strategy_class,
        cash=cash,
        commission=commission,
        margin=margin,
        exclusive_orders=True,
    )

    if optimize_params:
        stats = bt.optimize(
            **optimize_params,
            maximize="Sharpe Ratio",
            constraint=constraint,
            return_heatmap=False,
        )
    else:
        stats = bt.run()

    return stats, bt
