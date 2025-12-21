"""Triple-barrier hyperparameter configuration.

ADR: 2025-12-20-clickhouse-triple-barrier-backtest

This module defines the hyperparameter grid for triple-barrier labeling:
- Barrier percentages (b): Distance to upper/lower barriers
- Horizon bars (H): Maximum holding period before timeout

These parameters are optimized via purged cross-validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class BarrierConfig:
    """Configuration for triple-barrier label generation.

    Attributes:
        barrier_pct: Percentage distance to upper/lower barriers (e.g., 0.01 = 1%)
        horizon_bars: Maximum bars before timeout (label = 0)
        symmetric: If True, upper and lower barriers are symmetric
    """

    barrier_pct: float
    horizon_bars: int
    symmetric: bool = True

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.barrier_pct <= 0:
            msg = f"barrier_pct must be positive, got {self.barrier_pct}"
            raise ValueError(msg)
        if self.horizon_bars <= 0:
            msg = f"horizon_bars must be positive, got {self.horizon_bars}"
            raise ValueError(msg)


# =============================================================================
# Hyperparameter Grid for Cross-Validation Optimization
# =============================================================================

# Barrier percentages to search
# Range: 0.5% to 2.5% (designed for hourly crypto data)
BARRIER_PCT_GRID: Sequence[float] = (
    0.005,  # 0.5% - tight barrier, more signals
    0.010,  # 1.0% - moderate barrier
    0.015,  # 1.5% - balanced
    0.020,  # 2.0% - wider barrier
    0.025,  # 2.5% - loose barrier, fewer but clearer signals
)

# Horizon bars to search (at hourly granularity)
# Range: 12 to 96 hours (0.5 to 4 days)
HORIZON_BARS_GRID: Sequence[int] = (
    12,  # 12 hours - short-term
    24,  # 24 hours - 1 day
    48,  # 48 hours - 2 days
    72,  # 72 hours - 3 days
    96,  # 96 hours - 4 days
)


def generate_config_grid() -> list[BarrierConfig]:
    """Generate all combinations of barrier configurations.

    Returns:
        List of BarrierConfig instances for grid search
    """
    configs = []
    for b in BARRIER_PCT_GRID:
        for h in HORIZON_BARS_GRID:
            configs.append(BarrierConfig(barrier_pct=b, horizon_bars=h))
    return configs


# =============================================================================
# Preset Configurations
# =============================================================================

# Conservative: Wide barriers, long horizon (fewer but cleaner signals)
CONSERVATIVE_CONFIG = BarrierConfig(barrier_pct=0.02, horizon_bars=72)

# Aggressive: Tight barriers, short horizon (more signals, higher noise)
AGGRESSIVE_CONFIG = BarrierConfig(barrier_pct=0.005, horizon_bars=24)

# Balanced: Middle ground
BALANCED_CONFIG = BarrierConfig(barrier_pct=0.01, horizon_bars=48)

# Default configuration for quick testing
DEFAULT_CONFIG = BALANCED_CONFIG


# =============================================================================
# Timeframe-Specific Presets
# =============================================================================

def get_config_for_timeframe(timeframe: str) -> BarrierConfig:
    """Get recommended barrier config for a given timeframe.

    Args:
        timeframe: CCXT-style timeframe string (e.g., '1m', '5m', '1h', '4h', '1d')

    Returns:
        BarrierConfig with recommended parameters for the timeframe

    Raises:
        ValueError: If timeframe not recognized
    """
    # Parse timeframe to minutes
    unit = timeframe[-1]
    value = int(timeframe[:-1])

    if unit == "s":
        minutes = value / 60
    elif unit == "m":
        minutes = value
    elif unit == "h":
        minutes = value * 60
    elif unit == "d":
        minutes = value * 60 * 24
    else:
        msg = f"Unrecognized timeframe unit: {unit}"
        raise ValueError(msg)

    # Scale parameters based on timeframe
    # Higher frequency = tighter barriers, shorter horizons
    if minutes <= 5:
        # 1-5 minute: Very tight barriers, short horizon
        return BarrierConfig(barrier_pct=0.003, horizon_bars=60)
    elif minutes <= 15:
        # 5-15 minute: Tight barriers
        return BarrierConfig(barrier_pct=0.005, horizon_bars=48)
    elif minutes <= 60:
        # 15-60 minute: Moderate barriers
        return BarrierConfig(barrier_pct=0.01, horizon_bars=36)
    elif minutes <= 240:
        # 1-4 hour: Standard barriers
        return BarrierConfig(barrier_pct=0.015, horizon_bars=24)
    else:
        # Daily+: Wide barriers, long horizon
        return BarrierConfig(barrier_pct=0.025, horizon_bars=14)
