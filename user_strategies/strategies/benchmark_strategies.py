"""
Benchmark Strategies for Performance Comparison

Implementation of canonical benchmark strategies for accurate performance assessment.
Based on CLAUDE.md Benchmark Comparison Standards - EMPIRICALLY VALIDATED.
"""

from backtesting import Strategy


class BuyAndHoldBenchmark(Strategy):
    """
    Buy-and-Hold Benchmark for LONG strategies

    Canonical benchmark implementation that buys at the beginning
    and holds until the end. Used as baseline for LONG strategy comparison.
    """

    def init(self):
        """Initialize - no indicators needed for buy-and-hold"""
        pass

    def next(self):
        """Execute buy-and-hold strategy"""
        if not self.position:
            self.buy()


class ShortAndHoldBenchmark(Strategy):
    """
    Short-and-Hold Benchmark for SHORT strategies

    Canonical benchmark implementation that shorts at the beginning
    and holds until the end. Used as baseline for SHORT strategy comparison.
    """

    def init(self):
        """Initialize - no indicators needed for short-and-hold"""
        pass

    def next(self):
        """Execute short-and-hold strategy"""
        if not self.position:
            self.sell()