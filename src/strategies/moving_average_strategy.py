"""
Moving Average Crossover strategy.

RULE
-----
- BUY  when the fast SMA crosses ABOVE the slow SMA (short-term trend
  turning up relative to the longer-term trend).
- SELL when the fast SMA crosses BELOW the slow SMA.
- HOLD otherwise.

This reuses `calculate_sma()` from `src/indicators/technical_indicators.py`
for the actual moving-average math (no duplicated calculation here) and
`detect_crossover()` from `base_strategy.py` for the crossover logic
itself, which is identical in shape to the MACD strategy's rule.
"""

from __future__ import annotations

import pandas as pd

from src.indicators.technical_indicators import calculate_sma
from src.strategies.base_strategy import (
    BaseStrategy,
    StrategyInputError,
    _validate_positive_int,
    detect_crossover,
)


class MovingAverageCrossoverStrategy(BaseStrategy):
    """SMA(fast) vs. SMA(slow) crossover strategy.

    Parameters
    ----------
    fast_window : int, default 20
        Window (in bars) for the fast SMA.
    slow_window : int, default 50
        Window (in bars) for the slow SMA. Must be strictly greater than
        `fast_window` — a "fast" average that isn't faster than the
        "slow" one makes the strategy meaningless.
    price_column : str, default "Adj Close"
        Which price column to compute the moving averages on.
    """

    def __init__(
        self,
        fast_window: int = 20,
        slow_window: int = 50,
        price_column: str = "Adj Close",
    ) -> None:
        super().__init__(price_column=price_column)
        _validate_positive_int(fast_window, "fast_window")
        _validate_positive_int(slow_window, "slow_window")
        if fast_window >= slow_window:
            raise StrategyInputError(
                f"fast_window ({fast_window}) must be smaller than "
                f"slow_window ({slow_window})."
            )
        self.fast_window = fast_window
        self.slow_window = slow_window

    def generate_signals(self, price_data: pd.DataFrame) -> pd.Series:
        price = self._extract_price_series(price_data)
        fast_sma = calculate_sma(price, self.fast_window)
        slow_sma = calculate_sma(price, self.slow_window)
        return detect_crossover(fast_sma, slow_sma)
