"""
MACD (Moving Average Convergence Divergence) strategy.

RULE
-----
- BUY  when the MACD line crosses ABOVE the signal line.
- SELL when the MACD line crosses BELOW the signal line.
- HOLD otherwise.

This reuses `calculate_macd()` from
`src/indicators/technical_indicators.py` for the MACD math (macd_line,
signal_line, histogram) and `detect_crossover()` from `base_strategy.py`
for the crossover logic — the exact same "did line A cross line B?"
question the Moving Average strategy asks, just applied to a different
pair of lines.
"""

from __future__ import annotations

import pandas as pd

from src.indicators.technical_indicators import calculate_macd
from src.strategies.base_strategy import (
    BaseStrategy,
    StrategyInputError,
    _validate_positive_int,
    detect_crossover,
)


class MACDStrategy(BaseStrategy):
    """MACD line vs. signal line crossover strategy.

    Parameters
    ----------
    fast : int, default 12
    slow : int, default 26
    signal : int, default 9
        Standard MACD parameters — see `calculate_macd()` for what each
        one means. `fast` must be strictly less than `slow`.
    price_column : str, default "Adj Close"
    """

    def __init__(
        self,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        price_column: str = "Adj Close",
    ) -> None:
        super().__init__(price_column=price_column)
        _validate_positive_int(fast, "fast")
        _validate_positive_int(slow, "slow")
        _validate_positive_int(signal, "signal")
        if fast >= slow:
            raise StrategyInputError(f"fast ({fast}) must be smaller than slow ({slow}).")

        self.fast = fast
        self.slow = slow
        self.signal = signal

    def generate_signals(self, price_data: pd.DataFrame) -> pd.Series:
        price = self._extract_price_series(price_data)
        macd_df = calculate_macd(price, fast=self.fast, slow=self.slow, signal=self.signal)
        return detect_crossover(macd_df["macd_line"], macd_df["signal_line"])
