"""
RSI (Relative Strength Index) strategy.

RULE (level-based, not crossover-based — see note below)
------------------------------------------------------------
- BUY  on every bar where RSI is BELOW the oversold threshold (default 30).
- SELL on every bar where RSI is ABOVE the overbought threshold (default 70).
- HOLD otherwise (including any bar where RSI itself is still NaN, i.e.
  during its warm-up period).

DESIGN NOTE — why level-based instead of "crosses the threshold"?
-----------------------------------------------------------------
This is a genuine design choice worth being explicit about, since RSI
strategies are commonly written either way:
  - Level-based (what's implemented here): a signal fires on EVERY bar
    the condition holds, which can mean several BUY signals in a row
    while RSI stays low. This is the more common, simpler textbook
    definition of an RSI strategy.
  - Crossing-based (an alternative not implemented here): a signal would
    fire only once, the moment RSI crosses back above/below the
    threshold, similar to the Moving Average and MACD strategies.

Deciding what to DO about repeated signals (e.g. "don't buy again if
already holding a position") is deliberately left to the backtesting
engine (a later phase), not this strategy — this layer's only job is to
report what the indicator's rule says on each individual bar.

This reuses `calculate_rsi()` from
`src/indicators/technical_indicators.py` for the RSI math itself.
"""

from __future__ import annotations

import pandas as pd

from src.indicators.technical_indicators import calculate_rsi
from src.strategies.base_strategy import (
    BaseStrategy,
    Signal,
    StrategyInputError,
    _validate_positive_int,
)


class RSIStrategy(BaseStrategy):
    """RSI level-threshold strategy.

    Parameters
    ----------
    period : int, default 14
        RSI lookback period.
    oversold : float, default 30.0
        RSI below this level triggers BUY.
    overbought : float, default 70.0
        RSI above this level triggers SELL.
    price_column : str, default "Adj Close"
    """

    def __init__(
        self,
        period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        price_column: str = "Adj Close",
    ) -> None:
        super().__init__(price_column=price_column)
        _validate_positive_int(period, "period")

        if not isinstance(oversold, (int, float)) or isinstance(oversold, bool):
            raise StrategyInputError(f"'oversold' must be a number, got {type(oversold).__name__}.")
        if not isinstance(overbought, (int, float)) or isinstance(overbought, bool):
            raise StrategyInputError(f"'overbought' must be a number, got {type(overbought).__name__}.")
        if not (0 < oversold < overbought < 100):
            raise StrategyInputError(
                "Thresholds must satisfy 0 < oversold < overbought < 100 "
                f"(got oversold={oversold}, overbought={overbought})."
            )

        self.period = period
        self.oversold = float(oversold)
        self.overbought = float(overbought)

    def generate_signals(self, price_data: pd.DataFrame) -> pd.Series:
        price = self._extract_price_series(price_data)
        rsi = calculate_rsi(price, period=self.period)

        signals = pd.Series([Signal.HOLD] * len(rsi), index=rsi.index, name="signal")
        valid = rsi.notna()
        signals[valid & (rsi < self.oversold)] = Signal.BUY
        signals[valid & (rsi > self.overbought)] = Signal.SELL
        return signals
