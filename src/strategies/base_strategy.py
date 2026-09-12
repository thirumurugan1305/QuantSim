"""
Strategy abstraction for QuantSim.

WHAT THIS MODULE DOES
----------------------
Defines the common interface every trading strategy follows, plus one
piece of shared logic (`detect_crossover`) that more than one strategy
needs. Concrete strategies (moving_average_strategy.py, rsi_strategy.py,
macd_strategy.py) subclass `BaseStrategy` and implement only the part
that's actually unique to them.

WHY AN ABSTRACT BASE CLASS?
------------------------------
Every strategy needs to do the same three things: take price data in,
validate it, and hand back a standardized signal series. Defining that
contract once here means:
  - Adding a new strategy later means writing ONE new file that
    implements `generate_signals()` — nothing else needs to change.
  - Any code that calls a strategy (the UI today; a strategy-comparison
    tool or the backtesting engine later) can treat every strategy the
    same way, without caring which one it is.

SIGNAL REPRESENTATION
------------------------
Signals are represented with the `Signal` enum (BUY / SELL / HOLD)
defined below, not plain strings. Inheriting from `str` as well as `Enum`
means a `Signal` still behaves like a string (compares equal to "BUY",
prints cleanly, stores fine in a DataFrame or later a database column),
while still giving us the type safety of an enum: a typo like `"BYU"`
would just be a wrong string, but `Signal.BYU` doesn't exist and would
fail immediately and loudly.

NO LOOK-AHEAD BIAS
--------------------
`detect_crossover()` below only ever compares "now" (row t) to "one bar
ago" (row t-1) — never a future row. Every concrete strategy in this
project builds its signals either from `detect_crossover()` or from a
same-row indicator comparison (RSI vs. a fixed threshold), so no strategy
here can accidentally use information that wouldn't have been available
at the time.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

import pandas as pd


class Signal(str, Enum):
    """The three standardized trading signals every strategy emits."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

    def __str__(self) -> str:  # clean display in tables/UI instead of "Signal.BUY"
        return self.value


class StrategyInputError(ValueError):
    """Raised for invalid strategy configuration or malformed input data.

    Mirrors `MarketDataValidationError` (data layer) and
    `IndicatorInputError` (indicator layer) — each layer of this project
    raises its own clearly-named error, so calling code always knows
    which layer a problem came from.
    """


def _validate_positive_int(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise StrategyInputError(f"'{name}' must be an integer, got {type(value).__name__}.")
    if value < 1:
        raise StrategyInputError(f"'{name}' must be a positive integer (got {value}).")


def detect_crossover(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """Shared crossover-detection logic, used by both the Moving Average
    strategy (fast SMA vs. slow SMA) and the MACD strategy (MACD line vs.
    signal line) — "did line A cross line B?" is the same question in
    both cases, so the logic lives here once instead of twice.

    BUY  is emitted the bar `fast` crosses ABOVE `slow`
         (fast[t] > slow[t]  AND  fast[t-1] <= slow[t-1])
    SELL is emitted the bar `fast` crosses BELOW `slow`
         (fast[t] < slow[t]  AND  fast[t-1] >= slow[t-1])
    HOLD otherwise — including every row where either line's current or
    previous value is NaN (e.g. during an indicator's warm-up period),
    since a crossover can't be meaningfully judged without both points.

    Only ever compares row t to row t-1, so this cannot use future data.
    """
    if len(fast) != len(slow):
        raise StrategyInputError(
            f"'fast' and 'slow' series must be the same length "
            f"(got {len(fast)} and {len(slow)})."
        )

    fast_prev = fast.shift(1)
    slow_prev = slow.shift(1)

    crossed_above = (fast > slow) & (fast_prev <= slow_prev)
    crossed_below = (fast < slow) & (fast_prev >= slow_prev)

    # Belt-and-suspenders: NaN comparisons already evaluate to False in
    # pandas (so a warm-up-period row can't accidentally register as a
    # crossover), but we mask explicitly rather than relying on that
    # implicit behavior, since "explicit is better than implicit" matters
    # even more in a project meant for learning from.
    valid = fast.notna() & slow.notna() & fast_prev.notna() & slow_prev.notna()

    signals = pd.Series([Signal.HOLD] * len(fast), index=fast.index, name="signal")
    signals[crossed_above & valid] = Signal.BUY
    signals[crossed_below & valid] = Signal.SELL
    return signals


class BaseStrategy(ABC):
    """Common interface every QuantSim strategy implements.

    Subclasses must implement `generate_signals()`. Everything else here
    (parameter storage via `vars(self)`, price-column extraction, input
    validation) is shared so concrete strategies stay short and only
    contain what's actually specific to them.
    """

    def __init__(self, price_column: str = "Adj Close") -> None:
        self.price_column = price_column

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def describe(self) -> dict:
        """Strategy name + current parameter values, for display in the UI.

        Uses `vars(self)` so this works automatically for every subclass
        without each one having to implement its own describe() — any
        attribute a subclass sets in __init__ (fast_window, period,
        oversold, ...) shows up here for free.
        """
        return {"name": self.name, "params": dict(vars(self))}

    def _extract_price_series(self, price_data: pd.DataFrame) -> pd.Series:
        """Validate `price_data` and pull out the configured price column.

        Raises StrategyInputError (not a generic KeyError/TypeError) so
        calling code — including the UI — can show a clear message.
        """
        if not isinstance(price_data, pd.DataFrame):
            raise StrategyInputError(
                f"price_data must be a pandas DataFrame, got {type(price_data).__name__}."
            )
        if self.price_column not in price_data.columns:
            raise StrategyInputError(
                f"price_data is missing the required column '{self.price_column}'. "
                f"Available columns: {list(price_data.columns)}"
            )
        return price_data[self.price_column]

    @abstractmethod
    def generate_signals(self, price_data: pd.DataFrame) -> pd.Series:
        """Compute a Signal for every row of `price_data`.

        Must return a `pandas.Series` of `Signal` values, the same length
        and index as `price_data`. Implementations must only use each
        row's own data and *earlier* rows — never later ones (see module
        docstring on look-ahead bias).
        """
        raise NotImplementedError
