"""
Data structures for the QuantSim backtesting engine.

WHAT THIS MODULE DOES
----------------------
Pure data — no simulation logic lives here (that's `engine.py`). Every
class below is a `dataclass`: a plain container with named, typed fields
and free auto-generated `__init__`/`__repr__`/`__eq__` methods. Using
dataclasses instead of, say, plain dicts means every field is documented,
typed, and autocompletable, and a typo in a field name fails immediately
instead of silently returning `None` from a dict lookup.

Kept deliberately free of any performance-metric calculations (Sharpe
ratio, max drawdown, win rate, CAGR, ...) — those belong to Phase 6. This
phase only records WHAT happened; Phase 6 will read `BacktestResult`
objects and compute statistics FROM them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.strategies.base_strategy import Signal


@dataclass
class Trade:
    """One COMPLETED round trip: a BUY that was later closed by a SELL.

    An open position that hasn't been sold yet is NOT a Trade — see
    `OpenPosition` below. This distinction matters: Phase 6's win-rate and
    trade-count statistics should only ever look at realized, completed
    trades, not positions still in flight.
    """

    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp
    exit_price: float
    quantity: int
    entry_signal: Signal
    exit_signal: Signal

    @property
    def pnl(self) -> float:
        """Dollar profit/loss for this trade (before any transaction costs
        already reflected in entry_price/exit_price — see engine.py)."""
        return (self.exit_price - self.entry_price) * self.quantity

    @property
    def pnl_pct(self) -> float:
        """Percentage return for this trade, relative to entry price."""
        if self.entry_price == 0:
            return 0.0
        return (self.exit_price / self.entry_price - 1) * 100

    @property
    def is_win(self) -> bool:
        return self.pnl > 0


@dataclass
class OpenPosition:
    """A position that was BOUGHT but not yet SOLD by the end of the
    backtest — i.e. it's still "open" when the historical data runs out.

    Kept separate from `Trade` because it has no exit price/date yet; its
    dollar value is unrealized (it would change if the backtest continued
    with more data).
    """

    entry_date: pd.Timestamp
    entry_price: float
    quantity: int
    entry_signal: Signal


@dataclass
class PortfolioSnapshot:
    """The full portfolio state at ONE bar (one row of historical data).

    One of these is recorded for every single bar processed, in
    chronological order — stringing them together is the "equity curve"
    (portfolio value over time) that Phase 6 will compute return-based
    statistics from, and that the UI charts directly.
    """

    date: pd.Timestamp
    price: float
    signal: Signal
    cash: float
    shares_held: int
    holdings_value: float  # shares_held * price
    total_value: float  # cash + holdings_value


# Columns used for an EMPTY equity curve, so `BacktestResult.equity_curve`
# always has a consistent, predictable shape (same column names) even
# when there's no data to process — callers never need to special-case
# "did this come from zero rows or many rows?".
EQUITY_CURVE_COLUMNS = [
    "date",
    "price",
    "signal",
    "cash",
    "shares_held",
    "holdings_value",
    "total_value",
]


@dataclass
class BacktestResult:
    """Everything a completed backtest run produced.

    This is intentionally a "dumb" container: it reports what happened
    (starting/ending values, every completed trade, the bar-by-bar
    equity curve, and any still-open position) without judging whether
    that was good or bad — no Sharpe ratio, drawdown, or win rate here.
    That interpretation is Phase 6's job, reading these exact fields.
    """

    initial_capital: float
    final_value: float
    trades: list[Trade]
    equity_curve: pd.DataFrame
    open_position: OpenPosition | None
    strategy_name: str
    strategy_params: dict
    warnings: list[str] = field(default_factory=list)

    @property
    def num_trades(self) -> int:
        return len(self.trades)
