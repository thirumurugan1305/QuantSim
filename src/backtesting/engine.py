"""
Backtesting engine for QuantSim.

WHAT THIS MODULE DOES
----------------------
`BacktestEngine.run()` walks through historical price data one bar (row)
at a time, in chronological order, and simulates what a long-only trader
following a given signal series would have done: enter a position on
BUY, exit on SELL, do nothing on HOLD or on a signal that doesn't apply
(e.g. a second BUY while already holding).

WHY THE ENGINE NEVER CALLS A STRATEGY DIRECTLY
------------------------------------------------
`BacktestEngine.run()` takes a `signals` Series as a plain input — it
does not import `BaseStrategy` or call `generate_signals()` itself. This
is a deliberate separation: the engine's job is "given these signals,
simulate the trades," full stop. It has no idea whether the signals came
from the Moving Average, RSI, or MACD strategy, or even from a strategy
at all (the tests below feed it hand-built signal series directly). This
means the engine can never accidentally duplicate strategy logic, and any
future strategy works with it automatically, with zero engine changes.

The `run_backtest()` convenience function at the bottom is the ONE place
that bridges the two: it calls `strategy.generate_signals()` once, then
hands the result to the engine. That's the only spot in this whole
module that even knows strategies exist.

EXECUTION MODEL (V1 — documented assumptions)
------------------------------------------------
- Long-only: the engine can be flat (0 shares) or long (positive shares).
  Short selling is not implemented.
- All-in / all-out position sizing: a BUY spends ALL available cash on as
  many whole shares as it can afford; a SELL sells the ENTIRE held
  position. No partial sizing, no fractional shares.
- Execution price = the SAME bar's price the signal was generated from
  (typically Adj Close). This assumes instantaneous execution the moment
  that bar's data becomes final — a standard simplifying assumption for
  an educational simulator. It is NOT look-ahead bias: the engine only
  ever reads bar t's own price to act on bar t's own signal, never a
  later bar. (A "trade on the NEXT bar's open instead" mode would be a
  reasonable future refinement, but is out of scope here.)
- No leverage: a BUY can never spend more cash than is currently held,
  because share count is `cash // execution_price` (floored).
- No transaction costs by default (`transaction_cost_pct=0.0`), but the
  parameter exists and is genuinely applied when non-zero, so adding
  realistic costs later needs no architecture changes — just a
  different number.
- An open position at the end of the data is NOT force-closed. It is
  reported as `open_position` (unrealized), separate from the list of
  actually completed `trades`.
- Repeated signals are handled safely: a BUY while already holding is
  ignored (still holding the same position); a SELL while flat is
  ignored (nothing to sell). Neither raises an error.

CHRONOLOGICAL PROCESSING
---------------------------
The engine processes rows strictly in the order given and requires the
`Date` column to already be sorted ascending — it will raise rather than
silently re-sort, because silently reordering data could mask a real
upstream bug. This matters because portfolio state (cash, shares held) is
inherently sequential: bar 50's cash balance depends on every decision
made at bars 0 through 49. Processing out of order would produce
meaningless results.
"""

from __future__ import annotations

import pandas as pd

from src.backtesting.models import (
    EQUITY_CURVE_COLUMNS,
    BacktestResult,
    OpenPosition,
    PortfolioSnapshot,
    Trade,
)
from src.strategies.base_strategy import BaseStrategy, Signal


class BacktestInputError(ValueError):
    """Raised for invalid backtest configuration or malformed input data.

    Mirrors `MarketDataValidationError`, `IndicatorInputError`, and
    `StrategyInputError` from the earlier layers of this project — each
    layer raises its own clearly-named error.
    """


def _validate_capital(initial_capital: float) -> None:
    if not isinstance(initial_capital, (int, float)) or isinstance(initial_capital, bool):
        raise BacktestInputError(
            f"'initial_capital' must be a number, got {type(initial_capital).__name__}."
        )
    if initial_capital <= 0:
        raise BacktestInputError(f"'initial_capital' must be positive (got {initial_capital}).")


def _validate_transaction_cost(transaction_cost_pct: float) -> None:
    if not isinstance(transaction_cost_pct, (int, float)) or isinstance(transaction_cost_pct, bool):
        raise BacktestInputError(
            f"'transaction_cost_pct' must be a number, got {type(transaction_cost_pct).__name__}."
        )
    if not (0 <= transaction_cost_pct < 1):
        raise BacktestInputError(
            f"'transaction_cost_pct' must be in [0, 1) (got {transaction_cost_pct})."
        )


class BacktestEngine:
    """Simulates a long-only, all-in/all-out trading strategy over
    historical data given a pre-computed signal series.

    Parameters
    ----------
    initial_capital : float
        Starting cash. Must be positive.
    transaction_cost_pct : float, default 0.0
        Fractional cost applied to both buys and sells (e.g. 0.001 = 0.1%
        per trade). Defaults to 0.0 (no transaction costs) for this
        phase, but is fully functional if set — see module docstring.
    """

    def __init__(self, initial_capital: float, transaction_cost_pct: float = 0.0) -> None:
        _validate_capital(initial_capital)
        _validate_transaction_cost(transaction_cost_pct)
        self.initial_capital = float(initial_capital)
        self.transaction_cost_pct = float(transaction_cost_pct)

    def _validate_run_inputs(
        self, price_data: pd.DataFrame, signals: pd.Series, price_column: str
    ) -> None:
        if not isinstance(price_data, pd.DataFrame):
            raise BacktestInputError(
                f"price_data must be a pandas DataFrame, got {type(price_data).__name__}."
            )
        if "Date" not in price_data.columns:
            raise BacktestInputError("price_data is missing the required 'Date' column.")
        if price_column not in price_data.columns:
            raise BacktestInputError(
                f"price_data is missing the required column '{price_column}'. "
                f"Available columns: {list(price_data.columns)}"
            )
        if not isinstance(signals, pd.Series):
            raise BacktestInputError(
                f"signals must be a pandas Series, got {type(signals).__name__}."
            )
        if len(price_data) != len(signals):
            raise BacktestInputError(
                f"price_data and signals must be the same length "
                f"(got {len(price_data)} and {len(signals)})."
            )
        if len(price_data) > 1 and not price_data["Date"].is_monotonic_increasing:
            raise BacktestInputError(
                "price_data must be sorted chronologically (ascending by Date) "
                "before backtesting — the engine processes bars strictly in "
                "order and will not silently re-sort your data."
            )

    def run(
        self,
        price_data: pd.DataFrame,
        signals: pd.Series,
        price_column: str = "Adj Close",
        strategy_name: str = "",
        strategy_params: dict | None = None,
    ) -> BacktestResult:
        """Simulate trades bar-by-bar and return a full BacktestResult.

        `price_data` and `signals` must be the same length. Position
        aligned by row order (both are re-indexed positionally
        internally) — the caller's original objects are never mutated.
        """
        self._validate_run_inputs(price_data, signals, price_column)
        strategy_params = strategy_params or {}

        # Re-index positionally so this loop never depends on whatever
        # index price_data/signals happened to carry in from upstream.
        price_data = price_data.reset_index(drop=True)
        signals = signals.reset_index(drop=True)

        cash = self.initial_capital
        shares_held = 0
        entry_price: float | None = None
        entry_date = None
        entry_signal: Signal | None = None

        trades: list[Trade] = []
        snapshots: list[PortfolioSnapshot] = []

        for i in range(len(price_data)):
            date = price_data.loc[i, "Date"]
            price = float(price_data.loc[i, price_column])
            signal = signals.iloc[i]

            if signal == Signal.BUY and shares_held == 0:
                # All-in: spend every available dollar on whole shares.
                cost_per_share = price * (1 + self.transaction_cost_pct)
                qty = int(cash // cost_per_share) if cost_per_share > 0 else 0
                if qty > 0:
                    cash -= qty * cost_per_share
                    shares_held = qty
                    entry_price = price
                    entry_date = date
                    entry_signal = signal
                # else: not enough cash to buy even 1 share -- stay flat,
                # this is not an error, just a no-op bar.

            elif signal == Signal.SELL and shares_held > 0:
                # All-out: sell the entire position.
                proceeds_per_share = price * (1 - self.transaction_cost_pct)
                cash += shares_held * proceeds_per_share
                trades.append(
                    Trade(
                        entry_date=entry_date,
                        entry_price=entry_price,
                        exit_date=date,
                        exit_price=price,
                        quantity=shares_held,
                        entry_signal=entry_signal,
                        exit_signal=signal,
                    )
                )
                shares_held = 0
                entry_price = None
                entry_date = None
                entry_signal = None

            # BUY while already holding -> ignored (repeated signal; stay
            # in the existing position rather than buying more).
            # SELL while flat -> ignored (nothing to sell).
            # HOLD -> no action, always.

            holdings_value = shares_held * price
            total_value = cash + holdings_value
            snapshots.append(
                PortfolioSnapshot(
                    date=date,
                    price=price,
                    signal=signal,
                    cash=cash,
                    shares_held=shares_held,
                    holdings_value=holdings_value,
                    total_value=total_value,
                )
            )

        if snapshots:
            equity_curve = pd.DataFrame([vars(s) for s in snapshots])
            final_value = snapshots[-1].total_value
        else:
            equity_curve = pd.DataFrame(columns=EQUITY_CURVE_COLUMNS)
            final_value = self.initial_capital

        open_position = None
        if shares_held > 0:
            open_position = OpenPosition(
                entry_date=entry_date,
                entry_price=entry_price,
                quantity=shares_held,
                entry_signal=entry_signal,
            )

        return BacktestResult(
            initial_capital=self.initial_capital,
            final_value=final_value,
            trades=trades,
            equity_curve=equity_curve,
            open_position=open_position,
            strategy_name=strategy_name,
            strategy_params=strategy_params,
        )


def run_backtest(
    price_data: pd.DataFrame,
    strategy: BaseStrategy,
    initial_capital: float,
    transaction_cost_pct: float = 0.0,
) -> BacktestResult:
    """Convenience function: generate signals from `strategy` and run
    the backtest engine on them.

    This is the ONLY place in the backtesting module that calls
    `strategy.generate_signals()` — everything past this point
    (`BacktestEngine.run()`) works purely off the resulting Signal
    series and has no dependency on strategy internals whatsoever.
    """
    signals = strategy.generate_signals(price_data)
    info = strategy.describe()

    engine = BacktestEngine(
        initial_capital=initial_capital,
        transaction_cost_pct=transaction_cost_pct,
    )
    return engine.run(
        price_data=price_data,
        signals=signals,
        price_column=strategy.price_column,
        strategy_name=info["name"],
        strategy_params=info["params"],
    )
