"""
Performance analytics for QuantSim.

WHAT THIS MODULE DOES
----------------------
Reads an already-completed `BacktestResult` (Phase 5's output) and
computes descriptive statistics from it: total return, win rate, max
drawdown, Sharpe ratio, and a few others. Nothing here re-runs a
strategy, re-simulates trades, or touches `BacktestEngine` in any way —
this is a pure, read-only consumer of `result.trades` and
`result.equity_curve`. That separation matters for the same reason it
mattered in every earlier phase: `calculate_performance_metrics()` can
be called seconds or days after a backtest ran, on any BacktestResult
from anywhere (including hand-built ones in tests), with zero coupling
to how that result was produced.

NEVER RETURNS inf OR NaN
----------------------------
Every metric below that could mathematically produce `inf`, `-inf`, or
`NaN` (for example, dividing by a standard deviation of zero) is
explicitly guarded and returns `None` instead. `None` means "not
meaningful for this data," which is a different and more honest signal
than a number that merely looks valid but isn't.

TRADES VS. OPEN POSITION
----------------------------
Every trade-based statistic (win rate, average win, average loss, trade
count) only ever looks at `result.trades` — completed round trips.
`result.open_position` (a position still held when the data ran out) is
NEVER counted as a trade here. Its unrealized value is already folded
into `result.final_value` by the backtesting engine, so total return and
absolute P&L correctly reflect it — but `has_open_position` is exposed
explicitly so a caller (like the UI) can flag that the return includes
something not yet "cashed in."
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from src.backtesting.models import BacktestResult, Trade
from src.config import TRADING_DAYS_PER_YEAR


@dataclass
class PerformanceMetrics:
    """Every statistic this module computes from one BacktestResult.

    Fields that can be legitimately undefined for some inputs (e.g. no
    completed trades, or too little data to form a return series) are
    typed `float | None` — `None` always means "not applicable," never
    "zero" and never a crash.
    """

    total_return_pct: float
    absolute_pnl: float
    num_trades: int
    win_rate_pct: float | None
    average_win: float | None
    average_loss: float | None
    max_drawdown_pct: float | None
    sharpe_ratio: float | None
    annualized_volatility_pct: float | None
    best_period_return_pct: float | None
    worst_period_return_pct: float | None
    has_open_position: bool
    risk_free_rate_annual: float


# ---------------------------------------------------------------------------
# Trade-based statistics (use result.trades ONLY -- never open_position)
# ---------------------------------------------------------------------------

def _win_rate_pct(trades: list[Trade]) -> float | None:
    if not trades:
        return None
    wins = sum(1 for t in trades if t.is_win)
    return wins / len(trades) * 100


def _average_win(trades: list[Trade]) -> float | None:
    winning_pnls = [t.pnl for t in trades if t.is_win]
    if not winning_pnls:
        return None
    return sum(winning_pnls) / len(winning_pnls)


def _average_loss(trades: list[Trade]) -> float | None:
    # "Losing/non-winning" matches Trade.is_win's own definition (pnl > 0
    # is a win), so this bucket is every trade where is_win is False --
    # i.e. pnl <= 0, including an exact break-even trade.
    losing_pnls = [t.pnl for t in trades if not t.is_win]
    if not losing_pnls:
        return None
    return sum(losing_pnls) / len(losing_pnls)


# ---------------------------------------------------------------------------
# Equity-curve-based statistics
# ---------------------------------------------------------------------------

def _daily_returns(equity_curve: pd.DataFrame) -> pd.Series | None:
    """Day-over-day percentage change in total portfolio value.

    Returns None (not an empty Series) if there are fewer than 2 rows --
    a single point has no "change" to measure, so no return series can
    be formed at all. Every metric below that needs a sequence of
    returns (Sharpe, volatility, drawdown, best/worst period) is built
    on top of this one shared calculation, so there's exactly one place
    that decides what a "return" means in this module.
    """
    if equity_curve is None or len(equity_curve) < 2:
        return None
    returns = equity_curve["total_value"].pct_change().dropna()
    if returns.empty:
        return None
    return returns


def _max_drawdown_pct(equity_curve: pd.DataFrame) -> float | None:
    """Largest peak-to-trough decline in portfolio value, as a NEGATIVE
    percentage (e.g. -38.46 means a 38.46% decline from a prior peak).

    Method: track the running maximum ("the highest the portfolio has
    EVER been up to this point"), then measure how far below that peak
    the portfolio fell at each point. The most negative of those values,
    across the whole curve, is the maximum drawdown.

    Requires at least 2 rows for the same reason `_daily_returns` does:
    with only one data point, "decline from a peak" isn't a meaningful
    question yet.
    """
    if equity_curve is None or len(equity_curve) < 2:
        return None

    values = equity_curve["total_value"].astype(float)
    running_max = values.cummax()

    # Defensive rather than assumed: only divide where the running peak
    # is actually positive. In this long-only, no-leverage engine with a
    # required-positive initial_capital, total_value should never be
    # <=0 in practice, but a metrics function should not silently trust
    # that invariant from a different module.
    valid = running_max > 0
    if not valid.any():
        return None

    drawdown = (values[valid] - running_max[valid]) / running_max[valid]
    return float(drawdown.min() * 100)


def _sharpe_ratio(daily_returns: pd.Series | None, risk_free_rate_annual: float) -> float | None:
    """Risk-adjusted return: how much return per unit of volatility,
    relative to a risk-free baseline.

    ASSUMPTION -- stated explicitly, not hidden:
    `risk_free_rate_annual` is an ANNUAL rate (e.g. 0.05 = 5%/year). It is
    converted to a daily rate by simple division:
        daily_risk_free = risk_free_rate_annual / TRADING_DAYS_PER_YEAR
    This is a standard simplifying approximation (not compounding /
    geometric conversion) that is accurate enough for typical near-zero
    to moderate rates over the return horizons this simulator deals
    with. It is applied consistently -- the mean of DAILY returns is
    compared against a DAILY risk-free rate, never mixing an annual
    figure with a daily one.

    Returns None if there are fewer than 2 daily returns, or if the
    standard deviation of returns is exactly zero (a flat or otherwise
    non-varying equity curve) -- dividing by a zero standard deviation
    is mathematically undefined, so this explicitly returns None rather
    than producing `inf` or `NaN`.
    """
    if daily_returns is None or daily_returns.empty:
        return None

    std = daily_returns.std()
    if std == 0 or pd.isna(std):
        return None

    daily_risk_free = risk_free_rate_annual / TRADING_DAYS_PER_YEAR
    sharpe = (daily_returns.mean() - daily_risk_free) / std * math.sqrt(TRADING_DAYS_PER_YEAR)
    return float(sharpe)


def _annualized_volatility_pct(daily_returns: pd.Series | None) -> float | None:
    """Annualized standard deviation of daily returns, as a percentage.

    Unlike Sharpe ratio, this does NOT divide by the standard deviation
    -- it simply reports it (scaled to an annual figure). So a perfectly
    flat equity curve legitimately has 0.0% volatility here, which is a
    meaningful, correct answer -- it is NOT the same situation as
    Sharpe's division-by-zero problem, and is deliberately not treated
    the same way.
    """
    if daily_returns is None or daily_returns.empty:
        return None
    std = daily_returns.std()
    if pd.isna(std):
        return None
    return float(std * math.sqrt(TRADING_DAYS_PER_YEAR) * 100)


def _best_worst_period_return_pct(
    daily_returns: pd.Series | None,
) -> tuple[float | None, float | None]:
    """The single best and single worst day-over-day return, as
    percentages. Both None together if there's no return series."""
    if daily_returns is None or daily_returns.empty:
        return None, None
    return float(daily_returns.max() * 100), float(daily_returns.min() * 100)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def calculate_performance_metrics(
    result: BacktestResult,
    risk_free_rate: float = 0.0,
) -> PerformanceMetrics:
    """Compute every Phase 6 statistic from a completed BacktestResult.

    Parameters
    ----------
    result : BacktestResult
        The output of `BacktestEngine.run()` (or `run_backtest()`).
        Never mutated, never re-simulated -- only read.
    risk_free_rate : float, default 0.0
        ANNUAL risk-free rate used for the Sharpe ratio (e.g. 0.05 for
        5%/year). Defaults to 0.0 because this is an educational
        simulator with no live connection to actual treasury rates --
        hardcoding any other specific value would be a stale, arbitrary
        guess. See `_sharpe_ratio()` for exactly how this is converted
        to a daily rate.

    Returns
    -------
    PerformanceMetrics
        Always a complete object. Fields that aren't meaningful for the
        given data (e.g. no completed trades, or too little equity-curve
        history) are `None`, never `NaN`/`inf`, and this function never
        raises for the documented edge cases (zero trades, all
        winning/losing trades, a flat or empty/single-row equity curve,
        or an open position at the end).
    """
    total_return_pct = (
        (result.final_value / result.initial_capital - 1) * 100
        if result.initial_capital
        else 0.0
    )
    absolute_pnl = result.final_value - result.initial_capital

    daily_returns = _daily_returns(result.equity_curve)
    best, worst = _best_worst_period_return_pct(daily_returns)

    return PerformanceMetrics(
        total_return_pct=total_return_pct,
        absolute_pnl=absolute_pnl,
        num_trades=result.num_trades,
        win_rate_pct=_win_rate_pct(result.trades),
        average_win=_average_win(result.trades),
        average_loss=_average_loss(result.trades),
        max_drawdown_pct=_max_drawdown_pct(result.equity_curve),
        sharpe_ratio=_sharpe_ratio(daily_returns, risk_free_rate),
        annualized_volatility_pct=_annualized_volatility_pct(daily_returns),
        best_period_return_pct=best,
        worst_period_return_pct=worst,
        has_open_position=result.open_position is not None,
        risk_free_rate_annual=risk_free_rate,
    )
