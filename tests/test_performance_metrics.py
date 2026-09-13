"""
Tests for src/analytics/performance_metrics.py

All tests use hand-built `BacktestResult` objects (never a real backtest
run) so every expected number can be independently verified -- several
tests deliberately compute their expected values using Python's stdlib
`statistics` module rather than pandas, as an independent check against
the production code's pandas-based implementation.

Run with:
    pytest tests/test_performance_metrics.py -v
"""

import math
import statistics

import pandas as pd
import pytest

from src.analytics.performance_metrics import calculate_performance_metrics
from src.backtesting.models import BacktestResult, OpenPosition, Trade
from src.config import TRADING_DAYS_PER_YEAR


# ---------------------------------------------------------------------------
# Helpers to build hand-crafted BacktestResult objects
# ---------------------------------------------------------------------------

def make_equity_curve(total_values: list[float]) -> pd.DataFrame:
    """A minimal equity curve DataFrame. Only the 'total_value' column is
    actually read by the analytics module -- the other columns are
    filled with plausible placeholder values purely so the DataFrame's
    shape matches what BacktestEngine really produces.
    """
    n = len(total_values)
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="D") if n else pd.Series([], dtype="datetime64[ns]"),
            "price": total_values,
            "signal": ["HOLD"] * n,
            "cash": [0.0] * n,
            "shares_held": [0] * n,
            "holdings_value": [0.0] * n,
            "total_value": total_values,
        }
    )


def make_trade(entry_price: float, exit_price: float, quantity: int = 100) -> Trade:
    return Trade(
        entry_date=pd.Timestamp("2024-01-01"),
        entry_price=entry_price,
        exit_date=pd.Timestamp("2024-01-02"),
        exit_price=exit_price,
        quantity=quantity,
        entry_signal="BUY",
        exit_signal="SELL",
    )


def make_result(
    initial_capital: float,
    final_value: float,
    trades: list[Trade] | None = None,
    total_values: list[float] | None = None,
    open_position: OpenPosition | None = None,
) -> BacktestResult:
    return BacktestResult(
        initial_capital=initial_capital,
        final_value=final_value,
        trades=trades or [],
        equity_curve=make_equity_curve(total_values or []),
        open_position=open_position,
        strategy_name="TestStrategy",
        strategy_params={},
    )


# ---------------------------------------------------------------------------
# Total return / absolute P&L
# ---------------------------------------------------------------------------

class TestTotalReturnAndPnl:
    def test_positive_return(self):
        result = make_result(1000.0, 1500.0, total_values=[1000.0, 1500.0])
        m = calculate_performance_metrics(result)
        assert m.total_return_pct == pytest.approx(50.0)
        assert m.absolute_pnl == pytest.approx(500.0)

    def test_negative_return(self):
        result = make_result(1000.0, 600.0, total_values=[1000.0, 600.0])
        m = calculate_performance_metrics(result)
        assert m.total_return_pct == pytest.approx(-40.0)
        assert m.absolute_pnl == pytest.approx(-400.0)

    def test_zero_change(self):
        result = make_result(1000.0, 1000.0, total_values=[1000.0, 1000.0])
        m = calculate_performance_metrics(result)
        assert m.total_return_pct == pytest.approx(0.0)
        assert m.absolute_pnl == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Trade count / win rate / average win / average loss
# ---------------------------------------------------------------------------

class TestTradeStatistics:
    def test_mixed_winning_and_losing_trades(self):
        trades = [
            make_trade(10, 20),   # win, pnl = +1000
            make_trade(20, 10),   # loss, pnl = -1000
            make_trade(10, 15),   # win, pnl = +500
            make_trade(15, 12),   # loss, pnl = -300
        ]
        result = make_result(1000.0, 1200.0, trades=trades, total_values=[1000.0, 1200.0])
        m = calculate_performance_metrics(result)

        assert m.num_trades == 4
        assert m.win_rate_pct == pytest.approx(50.0)
        assert m.average_win == pytest.approx((1000 + 500) / 2)
        assert m.average_loss == pytest.approx((-1000 - 300) / 2)

    def test_zero_trades(self):
        result = make_result(1000.0, 1000.0, trades=[], total_values=[1000.0, 1000.0])
        m = calculate_performance_metrics(result)
        assert m.num_trades == 0
        assert m.win_rate_pct is None
        assert m.average_win is None
        assert m.average_loss is None

    def test_all_winning_trades(self):
        trades = [make_trade(10, 20), make_trade(10, 30)]
        result = make_result(1000.0, 2000.0, trades=trades, total_values=[1000.0, 2000.0])
        m = calculate_performance_metrics(result)
        assert m.win_rate_pct == pytest.approx(100.0)
        assert m.average_loss is None
        assert m.average_win == pytest.approx((1000 + 2000) / 2)

    def test_all_losing_trades(self):
        trades = [make_trade(20, 10), make_trade(30, 10)]
        result = make_result(1000.0, 500.0, trades=trades, total_values=[1000.0, 500.0])
        m = calculate_performance_metrics(result)
        assert m.win_rate_pct == pytest.approx(0.0)
        assert m.average_win is None
        assert m.average_loss == pytest.approx((-1000 - 2000) / 2)

    def test_breakeven_trade_counts_as_non_winning(self):
        # pnl == 0 exactly -> Trade.is_win is False (pnl > 0 required) ->
        # counted as "losing/non-winning", matching Trade.is_win's own definition.
        trades = [make_trade(10, 10)]  # pnl = 0
        result = make_result(1000.0, 1000.0, trades=trades, total_values=[1000.0, 1000.0])
        m = calculate_performance_metrics(result)
        assert m.win_rate_pct == pytest.approx(0.0)
        assert m.average_win is None
        assert m.average_loss == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Maximum drawdown
# ---------------------------------------------------------------------------

class TestMaxDrawdown:
    def test_hand_verifiable_drawdown(self):
        # Peaks at 1200 then 1300; worst decline is from 1300 down to 800.
        values = [1000, 1200, 900, 1100, 1300, 800]
        result = make_result(1000, 800, total_values=values)
        m = calculate_performance_metrics(result)
        expected = (800 - 1300) / 1300 * 100
        assert m.max_drawdown_pct == pytest.approx(expected)

    def test_drawdown_is_reported_as_negative(self):
        values = [1000, 1200, 900]
        result = make_result(1000, 900, total_values=values)
        m = calculate_performance_metrics(result)
        assert m.max_drawdown_pct < 0

    def test_ever_increasing_curve_has_zero_drawdown(self):
        values = [1000, 1100, 1200, 1300]
        result = make_result(1000, 1300, total_values=values)
        m = calculate_performance_metrics(result)
        assert m.max_drawdown_pct == pytest.approx(0.0)

    def test_flat_curve_has_zero_drawdown(self):
        result = make_result(1000, 1000, total_values=[1000, 1000, 1000])
        m = calculate_performance_metrics(result)
        assert m.max_drawdown_pct == pytest.approx(0.0)

    def test_result_reflects_only_the_window_given_not_future_data(self):
        """A drawdown that only happens LATER must not appear in a result
        computed from an equity curve that stops BEFORE that decline --
        the function only ever sees what's in the DataFrame it's given."""
        early_window = [1000, 1100, 1050]  # mild dip, ~4.5% drawdown
        full_curve = early_window + [1200, 300]  # huge crash added later

        early_result = make_result(1000, 1050, total_values=early_window)
        full_result = make_result(1000, 300, total_values=full_curve)

        m_early = calculate_performance_metrics(early_result)
        m_full = calculate_performance_metrics(full_result)

        assert m_early.max_drawdown_pct == pytest.approx((1050 - 1100) / 1100 * 100)
        assert m_full.max_drawdown_pct < m_early.max_drawdown_pct  # much worse
        assert m_early.max_drawdown_pct != m_full.max_drawdown_pct


# ---------------------------------------------------------------------------
# Sharpe ratio / volatility / best-worst period -- independently verified
# ---------------------------------------------------------------------------

class TestSharpeVolatilityAndPeriodReturns:
    VALUES = [100, 110, 99, 108.9]  # daily returns: +10%, -10%, +10%

    def _expected_returns(self):
        v = self.VALUES
        return [v[i] / v[i - 1] - 1 for i in range(1, len(v))]

    def test_sharpe_ratio_matches_independent_calculation(self):
        returns = self._expected_returns()
        expected_sharpe = statistics.mean(returns) / statistics.stdev(returns) * math.sqrt(TRADING_DAYS_PER_YEAR)

        result = make_result(100, self.VALUES[-1], total_values=self.VALUES)
        m = calculate_performance_metrics(result)
        assert m.sharpe_ratio == pytest.approx(expected_sharpe)

    def test_risk_free_rate_is_treated_as_annual_and_converted_to_daily(self):
        returns = self._expected_returns()
        annual_rf = 0.0504  # chosen so daily = 0.0504 / 252 = exactly 0.0002
        daily_rf = annual_rf / TRADING_DAYS_PER_YEAR
        expected_sharpe = (statistics.mean(returns) - daily_rf) / statistics.stdev(returns) * math.sqrt(TRADING_DAYS_PER_YEAR)

        result = make_result(100, self.VALUES[-1], total_values=self.VALUES)
        m = calculate_performance_metrics(result, risk_free_rate=annual_rf)
        assert m.sharpe_ratio == pytest.approx(expected_sharpe)
        assert m.risk_free_rate_annual == annual_rf

    def test_annualized_volatility_matches_independent_calculation(self):
        returns = self._expected_returns()
        expected_vol = statistics.stdev(returns) * math.sqrt(TRADING_DAYS_PER_YEAR) * 100

        result = make_result(100, self.VALUES[-1], total_values=self.VALUES)
        m = calculate_performance_metrics(result)
        assert m.annualized_volatility_pct == pytest.approx(expected_vol)

    def test_best_and_worst_period_return(self):
        returns = self._expected_returns()
        result = make_result(100, self.VALUES[-1], total_values=self.VALUES)
        m = calculate_performance_metrics(result)
        assert m.best_period_return_pct == pytest.approx(max(returns) * 100)
        assert m.worst_period_return_pct == pytest.approx(min(returns) * 100)


# ---------------------------------------------------------------------------
# Flat / empty / single-row equity curves
# ---------------------------------------------------------------------------

class TestFlatEmptyAndSingleRowCurves:
    def test_flat_curve(self):
        result = make_result(1000, 1000, total_values=[1000, 1000, 1000, 1000])
        m = calculate_performance_metrics(result)
        assert m.total_return_pct == pytest.approx(0.0)
        assert m.max_drawdown_pct == pytest.approx(0.0)
        assert m.sharpe_ratio is None  # zero std -> division by zero, guarded
        assert m.annualized_volatility_pct == pytest.approx(0.0)  # legitimately zero, not None

    def test_empty_equity_curve_does_not_crash(self):
        result = make_result(1000, 1000, total_values=[])
        m = calculate_performance_metrics(result)
        assert m.max_drawdown_pct is None
        assert m.sharpe_ratio is None
        assert m.annualized_volatility_pct is None
        assert m.best_period_return_pct is None
        assert m.worst_period_return_pct is None
        # total return/pnl don't depend on the equity curve at all, so
        # they're still perfectly well-defined even with zero rows.
        assert m.total_return_pct == pytest.approx(0.0)

    def test_single_row_equity_curve_does_not_crash(self):
        result = make_result(1000, 1000, total_values=[1000])
        m = calculate_performance_metrics(result)
        assert m.max_drawdown_pct is None
        assert m.sharpe_ratio is None
        assert m.annualized_volatility_pct is None


# ---------------------------------------------------------------------------
# Open position handling
# ---------------------------------------------------------------------------

class TestOpenPosition:
    def test_open_position_flag_and_exclusion_from_trade_stats(self):
        completed_trade = make_trade(10, 20)  # one completed win
        op = OpenPosition(
            entry_date=pd.Timestamp("2024-01-05"),
            entry_price=50.0,
            quantity=10,
            entry_signal="BUY",
        )
        result = make_result(
            1000, 1400, trades=[completed_trade], total_values=[1000, 1200, 1400], open_position=op
        )
        m = calculate_performance_metrics(result)

        assert m.has_open_position is True
        assert m.num_trades == 1  # only the completed trade, never the open one
        assert m.win_rate_pct == pytest.approx(100.0)  # based on the 1 completed trade only

    def test_no_open_position_flag_false(self):
        result = make_result(1000, 1000, total_values=[1000, 1000])
        m = calculate_performance_metrics(result)
        assert m.has_open_position is False

    def test_open_position_unrealized_value_still_reflected_in_total_return(self):
        # final_value already includes the open position's unrealized
        # value (that's Phase 5's job, not this module's) -- confirm
        # total_return_pct is computed from that final_value as-is.
        op = OpenPosition(pd.Timestamp("2024-01-01"), 10.0, 100, "BUY")
        result = make_result(1000, 1400, trades=[], total_values=[1000, 1400], open_position=op)
        m = calculate_performance_metrics(result)
        assert m.total_return_pct == pytest.approx(40.0)
        assert m.has_open_position is True
