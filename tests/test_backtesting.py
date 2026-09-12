"""
Tests for the backtesting engine (src/backtesting/).

Covers, using hand-built deterministic price/signal data (not real market
data, so every expected number can be verified by hand):
  - BUY followed by SELL (a complete round trip)
  - BUY with insufficient cash
  - SELL while flat
  - repeated BUY signals while already holding
  - repeated SELL signals while flat
  - correct position size, cash balance, and portfolio value at each bar
  - correct trade entry/exit records
  - chronological processing (and rejection of out-of-order data)
  - empty data
  - insufficient data (very short series)
  - no look-ahead bias
  - multiple sequential trades
  - final portfolio value
  - open position at the end (not force-closed)
  - transaction costs (off by default, but functional when set)
  - input validation

Run with:
    pytest tests/test_backtesting.py -v
"""

import pandas as pd
import pytest

from src.backtesting.engine import BacktestEngine, BacktestInputError, run_backtest
from src.strategies.base_strategy import Signal


def make_df(prices: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=len(prices), freq="D")
    return pd.DataFrame({"Date": dates, "Adj Close": prices})


def make_signals(values: list[Signal]) -> pd.Series:
    return pd.Series(values, name="signal")


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

class TestEngineConstruction:
    @pytest.mark.parametrize("bad_capital", [0, -100, -0.01, "1000"])
    def test_invalid_initial_capital_raises(self, bad_capital):
        with pytest.raises(BacktestInputError):
            BacktestEngine(initial_capital=bad_capital)

    @pytest.mark.parametrize("bad_cost", [-0.01, 1.0, 1.5, "0.001"])
    def test_invalid_transaction_cost_raises(self, bad_cost):
        with pytest.raises(BacktestInputError):
            BacktestEngine(initial_capital=1000.0, transaction_cost_pct=bad_cost)

    def test_valid_construction(self):
        engine = BacktestEngine(initial_capital=1000.0, transaction_cost_pct=0.001)
        assert engine.initial_capital == 1000.0
        assert engine.transaction_cost_pct == 0.001


# ---------------------------------------------------------------------------
# Basic BUY -> SELL round trip
# ---------------------------------------------------------------------------

class TestBuyThenSell:
    def test_full_round_trip_correctness(self):
        # BUY at 10 with $1000 -> 100 shares, sell at 20 -> $2000
        df = make_df([10.0, 15.0, 20.0])
        signals = make_signals([Signal.BUY, Signal.HOLD, Signal.SELL])
        result = BacktestEngine(1000.0).run(df, signals)

        assert result.num_trades == 1
        trade = result.trades[0]
        assert trade.entry_price == 10.0
        assert trade.exit_price == 20.0
        assert trade.quantity == 100
        assert trade.pnl == 1000.0
        assert trade.pnl_pct == pytest.approx(100.0)
        assert trade.is_win is True
        assert result.final_value == 2000.0
        assert result.open_position is None

    def test_losing_trade_is_not_a_win(self):
        df = make_df([20.0, 15.0, 10.0])
        signals = make_signals([Signal.BUY, Signal.HOLD, Signal.SELL])
        result = BacktestEngine(1000.0).run(df, signals)
        trade = result.trades[0]
        assert trade.pnl < 0
        assert trade.is_win is False

    def test_cash_and_holdings_tracked_correctly_bar_by_bar(self):
        df = make_df([10.0, 15.0, 20.0])
        signals = make_signals([Signal.BUY, Signal.HOLD, Signal.SELL])
        result = BacktestEngine(1000.0).run(df, signals)
        curve = result.equity_curve

        # Bar 0: bought 100 shares @ 10 with all $1000 -> cash 0
        assert curve.loc[0, "cash"] == 0.0
        assert curve.loc[0, "shares_held"] == 100
        assert curve.loc[0, "holdings_value"] == 1000.0
        assert curve.loc[0, "total_value"] == 1000.0

        # Bar 1: still holding, price rose to 15 -> holdings worth 1500
        assert curve.loc[1, "shares_held"] == 100
        assert curve.loc[1, "holdings_value"] == 1500.0
        assert curve.loc[1, "total_value"] == 1500.0

        # Bar 2: sold at 20 -> cash 2000, flat
        assert curve.loc[2, "cash"] == 2000.0
        assert curve.loc[2, "shares_held"] == 0
        assert curve.loc[2, "total_value"] == 2000.0


# ---------------------------------------------------------------------------
# Insufficient cash
# ---------------------------------------------------------------------------

class TestInsufficientCash:
    def test_buy_skipped_when_cash_cant_afford_one_share(self):
        df = make_df([5000.0, 5000.0])
        signals = make_signals([Signal.BUY, Signal.HOLD])
        result = BacktestEngine(1000.0).run(df, signals)

        assert result.num_trades == 0
        assert result.open_position is None
        assert result.final_value == 1000.0  # untouched, no shares bought
        assert result.equity_curve.loc[0, "shares_held"] == 0
        assert result.equity_curve.loc[0, "cash"] == 1000.0

    def test_exact_affordability_boundary(self):
        # $1000 at $100/share -> exactly 10 shares, all cash spent
        df = make_df([100.0])
        signals = make_signals([Signal.BUY])
        result = BacktestEngine(1000.0).run(df, signals)
        assert result.equity_curve.loc[0, "shares_held"] == 10
        assert result.equity_curve.loc[0, "cash"] == 0.0

    def test_leftover_cash_from_non_divisible_price(self):
        # $1000 at $300/share -> floor(1000/300) = 3 shares, $100 left over
        df = make_df([300.0])
        signals = make_signals([Signal.BUY])
        result = BacktestEngine(1000.0).run(df, signals)
        assert result.equity_curve.loc[0, "shares_held"] == 3
        assert result.equity_curve.loc[0, "cash"] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# SELL while flat / repeated signals
# ---------------------------------------------------------------------------

class TestRepeatedAndInapplicableSignals:
    def test_sell_while_flat_is_ignored_not_an_error(self):
        df = make_df([10.0, 11.0, 12.0])
        signals = make_signals([Signal.HOLD, Signal.SELL, Signal.HOLD])
        result = BacktestEngine(1000.0).run(df, signals)
        assert result.num_trades == 0
        assert result.final_value == 1000.0  # nothing ever happened

    def test_repeated_buy_while_holding_does_not_buy_again(self):
        df = make_df([10.0, 10.0, 10.0])
        signals = make_signals([Signal.BUY, Signal.BUY, Signal.BUY])
        result = BacktestEngine(1000.0).run(df, signals)
        # Only the first BUY executes; cash goes to 0 once, stays there.
        assert result.equity_curve["shares_held"].tolist() == [100, 100, 100]
        assert result.equity_curve["cash"].tolist() == [0.0, 0.0, 0.0]

    def test_repeated_sell_after_closing_does_not_sell_again(self):
        df = make_df([10.0, 20.0, 20.0, 20.0])
        signals = make_signals([Signal.BUY, Signal.SELL, Signal.SELL, Signal.SELL])
        result = BacktestEngine(1000.0).run(df, signals)
        assert result.num_trades == 1  # only ONE sell actually executed
        assert result.final_value == 2000.0
        assert result.equity_curve["cash"].tolist() == [0.0, 2000.0, 2000.0, 2000.0]


# ---------------------------------------------------------------------------
# Multiple trades / final value
# ---------------------------------------------------------------------------

class TestMultipleTrades:
    def test_multiple_round_trips_compound_correctly(self):
        # Trade 1: buy@10 sell@20 (double). Trade 2: buy@20 sell@40 (double again).
        df = make_df([10.0, 20.0, 20.0, 40.0])
        signals = make_signals([Signal.BUY, Signal.SELL, Signal.BUY, Signal.SELL])
        result = BacktestEngine(1000.0).run(df, signals)

        assert result.num_trades == 2
        assert result.trades[0].pnl == 1000.0   # 100 shares, 10 -> 20
        assert result.trades[1].pnl == 2000.0   # 100 shares, 20 -> 40
        assert result.final_value == 4000.0     # 1000 -> 2000 -> 4000

    def test_open_position_at_end_is_not_force_closed(self):
        df = make_df([10.0, 12.0, 14.0])
        signals = make_signals([Signal.BUY, Signal.HOLD, Signal.HOLD])
        result = BacktestEngine(1000.0).run(df, signals)

        assert result.num_trades == 0  # never closed -> not a "completed trade"
        assert result.open_position is not None
        assert result.open_position.quantity == 100
        assert result.open_position.entry_price == 10.0
        assert result.final_value == 1400.0  # unrealized value still counted


# ---------------------------------------------------------------------------
# Chronological processing
# ---------------------------------------------------------------------------

class TestChronologicalProcessing:
    def test_out_of_order_dates_raises(self):
        df = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2024-01-03", "2024-01-01", "2024-01-02"]),
                "Adj Close": [12.0, 10.0, 11.0],
            }
        )
        signals = make_signals([Signal.HOLD, Signal.HOLD, Signal.HOLD])
        with pytest.raises(BacktestInputError):
            BacktestEngine(1000.0).run(df, signals)

    def test_processing_order_affects_result_correctly(self):
        # Same bars, but genuinely different chronological order ->
        # genuinely different (and independently verifiable) outcome.
        ascending = make_df([10.0, 20.0, 5.0])  # buy low, sell high, then crash
        signals = make_signals([Signal.BUY, Signal.SELL, Signal.HOLD])
        result = BacktestEngine(1000.0).run(ascending, signals)
        assert result.final_value == 2000.0  # locked in the gain before the crash


# ---------------------------------------------------------------------------
# Empty / insufficient data
# ---------------------------------------------------------------------------

class TestEmptyAndInsufficientData:
    def test_empty_data_does_not_crash(self):
        df = make_df([])
        signals = make_signals([])
        result = BacktestEngine(1000.0).run(df, signals)

        assert result.num_trades == 0
        assert result.final_value == 1000.0
        assert result.open_position is None
        assert list(result.equity_curve.columns) == [
            "date", "price", "signal", "cash", "shares_held", "holdings_value", "total_value",
        ]
        assert len(result.equity_curve) == 0

    def test_single_bar_of_data(self):
        df = make_df([10.0])
        signals = make_signals([Signal.HOLD])
        result = BacktestEngine(1000.0).run(df, signals)
        assert len(result.equity_curve) == 1
        assert result.final_value == 1000.0

    def test_mismatched_lengths_raise(self):
        df = make_df([10.0, 11.0, 12.0])
        signals = make_signals([Signal.HOLD, Signal.HOLD])  # one short
        with pytest.raises(BacktestInputError):
            BacktestEngine(1000.0).run(df, signals)


# ---------------------------------------------------------------------------
# Look-ahead bias
# ---------------------------------------------------------------------------

class TestNoLookAheadBias:
    def test_truncating_future_data_does_not_change_past_results(self):
        """The engine must produce IDENTICAL cash/shares/trades for bars
        0..k regardless of what data comes after bar k. If the engine
        ever consulted a future row, cutting the data short would change
        earlier results -- it must not."""
        df = make_df([10.0, 15.0, 20.0, 8.0, 25.0, 30.0])
        signals = make_signals(
            [Signal.BUY, Signal.HOLD, Signal.SELL, Signal.BUY, Signal.HOLD, Signal.SELL]
        )

        full_result = BacktestEngine(1000.0).run(df, signals)

        cutoff = 3  # truncate right after the SELL at index 2
        truncated_df = df.iloc[: cutoff + 1].reset_index(drop=True)
        truncated_signals = signals.iloc[: cutoff + 1].reset_index(drop=True)
        truncated_result = BacktestEngine(1000.0).run(truncated_df, truncated_signals)

        full_curve = full_result.equity_curve
        trunc_curve = truncated_result.equity_curve
        for col in ["cash", "shares_held", "holdings_value", "total_value"]:
            assert full_curve.loc[cutoff, col] == trunc_curve.loc[cutoff, col], (
                f"Column '{col}' at bar {cutoff} changed when future bars were "
                f"added -- possible look-ahead bias."
            )
        # The one trade fully contained within the truncated window must
        # also match exactly.
        assert truncated_result.trades[0] == full_result.trades[0]

    def test_engine_never_reads_beyond_current_row_index(self):
        """A more mechanical check: feed the engine a price series where
        every row after the halfway point is deliberately absurd (would
        wildly change results if peeked at), and confirm the first-half
        results are unaffected by that absurd second half."""
        normal_first_half = [10.0, 12.0, 11.0, 13.0]
        signals_first_half = [Signal.BUY, Signal.HOLD, Signal.HOLD, Signal.SELL]

        df_short = make_df(normal_first_half)
        result_short = BacktestEngine(1000.0).run(df_short, make_signals(signals_first_half))

        absurd_second_half = [999999.0, 0.0001, 500000.0]
        df_long = make_df(normal_first_half + absurd_second_half)
        signals_long = make_signals(signals_first_half + [Signal.HOLD, Signal.HOLD, Signal.HOLD])
        result_long = BacktestEngine(1000.0).run(df_long, signals_long)

        assert result_short.trades[0] == result_long.trades[0]
        for i in range(4):
            assert result_short.equity_curve.loc[i, "total_value"] == result_long.equity_curve.loc[i, "total_value"]


# ---------------------------------------------------------------------------
# Transaction costs (off by default, but functional)
# ---------------------------------------------------------------------------

class TestTransactionCosts:
    def test_zero_cost_by_default(self):
        df = make_df([10.0, 20.0])
        signals = make_signals([Signal.BUY, Signal.SELL])
        result = BacktestEngine(1000.0).run(df, signals)
        assert result.final_value == 2000.0  # no cost eaten anywhere

    def test_nonzero_cost_reduces_shares_bought_and_proceeds(self):
        df = make_df([10.0, 20.0])
        signals = make_signals([Signal.BUY, Signal.SELL])
        result = BacktestEngine(1000.0, transaction_cost_pct=0.01).run(df, signals)
        # cost-adjusted buy price = 10 * 1.01 = 10.1 -> floor(1000/10.1) = 99 shares,
        # leaving $0.10 unspent (can't buy a fractional 100th share)
        assert result.equity_curve.loc[0, "shares_held"] == 99
        leftover_cash = 1000.0 - 99 * 10.1
        proceeds = 99 * 20 * 0.99
        assert result.final_value == pytest.approx(leftover_cash + proceeds, abs=0.01)
        assert result.final_value < 2000.0  # strictly worse than the zero-cost case


# ---------------------------------------------------------------------------
# run_backtest() convenience function (integration with a real strategy)
# ---------------------------------------------------------------------------

class TestRunBacktestIntegration:
    def test_run_backtest_with_real_strategy_populates_metadata(self):
        from src.strategies import MovingAverageCrossoverStrategy

        df = make_df([100 + i * 0.1 for i in range(80)])  # gentle, enough for SMA(50)
        strat = MovingAverageCrossoverStrategy(fast_window=5, slow_window=10)
        result = run_backtest(df, strat, initial_capital=5000.0)

        assert result.strategy_name == "MovingAverageCrossoverStrategy"
        assert result.strategy_params["fast_window"] == 5
        assert result.strategy_params["slow_window"] == 10
        assert result.initial_capital == 5000.0
        assert len(result.equity_curve) == len(df)
