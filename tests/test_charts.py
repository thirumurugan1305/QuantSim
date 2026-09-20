"""
Tests for src/ui/charts.py

These are pure-function tests: build a chart from hand-built data and
check the returned `plotly.graph_objects.Figure` -- trace counts,
titles, and (most importantly) that marker/line coordinates exactly
match the input data, not just "it didn't crash." No Streamlit import
anywhere in this file, matching the module under test.

Run with:
    pytest tests/test_charts.py -v
"""

import pandas as pd
import plotly.graph_objects as go
import pytest

from src.backtesting.models import Trade
from src.ui.charts import (
    _drawdown_series_pct,
    build_drawdown_chart,
    build_equity_curve_chart,
    build_macd_chart,
    build_price_chart,
    build_rsi_chart,
    build_signal_pulse_chart,
)


def make_price_df(prices: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"Date": pd.date_range("2024-01-01", periods=len(prices), freq="D"), "Adj Close": prices})


def make_equity_curve(total_values: list[float]) -> pd.DataFrame:
    n = len(total_values)
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "price": total_values,
            "signal": ["HOLD"] * n,
            "cash": [0.0] * n,
            "shares_held": [0] * n,
            "holdings_value": total_values,
            "total_value": total_values,
        }
    )


def make_trade(entry_price, entry_date, exit_price, exit_date, quantity=100) -> Trade:
    return Trade(
        entry_date=pd.Timestamp(entry_date),
        entry_price=entry_price,
        exit_date=pd.Timestamp(exit_date),
        exit_price=exit_price,
        quantity=quantity,
        entry_signal="BUY",
        exit_signal="SELL",
    )


# ---------------------------------------------------------------------------
# Price chart
# ---------------------------------------------------------------------------

class TestBuildPriceChart:
    def test_returns_a_figure(self):
        df = make_price_df([10, 11, 12])
        fig = build_price_chart(df)
        assert isinstance(fig, go.Figure)

    def test_price_only_has_one_trace(self):
        df = make_price_df([10, 11, 12])
        fig = build_price_chart(df)
        assert len(fig.data) == 1
        assert fig.data[0].name == "Adj Close"

    def test_sma_overlay_adds_traces_with_correct_labels(self):
        df = make_price_df([10, 11, 12, 13])
        sma = {"SMA 2": df["Adj Close"].rolling(2).mean()}
        fig = build_price_chart(df, sma_series=sma)
        assert len(fig.data) == 2
        assert fig.data[1].name == "SMA 2"

    def test_trade_markers_added_with_exact_coordinates(self):
        df = make_price_df([10, 11, 12, 13, 14])
        trade = make_trade(11.0, "2024-01-02", 13.0, "2024-01-04")
        fig = build_price_chart(df, trades=[trade])

        # price line + BUY marker trace + SELL marker trace = 3 traces
        assert len(fig.data) == 3
        buy_trace = next(t for t in fig.data if t.name == "BUY")
        sell_trace = next(t for t in fig.data if t.name == "SELL")

        assert buy_trace.x[0] == pd.Timestamp("2024-01-02")
        assert buy_trace.y[0] == 11.0
        assert sell_trace.x[0] == pd.Timestamp("2024-01-04")
        assert sell_trace.y[0] == 13.0

    def test_multiple_trades_produce_matching_marker_counts(self):
        df = make_price_df([10, 11, 12, 13, 14, 15, 16])
        trades = [
            make_trade(10.0, "2024-01-01", 12.0, "2024-01-03"),
            make_trade(12.0, "2024-01-03", 16.0, "2024-01-07"),
        ]
        fig = build_price_chart(df, trades=trades)
        buy_trace = next(t for t in fig.data if t.name == "BUY")
        sell_trace = next(t for t in fig.data if t.name == "SELL")
        assert len(buy_trace.x) == 2
        assert len(sell_trace.x) == 2
        assert list(buy_trace.y) == [10.0, 12.0]
        assert list(sell_trace.y) == [12.0, 16.0]

    def test_no_trades_means_no_marker_traces(self):
        df = make_price_df([10, 11, 12])
        fig = build_price_chart(df, trades=None)
        names = [t.name for t in fig.data]
        assert "BUY" not in names
        assert "SELL" not in names

    def test_axis_title_includes_units(self):
        df = make_price_df([10, 11])
        fig = build_price_chart(df, price_column="Adj Close")
        assert "$" in fig.layout.yaxis.title.text


# ---------------------------------------------------------------------------
# RSI chart
# ---------------------------------------------------------------------------

class TestBuildRsiChart:
    def test_returns_a_figure_with_one_line_trace(self):
        dates = pd.date_range("2024-01-01", periods=5)
        rsi = pd.Series([40, 50, 60, 70, 80])
        fig = build_rsi_chart(dates, rsi)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1

    def test_reference_lines_use_configured_thresholds(self):
        dates = pd.date_range("2024-01-01", periods=5)
        rsi = pd.Series([40, 50, 60, 70, 80])
        fig = build_rsi_chart(dates, rsi, oversold=20, overbought=80)
        hline_ys = [shape.y0 for shape in fig.layout.shapes]
        assert 20 in hline_ys
        assert 80 in hline_ys

    def test_yaxis_fixed_to_0_100(self):
        dates = pd.date_range("2024-01-01", periods=3)
        rsi = pd.Series([10, 50, 90])
        fig = build_rsi_chart(dates, rsi)
        assert list(fig.layout.yaxis.range) == [0, 100]


# ---------------------------------------------------------------------------
# MACD chart
# ---------------------------------------------------------------------------

class TestBuildMacdChart:
    def test_has_three_traces(self):
        dates = pd.date_range("2024-01-01", periods=5)
        macd_df = pd.DataFrame(
            {"macd_line": [0.1, 0.2, 0.1, 0.0, -0.1], "signal_line": [0.05] * 5, "histogram": [0.05, 0.15, 0.05, -0.05, -0.15]}
        )
        fig = build_macd_chart(dates, macd_df)
        names = {t.name for t in fig.data}
        assert names == {"Histogram", "MACD line", "Signal line"}

    def test_histogram_values_match_input_exactly(self):
        dates = pd.date_range("2024-01-01", periods=3)
        macd_df = pd.DataFrame({"macd_line": [1, 2, 3], "signal_line": [0, 1, 2], "histogram": [1, 1, 1]})
        fig = build_macd_chart(dates, macd_df)
        hist_trace = next(t for t in fig.data if t.name == "Histogram")
        assert list(hist_trace.y) == [1, 1, 1]


# ---------------------------------------------------------------------------
# Equity curve chart
# ---------------------------------------------------------------------------

class TestBuildEquityCurveChart:
    def test_line_matches_total_value_exactly(self):
        values = [1000.0, 1100.0, 1050.0]
        eq = make_equity_curve(values)
        fig = build_equity_curve_chart(eq)
        assert list(fig.data[0].y) == values

    def test_axis_title_includes_dollar_sign(self):
        eq = make_equity_curve([1000.0, 1100.0])
        fig = build_equity_curve_chart(eq)
        assert "$" in fig.layout.yaxis.title.text


# ---------------------------------------------------------------------------
# Drawdown chart / series
# ---------------------------------------------------------------------------

class TestDrawdown:
    def test_drawdown_series_never_positive(self):
        eq = make_equity_curve([1000, 1200, 900, 1100, 1300, 800])
        dd = _drawdown_series_pct(eq)
        assert (dd <= 0).all()

    def test_drawdown_matches_hand_calculation(self):
        eq = make_equity_curve([1000, 1200, 900, 1100, 1300, 800])
        dd = _drawdown_series_pct(eq)
        expected_min = (800 - 1300) / 1300 * 100
        assert dd.min() == pytest.approx(expected_min)

    def test_ever_increasing_curve_has_zero_drawdown_throughout(self):
        eq = make_equity_curve([1000, 1100, 1200, 1300])
        dd = _drawdown_series_pct(eq)
        assert (dd == 0).all()

    def test_chart_is_a_figure(self):
        eq = make_equity_curve([1000, 900, 1100])
        fig = build_drawdown_chart(eq)
        assert isinstance(fig, go.Figure)
        assert "%" in fig.layout.yaxis.title.text


# ---------------------------------------------------------------------------
# Signal pulse chart
# ---------------------------------------------------------------------------

class TestSignalPulseChart:
    def test_values_match_input_exactly(self):
        dates = pd.date_range("2024-01-01", periods=5)
        signal_numeric = pd.Series([1, 0, -1, 0, 1])
        fig = build_signal_pulse_chart(dates, signal_numeric)
        assert list(fig.data[0].y) == [1, 0, -1, 0, 1]

    def test_is_a_figure_with_one_bar_trace(self):
        dates = pd.date_range("2024-01-01", periods=3)
        signal_numeric = pd.Series([1, -1, 0])
        fig = build_signal_pulse_chart(dates, signal_numeric)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
