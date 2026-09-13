"""
QuantSim — Algorithmic Trading & Backtesting Platform
Entry point for the Streamlit application.

Run with:
    streamlit run app.py

Phase 2 scope (this file, today):
    - App shell: title, tagline, disclaimer (from Phase 1).
    - Sidebar controls are now REAL: the ticker and date range you enter
      are passed to `src.data.market_data.fetch_market_data()`.
    - That service tries a live yfinance download first, and transparently
      falls back to the bundled synthetic sample data if the download
      fails for any reason. The UI always tells you which one you got.
    - Any cleaning warnings (missing rows, bad prices, etc.) are surfaced
      to you, not silently swallowed.

Phase 3 scope (this file, today):
    - A new "Technical Indicators Preview" section below the market data,
      computed from `src.indicators.technical_indicators` on whatever
      data (live or fallback) is currently loaded: SMA(20)/SMA(50)
      overlaid on price, RSI(14), and MACD(12, 26, 9).
    - This is a PREVIEW for learning/verification purposes only — it does
      not generate BUY/SELL signals or trade anything. That's Phase 4
      (strategies) and Phase 5 (backtesting engine).

Phase 4 scope (this file, today):
    - The sidebar's "Strategy" dropdown is now REAL, with parameter
      inputs specific to whichever strategy is selected.
    - A new "Strategy Signals Preview" section computes BUY/SELL/HOLD
      signals via `src.strategies` on the currently loaded data and
      displays them as counts, a signal-events table, and a simple
      signal-pulse chart.
    - This is STILL a preview only: no trades are executed, no
      profit/loss is calculated, and no portfolio is tracked. That's
      Phase 5 (the backtesting engine). The "Run Backtest" button stays
      disabled.

Phase 5 scope (this file, today):
    - The sidebar's "Run Backtest" button is now REAL and enabled once a
      valid strategy is configured. Clicking it runs
      `src.backtesting.engine.run_backtest()` — the strategy generates
      signals (Phase 4), and the engine simulates trades from them.
    - A new "Backtest Results (Preview)" section shows final portfolio
      value, completed-trade count, an equity curve, and a trade table.
    - This is STILL a preview: no performance ratios (Sharpe ratio, max
      drawdown, win rate, CAGR) are calculated yet — that's Phase 6.
    - The last backtest result is kept in `st.session_state` so it stays
      visible across unrelated reruns (e.g. moving a slider elsewhere),
      but it's always labeled with exactly which ticker/dates/strategy
      configuration it was actually run with, since that configuration
      may no longer match the sidebar's current values.

Phase 6 scope (this file, today):
    - A new "Performance Metrics (Preview)" section, shown right after
      "Backtest Results (Preview)", computed via
      `src.analytics.calculate_performance_metrics()` on the SAME
      BacktestResult already in `st.session_state` — no new backtest is
      run, no strategy is re-invoked.
    - Displays Total Return, Absolute P&L, Win Rate, Max Drawdown,
      Sharpe Ratio, Number of Trades, Average Win, Average Loss, and
      Annualized Volatility. Any metric that isn't meaningful for the
      current data (e.g. no completed trades yet) shows "N/A".
    - If the backtest ended with a still-open position, a note clarifies
      that the displayed return includes that position's unrealized
      value.

Still not built: the local SQLite database (Phase 7) and the richer
multi-tab UI (Phase 8).
"""

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from src.analytics import calculate_performance_metrics
from src.backtesting.engine import BacktestInputError, run_backtest
from src.config import (
    APP_NAME,
    APP_TAGLINE,
    DEFAULT_INITIAL_CAPITAL,
    DISCLAIMER,
)
from src.data.market_data import (
    SOURCE_LIVE,
    MarketDataValidationError,
    fetch_market_data,
)
from src.indicators.technical_indicators import (
    calculate_macd,
    calculate_rsi,
    calculate_sma,
)
from src.strategies import (
    MACDStrategy,
    MovingAverageCrossoverStrategy,
    RSIStrategy,
    Signal,
    StrategyInputError,
)

st.set_page_config(page_title=APP_NAME, layout="wide")

# Minimum rows genuinely needed before each indicator produces at least
# one non-NaN value (see docstrings in technical_indicators.py for why).
_SMA_LONG_WINDOW = 50
_RSI_PERIOD = 14
_MACD_FAST, _MACD_SLOW, _MACD_SIGNAL = 12, 26, 9
_MACD_MIN_ROWS = _MACD_SLOW + _MACD_SIGNAL - 1  # 34


def render_indicators_section(df: pd.DataFrame) -> None:
    """Compute and display SMA/RSI/MACD for the currently loaded data.

    This is a PREVIEW to prove the indicator math works on real (or
    fallback) data — it does not generate trading signals. That begins
    in Phase 4, which will reuse these exact same functions.
    """
    st.subheader("Technical Indicators Preview")
    st.caption(
        "Computed from the price data above using `src/indicators/technical_indicators.py`. "
        "This is a preview only — no BUY/SELL signals are generated yet (that's Phase 4)."
    )

    price = df.set_index("Date")["Adj Close"]
    n_rows = len(price)

    # --- SMA overlay ---------------------------------------------------
    st.markdown(f"**Moving Averages** (SMA 20 / SMA {_SMA_LONG_WINDOW})")
    if n_rows < _SMA_LONG_WINDOW:
        st.info(
            f"Need at least {_SMA_LONG_WINDOW} rows for SMA {_SMA_LONG_WINDOW}; "
            f"only {n_rows} available. Showing SMA 20 only where possible."
        )
    sma_df = pd.DataFrame(
        {
            "Adj Close": price,
            "SMA_20": calculate_sma(price, 20),
            f"SMA_{_SMA_LONG_WINDOW}": calculate_sma(price, _SMA_LONG_WINDOW),
        }
    )
    st.line_chart(sma_df, width="stretch")

    # --- RSI -------------------------------------------------------------
    st.markdown(f"**RSI ({_RSI_PERIOD})**")
    if n_rows < _RSI_PERIOD + 1:
        st.info(f"Need at least {_RSI_PERIOD + 1} rows for RSI({_RSI_PERIOD}); only {n_rows} available.")
    else:
        rsi = calculate_rsi(price, period=_RSI_PERIOD)
        st.line_chart(rsi, width="stretch")
        st.caption(
            "Conventionally, RSI above 70 is considered 'overbought' and below 30 "
            "'oversold' — shown here for reference only, not acted on yet."
        )

    # --- MACD --------------------------------------------------------------
    st.markdown(f"**MACD ({_MACD_FAST}, {_MACD_SLOW}, {_MACD_SIGNAL})**")
    if n_rows < _MACD_MIN_ROWS:
        st.info(
            f"Need at least {_MACD_MIN_ROWS} rows for a MACD({_MACD_FAST},{_MACD_SLOW},{_MACD_SIGNAL}) "
            f"signal line; only {n_rows} available."
        )
    else:
        macd_df = calculate_macd(price, fast=_MACD_FAST, slow=_MACD_SLOW, signal=_MACD_SIGNAL)
        st.line_chart(macd_df[["macd_line", "signal_line"]], width="stretch")
        st.bar_chart(macd_df["histogram"], width="stretch")
        st.caption(
            "Top: MACD line vs. signal line. Bottom: histogram (MACD line minus "
            "signal line) — the gap most 'MACD crossover' strategies watch."
        )


def render_strategy_signals_section(df: pd.DataFrame, strategy) -> None:
    """Run the selected strategy on the currently loaded data and show
    the resulting BUY/SELL/HOLD signals.

    PREVIEW ONLY: this shows what the strategy WOULD flag, bar by bar.
    It does not execute trades, track a portfolio, or calculate any
    profit/loss — that begins in Phase 5, which will call
    `strategy.generate_signals()` (this exact same method) as its
    starting point.
    """
    st.subheader("Strategy Signals Preview")
    info = strategy.describe()
    param_str = ", ".join(f"{k}={v}" for k, v in info["params"].items())
    st.caption(f"**{info['name']}** ({param_str}) — preview only, no trades are executed.")

    try:
        signals = strategy.generate_signals(df)
    except StrategyInputError as exc:
        st.error(f"Could not generate signals: {exc}")
        return

    counts = signals.value_counts()
    col1, col2, col3 = st.columns(3)
    col1.metric("BUY signals", int(counts.get(Signal.BUY, 0)))
    col2.metric("SELL signals", int(counts.get(Signal.SELL, 0)))
    col3.metric("HOLD (no action)", int(counts.get(Signal.HOLD, 0)))

    # A simple numeric "signal pulse" over time: +1 on BUY, -1 on SELL,
    # 0 on HOLD. This is a lightweight way to see WHEN signals fired
    # using only Streamlit's native charts (Plotly-based BUY/SELL price
    # markers are planned for Phase 8's richer UI).
    signal_numeric = signals.map({Signal.BUY: 1, Signal.SELL: -1, Signal.HOLD: 0})
    pulse_df = pd.DataFrame({"Date": df["Date"].values, "signal": signal_numeric.values}).set_index("Date")
    st.bar_chart(pulse_df, width="stretch")
    st.caption("+1 = BUY, -1 = SELL, 0 = HOLD — shows when each strategy rule fired over time.")

    non_hold = signals[signals != Signal.HOLD]
    if non_hold.empty:
        st.info("No BUY or SELL signals were generated for this data/parameter combination.")
    else:
        events_df = pd.DataFrame(
            {
                "Date": df.loc[non_hold.index, "Date"].dt.date.values,
                "Adj Close": df.loc[non_hold.index, "Adj Close"].round(2).values,
                "Signal": [str(s) for s in non_hold.values],
            }
        )
        st.markdown("**Signal events** (BUY/SELL rows only)")
        st.dataframe(events_df, width="stretch")


def render_backtest_section(result, context: dict) -> None:
    """Display a completed BacktestResult: final value, trade count, the
    equity curve, and a trade table.

    PREVIEW ONLY: deliberately does not compute or show Sharpe ratio,
    max drawdown, win rate, or CAGR — those are Phase 6. This section
    only reports the raw simulation output, the same way
    `render_strategy_signals_section` only reports raw signals rather
    than judging them.
    """
    st.subheader("Backtest Results (Preview)")
    param_str = ", ".join(f"{k}={v}" for k, v in context["strategy_params"].items())
    st.caption(
        f"Ran **{context['strategy_name']}** ({param_str}) on **{context['ticker']}**, "
        f"{context['start_date']} to {context['end_date']}, starting capital "
        f"${context['initial_capital']:,.2f}. Change settings and click **Run Backtest** "
        "again to update this."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Initial Capital", f"${result.initial_capital:,.2f}")
    col2.metric("Final Portfolio Value", f"${result.final_value:,.2f}")
    col3.metric("Completed Trades", result.num_trades)

    if result.open_position is not None:
        op = result.open_position
        st.info(
            f"Still holding an **open position** at the end of the data: "
            f"{op.quantity} shares bought at ${op.entry_price:,.2f} on "
            f"{op.entry_date.date()}. Not counted as a completed trade (it "
            "wasn't sold, so it isn't a realized round trip) — its current "
            "unrealized value IS included in the final portfolio value above."
        )

    if result.equity_curve.empty:
        st.info("No bars were processed for this data — nothing to chart.")
        return

    st.markdown("**Portfolio value over time**")
    st.line_chart(result.equity_curve.set_index("date")["total_value"], width="stretch")
    st.caption(
        "Total portfolio value (cash + holdings) at the close of every bar — "
        "this is what a real trader's account balance would have shown."
    )

    if result.trades:
        st.markdown("**Completed trades**")
        trades_df = pd.DataFrame(
            [
                {
                    "Entry Date": t.entry_date.date(),
                    "Entry Price": round(t.entry_price, 2),
                    "Exit Date": t.exit_date.date(),
                    "Exit Price": round(t.exit_price, 2),
                    "Quantity": t.quantity,
                    "P&L ($)": round(t.pnl, 2),
                    "P&L (%)": round(t.pnl_pct, 2),
                }
                for t in result.trades
            ]
        )
        st.dataframe(trades_df, width="stretch")
    else:
        st.info("No completed trades for this data/strategy/parameter combination.")


def _format_metric(value, suffix: str = "", decimals: int = 2) -> str:
    """Format a metric value for display, or 'N/A' if it's None.

    Centralizing this in one place means every metric card handles the
    "not applicable" case the exact same way, rather than each one
    inventing its own N/A formatting.
    """
    if value is None:
        return "N/A"
    return f"{value:,.{decimals}f}{suffix}"


def _format_dollars(value) -> str:
    """Like _format_metric, but with a leading '$' when the value isn't None."""
    if value is None:
        return "N/A"
    return f"${value:,.2f}"


def render_performance_metrics_section(result) -> None:
    """Compute and display Phase 6 performance statistics for the
    already-completed backtest `result`.

    This calls `calculate_performance_metrics()` fresh each time it
    renders (cheap, pure arithmetic over data already in memory) — it
    does NOT re-run the backtest or the strategy. Reuses the exact same
    BacktestResult already sitting in `st.session_state` from the
    "Backtest Results" section above.
    """
    st.subheader("Performance Metrics (Preview)")
    metrics = calculate_performance_metrics(result)

    if metrics.has_open_position:
        st.caption(
            "ℹ️ An open position was still held at the end of the data — "
            "the figures below include its unrealized value, since that's "
            "already folded into the final portfolio value."
        )

    row1 = st.columns(5)
    row1[0].metric("Total Return", _format_metric(metrics.total_return_pct, "%"))
    row1[1].metric("Absolute P&L", f"${metrics.absolute_pnl:,.2f}")
    row1[2].metric("Win Rate", _format_metric(metrics.win_rate_pct, "%"))
    row1[3].metric("Max Drawdown", _format_metric(metrics.max_drawdown_pct, "%"))
    row1[4].metric("Sharpe Ratio", _format_metric(metrics.sharpe_ratio, decimals=2))

    row2 = st.columns(4)
    row2[0].metric("Number of Trades", metrics.num_trades)
    row2[1].metric("Average Win", _format_dollars(metrics.average_win))
    row2[2].metric("Average Loss", _format_dollars(metrics.average_loss))
    row2[3].metric("Annualized Volatility", _format_metric(metrics.annualized_volatility_pct, "%"))

    st.caption(
        "Sharpe Ratio assumes a 0% annual risk-free rate (a stated simplifying "
        "assumption for this educational simulator, not a real-world rate). "
        "'N/A' means there isn't enough data or completed trades to compute "
        "that metric meaningfully — it is never silently shown as zero."
    )


def main() -> None:
    st.title(APP_NAME)
    st.caption(APP_TAGLINE)
    st.warning(DISCLAIMER, icon="⚠️")

    with st.sidebar:
        st.header("Backtest Setup")
        ticker = st.text_input(
            "Ticker symbol",
            value="AAPL",
            help="A stock ticker like AAPL, MSFT, or TSLA.",
        )
        default_end = date.today()
        default_start = default_end - timedelta(days=365)
        date_range = st.date_input(
            "Date range",
            value=(default_start, default_end),
            help="Historical window to load. Real data comes from yfinance; "
            "if that fails, synthetic sample data is used instead.",
        )
        initial_capital = st.number_input(
            "Initial virtual capital ($)",
            min_value=100.0,
            value=DEFAULT_INITIAL_CAPITAL,
            step=500.0,
            help="Simulated cash only — no real money is ever involved.",
        )
        st.markdown("**Strategy**")
        strategy_name = st.selectbox(
            "Strategy",
            options=["Moving Average Crossover", "RSI", "MACD"],
            label_visibility="collapsed",
            help="Choose which rule generates BUY/SELL/HOLD signals below.",
        )

        strategy = None
        strategy_error = None
        try:
            if strategy_name == "Moving Average Crossover":
                fast_window = st.number_input("Fast SMA window", min_value=1, value=20, step=1)
                slow_window = st.number_input("Slow SMA window", min_value=2, value=50, step=1)
                strategy = MovingAverageCrossoverStrategy(
                    fast_window=int(fast_window), slow_window=int(slow_window)
                )
            elif strategy_name == "RSI":
                rsi_period = st.number_input("RSI period", min_value=1, value=14, step=1)
                oversold = st.number_input("Oversold threshold", min_value=1.0, max_value=98.0, value=30.0, step=1.0)
                overbought = st.number_input("Overbought threshold", min_value=2.0, max_value=99.0, value=70.0, step=1.0)
                strategy = RSIStrategy(period=int(rsi_period), oversold=oversold, overbought=overbought)
            elif strategy_name == "MACD":
                macd_fast = st.number_input("Fast EMA", min_value=1, value=12, step=1)
                macd_slow = st.number_input("Slow EMA", min_value=2, value=26, step=1)
                macd_signal = st.number_input("Signal EMA", min_value=1, value=9, step=1)
                strategy = MACDStrategy(fast=int(macd_fast), slow=int(macd_slow), signal=int(macd_signal))
        except StrategyInputError as exc:
            strategy_error = str(exc)

        run_clicked = st.button(
            "Run Backtest",
            disabled=(strategy is None),
            help="Simulate trades using the selected strategy over the loaded data."
            if strategy is not None
            else "Fix the strategy parameters above first.",
        )

    st.subheader("Project status")
    st.markdown(
        "**Phase 5 is live**: configure a strategy and initial capital on the "
        "left, then click **Run Backtest** to simulate trades over the loaded "
        "data. Performance ratios (Sharpe, drawdown, win rate, CAGR) are still "
        "ahead — this shows the raw simulation only."
    )

    with st.expander("What do these terms mean? (OHLCV, adjusted price, etc.)"):
        st.markdown(
            "- **OHLCV** — Open, High, Low, Close, Volume: the five numbers "
            "that summarize one day of trading for a stock.\n"
            "- **DataFrame** — pandas' table structure; one row per trading "
            "day here, one column per OHLCV field.\n"
            "- **Time series** — data ordered by time, where order matters. "
            "You can't shuffle trading days without breaking the meaning.\n"
            "- **Adjusted vs. raw price** — 'Close' is the literal traded "
            "price that day. 'Adj Close' retroactively accounts for "
            "dividends and stock splits, so it reflects the true return an "
            "investor would have seen over time.\n"
            "- **Why validate input?** — A bad ticker or an impossible date "
            "range doesn't always throw a clear error from the data "
            "provider; sometimes it just comes back empty. Checking first "
            "catches this early with a message you can act on."
        )

    # ---- date_range can be a 1-tuple while the user is mid-selection in
    # the picker (they've clicked a start date but not an end date yet).
    # Guard against that instead of crashing.
    if not isinstance(date_range, tuple) or len(date_range) != 2:
        st.info("Pick both a start and end date in the sidebar to load data.")
        return

    start_date, end_date = date_range

    st.subheader("Market data")

    try:
        result = fetch_market_data(ticker, start_date, end_date)
    except MarketDataValidationError as exc:
        st.error(str(exc))
        return

    if result.source == SOURCE_LIVE:
        st.success(f"Loaded live historical data for **{result.ticker}** via yfinance.", icon="✅")
    else:
        st.warning(
            f"Could not load live data for **{result.ticker}** — showing "
            "**synthetic sample data** instead. This is NOT real market data "
            "for this ticker.",
            icon="⚠️",
        )

    for w in result.warnings:
        st.caption(f"ℹ️ {w}")

    df = result.data
    if df.empty:
        st.error("No usable data available for this request.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows loaded", len(df))
    col2.metric("First date", str(df["Date"].min().date()))
    col3.metric("Last date", str(df["Date"].max().date()))

    st.dataframe(df.head(10), width="stretch")
    st.line_chart(df.set_index("Date")["Adj Close"], width="stretch")
    st.caption(
        "Chart shows **Adj Close** (adjusted for splits/dividends) — see the "
        "explanation above for why that's usually the more meaningful series."
    )

    render_indicators_section(df)

    if strategy_error:
        st.error(f"Could not configure strategy: {strategy_error}")
    elif strategy is not None:
        render_strategy_signals_section(df, strategy)

    if run_clicked and strategy is not None:
        try:
            bt_result = run_backtest(df, strategy, initial_capital=float(initial_capital))
            st.session_state["backtest_result"] = bt_result
            st.session_state["backtest_context"] = {
                "strategy_name": strategy.name,
                "strategy_params": strategy.describe()["params"],
                "ticker": result.ticker,
                "start_date": start_date,
                "end_date": end_date,
                "initial_capital": float(initial_capital),
            }
        except (BacktestInputError, StrategyInputError) as exc:
            st.error(f"Backtest failed: {exc}")

    if "backtest_result" in st.session_state:
        render_backtest_section(st.session_state["backtest_result"], st.session_state["backtest_context"])
        render_performance_metrics_section(st.session_state["backtest_result"])


if __name__ == "__main__":
    main()
