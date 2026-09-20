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

Phase 8 scope (this file, today):
    - The single long vertical page is reorganized into 4 tabs: "Market
      & Signals", "Backtest Results", "Performance", and "Saved
      Backtests" — the same sections that existed before, regrouped so
      the page reads as a dashboard with distinct stages rather than one
      long scroll.
    - Charts that benefit from it now use the new pure Plotly builders in
      `src/ui/charts.py` (price + SMA overlay + BUY/SELL trade markers,
      RSI with oversold/overbought bands, MACD with a zero line, the
      equity curve with a cash/holdings hover breakdown, and a new
      drawdown-from-peak chart) instead of Streamlit's native
      line/bar charts. Every one of these charts is built from data
      Phases 2-7 already compute — no new calculation, only a richer
      way to look at the same numbers.
    - `st.spinner(...)` now wraps the two genuinely slow operations
      (fetching market data, running a backtest) so a slow yfinance
      call or a large backtest visibly shows it's working.
    - The sidebar's inputs are unchanged but now grouped under small
      section labels (Market Data / Strategy / Backtest) instead of one
      flat list.

No business logic changed in this phase: every `render_*` function
still calls the exact same Phase 2-7 functions it always did
(`fetch_market_data`, `run_backtest`, `calculate_performance_metrics`,
`save_backtest_result`, etc.) — only how results are laid out and
charted has changed.
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
from src.database.repository import (
    delete_backtest_result,
    init_db,
    list_saved_backtests,
    load_backtest_result,
    save_backtest_result,
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
from src.ui.charts import (
    build_drawdown_chart,
    build_equity_curve_chart,
    build_macd_chart,
    build_price_chart,
    build_rsi_chart,
    build_signal_pulse_chart,
)

st.set_page_config(page_title=APP_NAME, layout="wide")

# Safe to call every run: init_db() only creates what's missing
# (CREATE TABLE IF NOT EXISTS) and never touches existing data.
init_db()

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
    sma_20 = calculate_sma(price, 20)
    sma_long = calculate_sma(price, _SMA_LONG_WINDOW)
    price_fig = build_price_chart(
        df,
        price_column="Adj Close",
        sma_series={"SMA 20": sma_20, f"SMA {_SMA_LONG_WINDOW}": sma_long},
        title="Price with Moving Averages",
    )
    st.plotly_chart(price_fig, width="stretch")

    # --- RSI -------------------------------------------------------------
    st.markdown(f"**RSI ({_RSI_PERIOD})**")
    if n_rows < _RSI_PERIOD + 1:
        st.info(f"Need at least {_RSI_PERIOD + 1} rows for RSI({_RSI_PERIOD}); only {n_rows} available.")
    else:
        rsi = calculate_rsi(price, period=_RSI_PERIOD)
        st.plotly_chart(build_rsi_chart(df["Date"], rsi, period=_RSI_PERIOD), width="stretch")
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
        st.plotly_chart(build_macd_chart(df["Date"], macd_df), width="stretch")
        st.caption(
            "MACD line vs. signal line, with the histogram (their difference) "
            "shaded below — the gap most 'MACD crossover' strategies watch."
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
    # 0 on HOLD -- a lightweight view of when the strategy's raw rule
    # fired (distinct from the "Backtest Results" tab's price chart,
    # which marks where trades were actually EXECUTED).
    signal_numeric = signals.map({Signal.BUY: 1, Signal.SELL: -1, Signal.HOLD: 0})
    st.plotly_chart(build_signal_pulse_chart(df["Date"], signal_numeric), width="stretch")
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


def render_backtest_section(result, context: dict, price_df: pd.DataFrame | None = None) -> None:
    """Display a completed BacktestResult: final value, trade count, a
    price chart with BUY/SELL markers, the equity curve, a drawdown
    chart, and a trade table.

    PREVIEW ONLY: deliberately does not compute or show Sharpe ratio,
    max drawdown (as a number), win rate, or CAGR — those are Phase 6's
    "Performance Metrics" section. This section only reports the raw
    simulation output, the same way `render_strategy_signals_section`
    only reports raw signals rather than judging them.

    `price_df` (the same market-data DataFrame already loaded on the
    "Market & Signals" tab) is optional so a LOADED (from the database)
    backtest can still render its equity curve and trade table even
    when the original price series isn't currently loaded in this
    session — only the price-chart-with-markers is skipped in that case.
    """
    st.subheader("Backtest Results")
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

    if price_df is not None and not price_df.empty:
        st.markdown("**Price with executed trades**")
        st.plotly_chart(
            build_price_chart(price_df, trades=result.trades, title="Price — BUY/SELL Markers"),
            width="stretch",
        )
        st.caption(
            "▲ green = BUY, ▼ red = SELL, at the exact price and date each trade "
            "actually executed."
        )

    if result.equity_curve.empty:
        st.info("No bars were processed for this data — nothing to chart.")
        return

    chart_col, drawdown_col = st.columns(2)
    with chart_col:
        st.markdown("**Portfolio value over time**")
        st.plotly_chart(build_equity_curve_chart(result.equity_curve), width="stretch")
        st.caption("Total portfolio value (cash + holdings) at the close of every bar.")
    with drawdown_col:
        st.markdown("**Drawdown from peak**")
        st.plotly_chart(build_drawdown_chart(result.equity_curve), width="stretch")
        st.caption("How far below its highest-ever value the portfolio has fallen, over time.")

    if result.trades:
        st.markdown("**Completed trades**")
        trades_df = pd.DataFrame(
            [
                {
                    "Entry Date": t.entry_date.date(),
                    "Entry Price": t.entry_price,
                    "Exit Date": t.exit_date.date(),
                    "Exit Price": t.exit_price,
                    "Quantity": t.quantity,
                    "P&L ($)": t.pnl,
                    "P&L (%)": t.pnl_pct,
                }
                for t in result.trades
            ]
        )
        st.dataframe(
            trades_df,
            width="stretch",
            hide_index=True,
            column_config={
                "Entry Price": st.column_config.NumberColumn(format="$%.2f"),
                "Exit Price": st.column_config.NumberColumn(format="$%.2f"),
                "P&L ($)": st.column_config.NumberColumn(format="$%.2f"),
                "P&L (%)": st.column_config.NumberColumn(format="%.2f%%"),
            },
        )
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
    st.subheader("Performance Metrics")
    metrics = calculate_performance_metrics(result)

    if metrics.has_open_position:
        st.caption(
            "ℹ️ An open position was still held at the end of the data — "
            "the figures below include its unrealized value, since that's "
            "already folded into the final portfolio value."
        )

    st.markdown("**Returns**")
    returns_row = st.columns(2)
    returns_row[0].metric("Total Return", _format_metric(metrics.total_return_pct, "%"))
    returns_row[1].metric("Absolute P&L", f"${metrics.absolute_pnl:,.2f}")

    st.markdown("**Risk**")
    risk_row = st.columns(3)
    risk_row[0].metric("Max Drawdown", _format_metric(metrics.max_drawdown_pct, "%"))
    risk_row[1].metric("Sharpe Ratio", _format_metric(metrics.sharpe_ratio, decimals=2))
    risk_row[2].metric("Annualized Volatility", _format_metric(metrics.annualized_volatility_pct, "%"))

    st.markdown("**Trades**")
    trades_row = st.columns(4)
    trades_row[0].metric("Number of Trades", metrics.num_trades)
    trades_row[1].metric("Win Rate", _format_metric(metrics.win_rate_pct, "%"))
    trades_row[2].metric("Average Win", _format_dollars(metrics.average_win))
    trades_row[3].metric("Average Loss", _format_dollars(metrics.average_loss))

    st.caption(
        "Sharpe Ratio assumes a 0% annual risk-free rate (a stated simplifying "
        "assumption for this educational simulator, not a real-world rate). "
        "'N/A' means there isn't enough data or completed trades to compute "
        "that metric meaningfully — it is never silently shown as zero."
    )


def _format_saved_backtest_label(row: dict) -> str:
    return_str = f"{row['total_return_pct']:.2f}%" if row["total_return_pct"] is not None else "N/A"
    return f"#{row['id']} — {row['ticker']} — {row['strategy_name']} — {return_str} — saved {row['created_at'][:19]}"


def render_save_load_section() -> None:
    """Save the current backtest to SQLite, and browse/load/delete
    previously saved ones.

    Reuses the SAME `st.session_state["backtest_result"]` /
    `["backtest_context"]` slots a live "Run Backtest" click populates —
    loading a saved backtest writes into those exact slots, so
    `render_backtest_section()` and `render_performance_metrics_section()`
    render it with no new branching logic, whether the result just came
    from a live run or from the database.
    """
    st.subheader("Save & Load Backtests")

    has_result = "backtest_result" in st.session_state
    already_saved = st.session_state.get("backtest_result_saved", False)

    save_col, status_col = st.columns([1, 3])
    with save_col:
        if st.button(
            "Save Backtest",
            disabled=(not has_result) or already_saved,
            help="Run a backtest first." if not has_result else None,
        ):
            result = st.session_state["backtest_result"]
            context = st.session_state["backtest_context"]
            # Reuses the exact same Phase 6 function used for on-screen
            # display -- a pure, cheap recalculation from data already in
            # memory, NOT a duplicated formula. The database layer itself
            # never computes metrics; it only stores what's passed in here.
            metrics = calculate_performance_metrics(result)
            new_id = save_backtest_result(
                result=result,
                metrics=metrics,
                ticker=context["ticker"],
                start_date=context["start_date"],
                end_date=context["end_date"],
            )
            st.session_state["backtest_result_saved"] = True
            st.session_state["last_saved_id"] = new_id
            # Rerun immediately so the button's `disabled=` (computed from
            # `already_saved` above, necessarily BEFORE this click's
            # result is known) reflects the just-saved state right away
            # -- without this, the button would still render as clickable
            # for one extra render, and a second real click in that
            # window could create a genuine duplicate row.
            st.rerun()

    with status_col:
        # Re-read fresh here, AFTER the button block above may have just
        # updated it in this same script run -- reusing the `already_saved`
        # local captured before the click would show a stale value and
        # miss the success message on the very rerun the save happened.
        if st.session_state.get("backtest_result_saved", False) and "last_saved_id" in st.session_state:
            st.success(f"Saved as Backtest #{st.session_state['last_saved_id']}.")
        elif not has_result:
            st.caption("Run a backtest above, then save it here.")

    with st.expander("Saved Backtests", expanded=False):
        saved = list_saved_backtests()

        if not saved:
            st.info("No saved backtests yet.")
            return

        table_df = pd.DataFrame(
            [
                {
                    "ID": row["id"],
                    "Saved": row["created_at"][:19],
                    "Ticker": row["ticker"],
                    "Strategy": row["strategy_name"],
                    "Date Range": f"{row['start_date']} → {row['end_date']}",
                    "Total Return": f"{row['total_return_pct']:.2f}%" if row["total_return_pct"] is not None else "N/A",
                }
                for row in saved
            ]
        )
        st.dataframe(table_df, width="stretch", hide_index=True)

        labels = {_format_saved_backtest_label(row): row for row in saved}
        selected_label = st.selectbox("Select a saved backtest", options=list(labels.keys()))
        selected_row = labels[selected_label]
        selected_id = selected_row["id"]

        load_col, delete_col = st.columns(2)

        with load_col:
            if st.button("Load Selected"):
                loaded_result = load_backtest_result(selected_id)
                if loaded_result is None:
                    st.error(f"Backtest #{selected_id} no longer exists (it may have just been deleted).")
                else:
                    st.session_state["backtest_result"] = loaded_result
                    st.session_state["backtest_context"] = {
                        "strategy_name": loaded_result.strategy_name,
                        "strategy_params": loaded_result.strategy_params,
                        "ticker": selected_row["ticker"],
                        "start_date": selected_row["start_date"],
                        "end_date": selected_row["end_date"],
                        "initial_capital": loaded_result.initial_capital,
                    }
                    # Already in the database -- mark as saved so the
                    # Save button doesn't invite creating a duplicate row.
                    st.session_state["backtest_result_saved"] = True
                    st.session_state["last_saved_id"] = selected_id
                    st.rerun()

        with delete_col:
            # Two-step confirmation, since Streamlit has no native
            # confirm dialog: the first click only arms a pending delete
            # for THIS specific id; a second, distinct click confirms it.
            pending_id = st.session_state.get("pending_delete_id")
            if pending_id == selected_id:
                st.warning(f"Delete Backtest #{selected_id}? This cannot be undone.")
                if st.button("Confirm Delete", type="primary"):
                    delete_backtest_result(selected_id)
                    st.session_state["pending_delete_id"] = None
                    # Deleting a backtest that happens to be the one
                    # currently loaded in memory does not affect the
                    # in-memory result at all -- it stays visible above
                    # until a new run or load replaces it; only a later
                    # attempt to re-load this same id would find it gone.
                    st.rerun()
                if st.button("Cancel"):
                    st.session_state["pending_delete_id"] = None
                    st.rerun()
            else:
                if st.button("Delete Selected"):
                    st.session_state["pending_delete_id"] = selected_id
                    st.rerun()


def main() -> None:
    st.title(APP_NAME)
    st.caption(APP_TAGLINE)
    st.warning(DISCLAIMER, icon="⚠️")

    with st.sidebar:
        st.markdown("### 📊 Market Data")
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

        st.divider()
        st.markdown("### ⚙️ Strategy")
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

        st.divider()
        st.markdown("### 💰 Backtest")
        initial_capital = st.number_input(
            "Initial virtual capital ($)",
            min_value=100.0,
            value=DEFAULT_INITIAL_CAPITAL,
            step=500.0,
            help="Simulated cash only — no real money is ever involved.",
        )
        run_clicked = st.button(
            "Run Backtest",
            disabled=(strategy is None),
            width="stretch",
            help="Simulate trades using the selected strategy over the loaded data."
            if strategy is not None
            else "Fix the strategy parameters above first.",
        )

    # ---- date_range can be a 1-tuple while the user is mid-selection in
    # the picker (they've clicked a start date but not an end date yet).
    # Guard against that instead of crashing. This check (and every
    # validation/computation step below) happens BEFORE the tabs are
    # built, so a failure here shows one clear message instead of an
    # empty or broken-looking set of tabs.
    if not isinstance(date_range, tuple) or len(date_range) != 2:
        st.info("Pick both a start and end date in the sidebar to load data.")
        return

    start_date, end_date = date_range

    try:
        with st.spinner(f"Fetching market data for {ticker}..."):
            result = fetch_market_data(ticker, start_date, end_date)
    except MarketDataValidationError as exc:
        st.error(str(exc))
        return

    df = result.data
    if df.empty:
        st.error("No usable data available for this request.")
        return

    if run_clicked and strategy is not None:
        try:
            with st.spinner("Running backtest..."):
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
            # A fresh live run is a NEW, not-yet-saved result, even if a
            # previous result (live or loaded) had already been saved.
            st.session_state["backtest_result_saved"] = False
            st.session_state.pop("last_saved_id", None)
        except (BacktestInputError, StrategyInputError) as exc:
            st.error(f"Backtest failed: {exc}")

    tab_market, tab_backtest, tab_performance, tab_saved = st.tabs(
        ["📊 Market & Signals", "📈 Backtest Results", "🎯 Performance", "💾 Saved Backtests"]
    )

    with tab_market:
        st.info(
            "👋 **New here?** Pick a ticker and date range in the sidebar, choose a "
            "strategy, then switch to the **Backtest Results** tab and click "
            "**Run Backtest**.",
            icon="👋",
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

        st.subheader("Market Data")

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

        col1, col2, col3 = st.columns(3)
        col1.metric("Rows loaded", len(df))
        col2.metric("First date", str(df["Date"].min().date()))
        col3.metric("Last date", str(df["Date"].max().date()))

        st.dataframe(df.head(10), width="stretch", hide_index=True)
        st.plotly_chart(build_price_chart(df, title=f"{result.ticker} — Adj Close"), width="stretch")
        st.caption(
            "Chart shows **Adj Close** (adjusted for splits/dividends) — see the "
            "explanation above for why that's usually the more meaningful series."
        )

        render_indicators_section(df)

        if strategy_error:
            st.error(f"Could not configure strategy: {strategy_error}")
        elif strategy is not None:
            render_strategy_signals_section(df, strategy)

    with tab_backtest:
        if "backtest_result" in st.session_state:
            bt_context = st.session_state["backtest_context"]
            # Only overlay the currently-loaded market data on the price
            # chart if it actually corresponds to the backtest being
            # displayed -- e.g. after loading a SAVED backtest for a
            # different ticker/date range than what's currently in the
            # sidebar, showing today's price series with that backtest's
            # trade markers would overlay trades onto the WRONG price
            # data. In that mismatch case, the price-with-markers chart
            # is simply skipped (render_backtest_section handles
            # `price_df=None` gracefully) -- the equity curve and trade
            # table still render either way, since those come from the
            # stored result itself, not from today's market data fetch.
            price_df_matches = (
                bt_context.get("ticker") == result.ticker
                and bt_context.get("start_date") == start_date
                and bt_context.get("end_date") == end_date
            )
            render_backtest_section(
                st.session_state["backtest_result"],
                bt_context,
                price_df=df if price_df_matches else None,
            )
        else:
            st.info(
                "No backtest has been run yet. Configure a strategy and click "
                "**Run Backtest** in the sidebar, or load a previous one from "
                "the **Saved Backtests** tab."
            )

    with tab_performance:
        if "backtest_result" in st.session_state:
            render_performance_metrics_section(st.session_state["backtest_result"])
        else:
            st.info(
                "No backtest has been run yet — performance statistics will "
                "appear here once one has."
            )

    with tab_saved:
        render_save_load_section()


if __name__ == "__main__":
    main()
