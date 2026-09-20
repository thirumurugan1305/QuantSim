"""
Chart-building functions for QuantSim's Streamlit UI (Phase 8).

WHAT THIS MODULE DOES
----------------------
Every function here takes plain data (a pandas DataFrame/Series, a list
of `Trade` objects) and returns a `plotly.graph_objects.Figure` — nothing
more. There is NO `import streamlit` anywhere in this file, and no
function here reads or writes `st.session_state`. That separation is
deliberate and mirrors every other phase's own pattern (pure calculation
in `src/`, Streamlit orchestration in `app.py`): a chart-building
function should be testable by simply calling it with known data and
checking the `Figure` it returns, without ever starting the Streamlit
app. See `tests/test_charts.py` for exactly that.

WHY PLOTLY HERE, WHEN Phases 1-7 USED st.line_chart/st.bar_chart
---------------------------------------------------------------------
Streamlit's native charts are quick to write but can't do a few things
this phase specifically needs: placing a marker at an exact (date, price)
point (needed for BUY/SELL markers on the price chart), custom hover
text, or a horizontal reference line (needed for RSI's 30/70 bands and
MACD's zero line). `plotly` has been listed in `requirements.txt` since
Phase 1 for exactly this eventual use — Phase 8 is the first phase that
actually imports it.

AVOIDING MISLEADING CHARTS
------------------------------
- Every figure sets an explicit axis title with units (e.g. "Portfolio
  Value ($)", never a bare unlabeled number line).
- BUY/SELL markers are placed at the EXACT execution price and date
  already computed by the backtesting engine (`Trade.entry_price`/
  `entry_date`, `.exit_price`/`exit_date`) -- never a smoothed, offset,
  or estimated position.
- The drawdown chart is mathematically guaranteed <= 0 (it's a decline
  from a running peak) and is titled accordingly, so it can't be
  misread as a second equity curve.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from src.backtesting.models import Trade

# A small, consistent palette used across every chart in this module, so
# "green means up/buy" and "red means down/sell" stay consistent no
# matter which chart the user is looking at.
_COLOR_PRICE = "#1f77b4"
_COLOR_BUY = "#2ca02c"
_COLOR_SELL = "#d62728"
_COLOR_DRAWDOWN = "#d62728"
_COLOR_NEUTRAL = "#7f7f7f"


def _base_layout(title: str, y_title: str, x_title: str = "Date") -> dict:
    """Shared layout settings so every chart in this module looks
    consistent (same margins, hover behavior, legend placement) without
    repeating the same dict literal in every function."""
    return dict(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=50, b=40),
        height=380,
    )


def build_price_chart(
    price_df: pd.DataFrame,
    price_column: str = "Adj Close",
    sma_series: dict[str, pd.Series] | None = None,
    trades: list[Trade] | None = None,
    title: str = "Price",
) -> go.Figure:
    """Price line, with optional SMA overlays and optional BUY/SELL
    trade markers.

    Parameters
    ----------
    price_df : DataFrame with a "Date" column and `price_column`.
    sma_series : optional dict of {label: pd.Series}, aligned to
        `price_df`'s row order (e.g. {"SMA 20": ..., "SMA 50": ...}) --
        used by the Technical Indicators preview.
    trades : optional list of completed Trade objects -- used by the
        Backtest Results view to show exactly where each BUY/SELL
        executed. `entry_date`/`entry_price` become a green upward
        triangle; `exit_date`/`exit_price` become a red downward
        triangle. Reusing the same function for both call sites (with
        the parameter that isn't needed left as None) avoids having two
        near-identical price-chart functions.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=price_df["Date"],
            y=price_df[price_column],
            mode="lines",
            name=price_column,
            line=dict(color=_COLOR_PRICE, width=1.75),
            hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.2f}<extra>" + price_column + "</extra>",
        )
    )

    if sma_series:
        for label, series in sma_series.items():
            fig.add_trace(
                go.Scatter(
                    x=price_df["Date"],
                    y=series,
                    mode="lines",
                    name=label,
                    line=dict(width=1.25, dash="dot"),
                    hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.2f}<extra>" + label + "</extra>",
                )
            )

    if trades:
        buy_x = [t.entry_date for t in trades]
        buy_y = [t.entry_price for t in trades]
        sell_x = [t.exit_date for t in trades]
        sell_y = [t.exit_price for t in trades]

        fig.add_trace(
            go.Scatter(
                x=buy_x,
                y=buy_y,
                mode="markers",
                name="BUY",
                marker=dict(symbol="triangle-up", size=12, color=_COLOR_BUY, line=dict(width=1, color="white")),
                hovertemplate="BUY<br>%{x|%Y-%m-%d}<br>$%{y:,.2f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=sell_x,
                y=sell_y,
                mode="markers",
                name="SELL",
                marker=dict(symbol="triangle-down", size=12, color=_COLOR_SELL, line=dict(width=1, color="white")),
                hovertemplate="SELL<br>%{x|%Y-%m-%d}<br>$%{y:,.2f}<extra></extra>",
            )
        )

    fig.update_layout(**_base_layout(title, f"{price_column} ($)"))
    return fig


def build_rsi_chart(
    dates: pd.Series,
    rsi: pd.Series,
    oversold: float = 30.0,
    overbought: float = 70.0,
    period: int = 14,
) -> go.Figure:
    """RSI line with shaded oversold/overbought reference bands.

    The reference lines are drawn at whatever thresholds are actually
    configured (not hardcoded 30/70), so this stays correct if a
    strategy's RSI thresholds are ever customized.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=rsi,
            mode="lines",
            name=f"RSI({period})",
            line=dict(color=_COLOR_PRICE, width=1.5),
            hovertemplate="%{x|%Y-%m-%d}<br>RSI: %{y:.1f}<extra></extra>",
        )
    )
    fig.add_hline(y=overbought, line=dict(color=_COLOR_SELL, dash="dash", width=1), annotation_text="Overbought")
    fig.add_hline(y=oversold, line=dict(color=_COLOR_BUY, dash="dash", width=1), annotation_text="Oversold")
    fig.update_yaxes(range=[0, 100])
    fig.update_layout(**_base_layout(f"RSI({period})", "RSI"))
    return fig


def build_macd_chart(dates: pd.Series, macd_df: pd.DataFrame) -> go.Figure:
    """MACD line, signal line, and histogram, with a zero reference line.

    `macd_df` must have the columns produced by
    `calculate_macd()`: 'macd_line', 'signal_line', 'histogram'.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=dates,
            y=macd_df["histogram"],
            name="Histogram",
            marker=dict(color=_COLOR_NEUTRAL, opacity=0.5),
            hovertemplate="%{x|%Y-%m-%d}<br>Histogram: %{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=macd_df["macd_line"],
            mode="lines",
            name="MACD line",
            line=dict(color=_COLOR_PRICE, width=1.5),
            hovertemplate="%{x|%Y-%m-%d}<br>MACD: %{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=macd_df["signal_line"],
            mode="lines",
            name="Signal line",
            line=dict(color=_COLOR_SELL, width=1.5),
            hovertemplate="%{x|%Y-%m-%d}<br>Signal: %{y:.3f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line=dict(color=_COLOR_NEUTRAL, width=1))
    fig.update_layout(**_base_layout("MACD", "Value"))
    return fig


def build_equity_curve_chart(equity_curve: pd.DataFrame) -> go.Figure:
    """Portfolio value over time, with cash/holdings shown on hover.

    `equity_curve` must have the columns BacktestEngine always produces:
    date, cash, shares_held, holdings_value, total_value (see
    `EQUITY_CURVE_COLUMNS` in `src/backtesting/models.py`).
    """
    fig = go.Figure()
    customdata = equity_curve[["cash", "holdings_value", "shares_held"]].to_numpy()
    fig.add_trace(
        go.Scatter(
            x=equity_curve["date"],
            y=equity_curve["total_value"],
            mode="lines",
            name="Portfolio Value",
            line=dict(color=_COLOR_PRICE, width=2),
            fill="tozeroy",
            fillcolor="rgba(31, 119, 180, 0.08)",
            customdata=customdata,
            hovertemplate=(
                "%{x|%Y-%m-%d}<br>"
                "Total: $%{y:,.2f}<br>"
                "Cash: $%{customdata[0]:,.2f}<br>"
                "Holdings: $%{customdata[1]:,.2f} (%{customdata[2]:.0f} shares)"
                "<extra></extra>"
            ),
        )
    )
    fig.update_layout(**_base_layout("Portfolio Value Over Time", "Portfolio Value ($)"))
    return fig


def _drawdown_series_pct(equity_curve: pd.DataFrame) -> pd.Series:
    """Running decline from the highest prior portfolio value, as a
    percentage (always <= 0).

    This is the SAME formula `PerformanceMetrics.max_drawdown_pct` uses
    (see `_max_drawdown_pct` in `src/analytics/performance_metrics.py`)
    -- that function reduces it to a single worst-case number; this one
    keeps every bar's value so it can be charted. Kept as a small,
    self-contained function here (rather than imported from the
    analytics module) so `src/analytics/performance_metrics.py` stays
    completely untouched by this UI-focused phase, per the approved
    Phase 8 scope.
    """
    values = equity_curve["total_value"].astype(float)
    running_max = values.cummax()
    valid = running_max > 0
    drawdown = pd.Series(0.0, index=values.index)
    drawdown[valid] = (values[valid] - running_max[valid]) / running_max[valid] * 100
    return drawdown


def build_drawdown_chart(equity_curve: pd.DataFrame) -> go.Figure:
    """Drawdown-from-peak over time, as a percentage. Always <= 0."""
    drawdown = _drawdown_series_pct(equity_curve)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=equity_curve["date"],
            y=drawdown,
            mode="lines",
            name="Drawdown",
            line=dict(color=_COLOR_DRAWDOWN, width=1.5),
            fill="tozeroy",
            fillcolor="rgba(214, 39, 40, 0.15)",
            hovertemplate="%{x|%Y-%m-%d}<br>Drawdown: %{y:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(**_base_layout("Drawdown From Peak", "Drawdown (%)"))
    return fig


def build_signal_pulse_chart(dates: pd.Series, signal_numeric: pd.Series) -> go.Figure:
    """+1 on BUY, -1 on SELL, 0 on HOLD, over time -- a lightweight view
    of when a strategy's raw rule fired, independent of whether a trade
    was actually executed (that's what `build_price_chart`'s markers
    show instead, using ACTUAL executed trades)."""
    fig = go.Figure()
    colors = [_COLOR_BUY if v > 0 else _COLOR_SELL if v < 0 else _COLOR_NEUTRAL for v in signal_numeric]
    fig.add_trace(
        go.Bar(
            x=dates,
            y=signal_numeric,
            marker=dict(color=colors),
            name="Signal",
            hovertemplate="%{x|%Y-%m-%d}<br>Signal: %{y}<extra></extra>",
        )
    )
    fig.update_layout(**_base_layout("Strategy Signal Over Time", "Signal (+1 BUY / -1 SELL / 0 HOLD)"))
    fig.update_layout(showlegend=False)
    return fig
