# QuantSim Architecture

This document describes how QuantSim is structured and why, for anyone
reading the codebase after the fact (including future-you). It
describes the system as actually built through Phase 8, not an
idealized or planned version of it.

## Layered structure

QuantSim is organized as a strict pipeline of layers, each one only
depending on the layer(s) below it:

```
┌─────────────────────────────────────────────────────────┐
│ app.py            Streamlit UI: layout, session state,   │
│                    orchestration only — no calculations   │
├─────────────────────────────────────────────────────────┤
│ src/ui/            Pure Plotly chart-building functions   │
│                    (data in, Figure out — no Streamlit)   │
├─────────────────────────────────────────────────────────┤
│ src/database/      SQLite persistence: save/load/list/    │
│                    delete a completed backtest             │
├─────────────────────────────────────────────────────────┤
│ src/analytics/     Performance metrics, computed from a    │
│                    BacktestResult only                     │
├─────────────────────────────────────────────────────────┤
│ src/backtesting/   Simulates trades from a signal series   │
├─────────────────────────────────────────────────────────┤
│ src/strategies/    Turns indicators into BUY/SELL/HOLD     │
├─────────────────────────────────────────────────────────┤
│ src/indicators/    SMA, EMA, RSI, MACD — pure math          │
├─────────────────────────────────────────────────────────┤
│ src/data/          Historical OHLCV data (yfinance +        │
│                    synthetic fallback)                      │
├─────────────────────────────────────────────────────────┤
│ src/config.py      Shared constants (paths, defaults)       │
└─────────────────────────────────────────────────────────┘
```

Each arrow points one direction only. `src/indicators/` has no idea
`src/strategies/` exists; `src/backtesting/` has no idea any particular
strategy exists — it only knows about a plain `Signal` series. This is
enforced by convention, not by any framework, and is checked by the
test suite's integration tests (`tests/test_integration.py`), which run
a real strategy through a real backtest through real metrics and
persistence to confirm the seams actually hold together.

## The core design pattern, repeated at every layer

Every layer below the UI follows the same shape: **pure calculation
functions that take plain data in and return plain data out**, with no
side effects, no I/O (except `src/data/` and `src/database/`, whose
whole job IS I/O), and no dependency on Streamlit. This is why:

- `src/indicators/technical_indicators.py` can be tested with a
  hand-built `pandas.Series` and no market data connection.
- `src/strategies/*` can be tested with a hand-built price DataFrame and
  no real indicator computation needed first.
- `src/backtesting/engine.py`'s `BacktestEngine.run()` takes a `Signal`
  series directly — it can be tested with a hand-built signal list and
  never needs to invoke a real strategy at all.
- `src/analytics/performance_metrics.py` takes a `BacktestResult` and
  never re-runs a backtest.
- `src/ui/charts.py` takes a DataFrame/list of trades and returns a
  `plotly.graph_objects.Figure` — it can be tested with no Streamlit
  process running at all.

`app.py` is the one place that ties all of this together into an
interactive page — it calls each layer's public functions in sequence
and renders their output, but contains no calculation logic of its own.

## Data structures

- **`Signal`** (`src/strategies/base_strategy.py`) — an `Enum` (also a
  `str` subclass) with exactly three values: `BUY`, `SELL`, `HOLD`. This
  is the single "language" every strategy speaks and the backtesting
  engine understands — the engine has no knowledge of which strategy
  produced a given signal.
- **`Trade`, `OpenPosition`, `PortfolioSnapshot`, `BacktestResult`**
  (`src/backtesting/models.py`) — plain dataclasses describing what a
  backtest produced. `Trade.pnl`, `.pnl_pct`, and `.is_win` are
  `@property` computations, never stored fields — this matters for the
  database layer (see below).
- **`PerformanceMetrics`** (`src/analytics/performance_metrics.py`) — a
  dataclass of statistics computed from a `BacktestResult`. Fields that
  aren't meaningful for the given data (e.g. no completed trades) are
  `None`, never `NaN`/`inf` and never a misleading `0`.

## Backtesting assumptions (deliberate, not accidental)

Documented in full in `src/backtesting/engine.py`'s module docstring;
summarized here:

- **Long-only, all-in/all-out.** A BUY spends all available cash on
  whole shares; a SELL sells the entire position. No shorting, no
  partial sizing, no fractional shares.
- **Same-bar execution.** A signal at bar *t* executes at bar *t*'s own
  closing price — an idealization of instantaneous reaction, not a
  look-ahead bias (the engine never reads a future row).
- **No transaction costs by default** (`transaction_cost_pct=0.0`), but
  the parameter is fully functional if set to something else.
- **An open position at the end is never force-closed** — it's reported
  separately (`BacktestResult.open_position`) from completed `trades`.
- **Repeated or inapplicable signals are silently no-ops**: a second BUY
  while already holding does nothing; a SELL while flat does nothing.

## Performance metrics assumptions

Documented in full in `src/analytics/performance_metrics.py`; summarized:

- **Sharpe Ratio assumes a 0% annual risk-free rate by default** —
  configurable via a parameter, but never silently defaulted to a
  specific "real" rate, since this project has no live rate source.
- The risk-free rate is treated as **annual** and converted to daily by
  simple division (`rate / TRADING_DAYS_PER_YEAR`), not compounding.
- **Trade-based statistics (win rate, average win/loss) only ever look
  at completed trades** — an open position never counts as a trade, but
  its unrealized value IS already folded into `final_value` by the
  backtesting engine, so Total Return correctly reflects it.

## Database schema (Phase 7)

Four tables in a plain SQLite file (`quantsim.db`, gitignored), created
by `src/database/repository.py`'s `init_db()`:

- **`backtests`** — one row per saved run: ticker, dates, strategy
  name/params (JSON), initial/final capital, and any open-position
  fields embedded directly (nullable) rather than in a separate table,
  since there's at most one open position per backtest.
- **`trades`** — one row per completed `Trade`, `ON DELETE CASCADE`
  from `backtests`. `pnl`/`pnl_pct`/`is_win` are **not** stored columns
  — they're recomputed after loading from the same raw numbers `Trade`
  itself uses, so a stored value can never drift from that logic.
- **`portfolio_snapshots`** — one row per bar of the equity curve,
  `ON DELETE CASCADE`.
- **`performance_metrics`** — one row per backtest (`backtest_id` is
  both the foreign key and the primary key, enforcing a 1:1
  relationship), storing a `PerformanceMetrics` snapshot taken at save
  time.

`PRAGMA foreign_keys = ON` is set on every connection inside the single
shared `get_connection()` helper, so cascading deletes can never be
silently disabled by a call site forgetting to enable it (SQLite
disables foreign key enforcement by default).

**Metrics: stored, not recalculated on load** — a deliberate hybrid.
`list_saved_backtests()` reads the stored `performance_metrics` row
directly for fast list views; loading one specific backtest reconstructs
a real `BacktestResult` that can be fed back into
`calculate_performance_metrics()` at any time to recompute fresh
numbers, which is exactly what `tests/test_integration.py` does to
confirm the two never disagree.

## UI layer (Phase 8)

`app.py` is a single Streamlit page organized into four tabs (Market &
Signals, Backtest Results, Performance, Saved Backtests). All
computation (fetching data, running a strategy, running a backtest)
happens once, in a fixed order, before the tabs are built — only the
*display* of results is routed into tabs, so a button click deep inside
one tab never needs to worry about what tab is currently active.

`src/ui/charts.py` holds every chart-building function, kept
deliberately free of Streamlit imports so each one is testable by
calling it directly and inspecting the `Figure` it returns — see
`tests/test_charts.py`.

## Testing philosophy

- **Unit tests per layer** (`test_indicators.py`, `test_strategies.py`,
  `test_backtesting.py`, `test_performance_metrics.py`,
  `test_database.py`, `test_charts.py`) use small, hand-built,
  independently-verifiable fixtures — numbers you can check by hand,
  not just "did it run."
- **Integration tests** (`test_integration.py`) run real strategies
  through a real backtest, through real metrics, through a real
  save/load round trip — using the project's own bundled sample data,
  never synthetic stand-ins — to catch bugs that only appear where two
  layers' real output meets, not where two independently hand-built
  fixtures happen to agree.
- **UI regression tests** (`test_app.py`) use `streamlit.testing.v1.AppTest`
  to click through the actual Run → Save → Load → Delete workflow
  against an isolated temporary database, so a future change that
  silently breaks the app's wiring gets caught automatically instead of
  requiring another manual verification pass.
- **Look-ahead-bias tests** appear at both the strategy and backtesting
  layers: they compute results on a truncated dataset and confirm
  identical historical output to the full dataset, proving neither layer
  can see into the future by construction, not just by inspection.
