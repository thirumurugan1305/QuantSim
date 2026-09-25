# QuantSim Learning Guide

QuantSim was built incrementally, one phase at a time, specifically so
each phase could be understood before the next one was added. This
guide pulls the concepts explained along the way into one place. It
assumes no prior background in trading, pandas, or backtesting.

## Data concepts (Phase 2)

- **OHLCV** — Open, High, Low, Close, Volume: the five numbers that
  describe one trading day (or other period) for a security.
- **DataFrame** — pandas' table structure: one row per observation (here,
  one row per trading day), one column per variable.
- **Time series** — data ordered by time, where order matters. You
  cannot shuffle trading days without destroying the data's meaning —
  this becomes critical once strategies and backtesting are involved.
- **Adjusted vs. raw price** — "Close" is the literal price that traded
  that day. "Adj Close" retroactively accounts for dividends and stock
  splits, so a chart of it reflects the true return an investor would
  have experienced. Mixing the two up is a common, easy-to-miss mistake
  that silently skews a backtest.
- **Why validate input before hitting the network** — a bad ticker or an
  impossible date range doesn't always produce a clear error from a data
  provider; sometimes it just comes back empty. Checking first (`src/data/market_data.py`'s
  `validate_inputs()`) catches this early with an actionable message.
- **Never fabricate data** — if a live download fails, QuantSim falls
  back to a clearly-labeled *synthetic* sample dataset rather than
  guessing or interpolating real-looking numbers.

## Indicator concepts (Phase 3)

- **SMA (Simple Moving Average)** — the plain average of the last *N*
  prices, recomputed at every point.
- **EMA (Exponential Moving Average)** — like SMA, but weights recent
  prices more heavily. Computed recursively: today's EMA depends on
  yesterday's EMA and today's price.
- **RSI (Relative Strength Index)** — measures the ratio of recent
  average gains to average losses, scaled to 0–100. Above 70 is
  conventionally "overbought," below 30 "oversold."
- **MACD** — the difference between a fast and a slow EMA (the "MACD
  line"), plus an EMA of that difference (the "signal line"). The gap
  between them (the "histogram") is what most MACD-based strategies
  watch for a sign change.
- **Why `min_periods` matters** — an indicator's first few values, before
  enough historical data exists, are set to `NaN` rather than computed
  from a too-small window. A partial-window "average" is a different
  (and quietly misleading) number from the real one.
- **No look-ahead bias, by construction** — every indicator here uses
  only `rolling()` or `ewm()`, both of which only ever look at the
  current row and *earlier* rows. This isn't a rule QuantSim follows by
  discipline; it's structurally impossible to violate with these
  particular pandas operations.

## Strategy concepts (Phase 4)

- **Signal** — a strategy's output is always one of exactly three
  values: BUY, SELL, or HOLD, represented by an `Enum` rather than a
  plain string so a typo can't silently produce a signal that matches
  nothing.
- **Crossover detection** — "did line A cross line B?" is answered by
  comparing *this* bar (`A > B`) to the *previous* bar (`A <= B`) — using
  only two consecutive points, never a future one. This one function
  (`detect_crossover`) is shared by both the Moving Average and MACD
  strategies, since it's the same question asked of different lines.
- **Level-based vs. crossover-based rules** — the RSI strategy fires on
  *every* bar RSI stays below/above its threshold (a level rule), while
  Moving Average and MACD fire only once, at the moment of crossing.
  This is a genuine design choice, not a universal law of RSI strategies
  — it's stated explicitly in `rsi_strategy.py`'s docstring rather than
  left as an unstated assumption.
- **Proving no look-ahead bias, not just claiming it** — the test suite
  computes a strategy's signals on the full dataset, then again on data
  truncated partway through, and checks the historical signals are
  identical either way. If a strategy secretly used future data,
  truncating the future would change the past — it doesn't.

## Backtesting concepts (Phase 5)

- **All-in / all-out position sizing** — the simplest model that
  satisfies "no fractional shares" and "never overspend": a BUY converts
  all available cash into whole shares; a SELL liquidates the entire
  position.
- **Why same-bar execution isn't look-ahead bias** — executing at bar
  *t*'s own closing price (the same bar the signal came from) only ever
  uses that bar's own data, never a future one. It's an idealization
  (instant reaction the moment a bar closes), not information leakage
  from the future.
- **Completed trades vs. an open position** — a `Trade` only exists once
  a position has been both entered *and* exited. A position still held
  when the data runs out is reported separately (`OpenPosition`) and is
  never counted in trade-based statistics like win rate.
- **Why chronological order is enforced, not assumed** — cash and share
  counts are inherently sequential (day 50's balance depends on every
  decision from days 0–49). The engine raises an error on out-of-order
  data rather than silently re-sorting it, since silent reordering could
  mask a real upstream bug.

## Performance analytics concepts (Phase 6)

- **Total Return / Absolute P&L** — the simplest possible summary:
  `(final/initial - 1) * 100` and `final - initial`.
- **Win Rate, Average Win, Average Loss** — computed only from completed
  trades. With zero completed trades, these are `None` — never `0`,
  which would misleadingly imply trades happened and all lost.
- **Maximum Drawdown** — the largest peak-to-trough decline in portfolio
  value, reported as a *negative* percentage, computed by tracking the
  running maximum and measuring how far below it the portfolio has ever
  fallen.
- **Sharpe Ratio** — return per unit of risk (volatility), relative to a
  risk-free baseline. QuantSim states its assumption plainly: 0% annual
  risk-free rate by default, since this project has no live rate source
  and hardcoding a specific "real" number would be a stale guess.
- **Why some results are `None`, never `NaN`/`inf`** — dividing by a
  standard deviation of exactly zero (a perfectly flat equity curve) is
  mathematically undefined. Rather than let that silently produce `inf`
  or `NaN`, the code explicitly checks for it and returns `None`.

## Persistence concepts (Phase 7)

- **Why no ORM** — for four small, stable tables and one supported
  database (SQLite), an ORM would add a new concept to learn (models,
  sessions, migrations) for a benefit that doesn't matter at this scale.
  Plain SQL keeps every column visible in the schema itself.
- **Foreign keys and cascading deletes** — SQLite does not enforce
  foreign key constraints unless `PRAGMA foreign_keys = ON` is set on
  *every* connection — a well-known gotcha, handled here by setting it
  inside the one shared connection helper so no call site can forget it.
- **Never storing derived data** — `Trade.pnl`/`pnl_pct`/`is_win` are
  never database columns; they're recomputed after loading from the raw
  numbers that are stored, so a cached copy can never silently disagree
  with the logic that defines it.
- **One transaction, not four** — saving a backtest's metadata, trades,
  equity curve, and metrics all happen inside a single SQLite
  transaction. If anything fails partway through, everything inserted
  so far in that call is rolled back — there's no way to end up with a
  backtest record missing its trades.

## UI concepts (Phase 8)

- **Pure chart functions, no Streamlit inside them** — `src/ui/charts.py`
  takes data in and returns a `plotly.graph_objects.Figure` out, with no
  dependency on Streamlit at all. This is what makes it possible to test
  a chart's correctness (e.g. "does a BUY marker land at the exact
  trade's entry price?") without ever starting the app.
- **Computation before display** — every tab's content is *displayed*
  inside a `st.tabs()` block, but every calculation (fetching data,
  running a strategy, running a backtest) happens once, in a fixed
  order, *before* any tab is built. This avoids a subtle class of bugs
  where a button's behavior would otherwise depend on which tab happens
  to be open.
- **Avoiding a misleading chart** — BUY/SELL markers are placed at the
  *exact* price and date the backtesting engine actually executed each
  trade — never a smoothed or estimated position — and a saved
  backtest's trade markers are only overlaid on today's price chart if
  the ticker and date range actually match; otherwise the chart is
  skipped rather than shown with mismatched data.

## Testing concepts (Phase 9)

- **Unit tests vs. integration tests** — a unit test checks one layer in
  isolation with a small, hand-built, independently-verifiable fixture.
  An integration test runs *real* output from one layer into the next,
  catching bugs that only appear at the seam between two layers, which
  matching hand-built fixtures at each layer can accidentally hide.
- **Why `tmp_path`, never the real database, in tests** — every
  database-touching test (`test_database.py`, `test_integration.py`,
  `test_app.py`) uses its own temporary SQLite file, so running the test
  suite can never corrupt or depend on the state of the real
  `quantsim.db` a person might be using interactively.
- **A real, hard-to-spot Python gotcha, found while building this
  guide's own tests** — a function's default parameter value
  (`def f(x=SOME_CONSTANT)`) is evaluated exactly once, when the
  function is *defined*, not each time it's called. Patching the
  constant afterward (`SOME_CONSTANT = new_value`) does **not**
  retroactively change a default that was already bound. The correct
  fix, used in `test_app.py`, is to patch the function's own
  `__defaults__` tuple directly.
