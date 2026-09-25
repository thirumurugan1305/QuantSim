# QuantSim — Project Report

## Overview

QuantSim is an educational algorithmic trading and backtesting platform
built entirely with free, open-source tools: Python, Streamlit, pandas,
NumPy, yfinance, Plotly, and SQLite (via Python's built-in `sqlite3`
module — no external database server, no paid services, no real-money
trading). A user selects a stock ticker, a historical date range, a
trading strategy and its parameters, and an amount of virtual starting
capital; QuantSim simulates how that strategy would have performed,
shows the resulting trades and portfolio value over time, computes
standard performance statistics, and can save/reload past backtests.

Built incrementally across 9 phases, each one verified with a real test
suite and a real running application before the next began.

## Phase-by-phase scope

| Phase | Delivered |
|---|---|
| 1 — Foundation | Project structure, config, first working Streamlit screen |
| 2 — Market Data | `yfinance` download with validation, cleaning, and synthetic-data fallback |
| 3 — Technical Indicators | SMA, EMA, RSI, MACD — pure, tested, reusable functions |
| 4 — Strategy Engine | Moving Average Crossover, RSI, and MACD strategies producing standardized BUY/SELL/HOLD signals |
| 5 — Backtesting Engine | Long-only, all-in/all-out trade simulation from a signal series |
| 6 — Performance Analytics | Total return, win rate, drawdown, Sharpe ratio, volatility, and more |
| 7 — SQLite Persistence | Save, load, list, and delete full backtest results locally |
| 8 — Professional UI/UX | 4-tab dashboard, interactive Plotly charts with trade markers, drawdown chart |
| 9 — Testing, Documentation & Repository Quality | Integration tests, permanent UI regression tests, this report and its companion docs |

## Final numbers

- **190 automated tests**, all passing:
  - 31 — technical indicators
  - 36 — strategies (including look-ahead-bias verification)
  - 30 — backtesting engine (including transaction-rollback and no-look-ahead tests)
  - 23 — performance analytics (independently cross-checked against Python's `statistics` module, not just against the implementation's own logic)
  - 31 — SQLite persistence (including cascading-delete and foreign-key-enforcement tests)
  - 20 — Plotly chart builders (checking actual marker coordinates against known data, not just "it rendered")
  - 8 — end-to-end integration tests (a real strategy through a real backtest through real metrics through a real save/load round trip)
  - 11 — permanent UI regression tests (`streamlit.testing.v1.AppTest`, covering the full Run → Save → Load → Delete workflow)
- **~3,400 lines** of application code across `app.py` and `src/`.
- **Zero external services, zero paid APIs, zero real-money trading capability.**

## Engineering decisions worth highlighting

- **No look-ahead bias, enforced by construction and proven by test.**
  Every indicator, every strategy signal, and every backtest execution
  step only ever reads the current row and earlier ones. This isn't
  just documented — it's tested directly: signals and drawdown values
  are computed on a truncated dataset and compared against the full
  dataset's historical values, which must match exactly if nothing
  downstream is peeking into the future.
- **Pure functions at every layer below the UI.** Indicators, strategies,
  the backtesting engine, performance metrics, and the chart-building
  functions all take plain data in and return plain data out, with no
  hidden state and no dependency on Streamlit. This is what makes the
  entire system testable without ever starting the app, and it's the
  same architectural pattern repeated consistently from Phase 3 through
  Phase 8.
- **Transactional, non-destructive persistence.** Saving a backtest
  writes its metadata, trades, equity curve, and metrics inside a single
  SQLite transaction — a failure partway through rolls back everything,
  never leaving a half-saved record. Derived values (a trade's
  profit/loss) are recomputed after loading rather than stored, so a
  cached number can never silently disagree with the logic that
  actually defines it.
- **Explicit, stated assumptions instead of silent defaults.** Wherever
  a calculation needed an assumption with no objectively "correct"
  value (the Sharpe ratio's risk-free rate, all-in/all-out position
  sizing, same-bar trade execution), the code documents *why* that
  choice was made rather than presenting it as the only possible answer.
- **`None` instead of `NaN`/`inf`.** Every statistic that could
  mathematically produce an undefined result (dividing by a zero
  standard deviation, averaging an empty list of trades) is explicitly
  guarded to return `None` — a signal that means "not meaningful for
  this data," which is a more honest answer than a number that merely
  looks valid.
- **Integration tests that use real data, not matching hand-built
  fixtures.** Every unit test suite (one per layer) is thorough at
  testing that layer alone with small, verifiable fixtures — but Phase 9
  specifically added tests that run *actual* output from one real layer
  into the next real layer, which is the only way to catch a bug that
  exists at the boundary between two layers rather than inside either
  one.

## Limitations (stated plainly)

- Backtesting is long-only, all-in/all-out, with no leverage, no short
  selling, and no transaction costs by default.
- Performance metrics assume a 0% annual risk-free rate unless
  configured otherwise, and use a simplified (non-compounding)
  annual-to-daily rate conversion.
- Persistence is a single local SQLite file with no migration framework
  — appropriate for an educational, single-user, local tool, not a
  production multi-user system.
- No benchmark comparison (e.g. vs. buy-and-hold) exists yet.

## What this project demonstrates

- Building a multi-layer Python application incrementally, with each
  layer's contract (what it takes in, what it returns, what it promises
  never to do) defined before the next layer is built on top of it.
- Writing tests that actually verify correctness (hand-computed expected
  values, independent cross-checks against a different library, explicit
  look-ahead-bias proofs) rather than tests that only confirm code runs
  without crashing.
- Recognizing the difference between a unit-test gap and an
  integration-test gap, and closing the right one without inflating the
  test count for its own sake.
- Working entirely within a free, local, open-source toolchain while
  still building something with the shape of a real financial analysis
  tool — data ingestion, technical analysis, strategy logic, simulation,
  statistics, persistence, and a professional interactive UI.
