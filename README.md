# QuantSim — Algorithmic Trading & Backtesting Platform

> **Educational simulator only.** Backtested performance does not guarantee
> future results. No real-money trading is performed, and this project does
> not connect to any brokerage.

## What this is

QuantSim is a free, open-source, local-first web app for learning how
algorithmic trading and backtesting work. You pick a stock ticker, a date
range, a virtual amount of starting capital, and a rule-based strategy
(e.g. moving-average crossover). The app simulates trades over historical
data and shows you the resulting portfolio value, trade log, and
performance metrics.

**This project is being built in phases.** This README will grow as each
phase lands. Right now, **Phases 1–9** are complete.

## Current status: Phase 9 — Testing, Documentation & Repository Quality

Phases 1–8 delivered the full working application — market data,
indicators, strategies, backtesting, performance analytics, SQLite
persistence, and a professional 4-tab UI (see the phase table in
`PROJECT_REPORT.md` for the full breakdown). Phase 9 adds no new
features; it closes real gaps in testing and documentation instead:

- **New: `tests/test_integration.py`** — end-to-end tests that run a
  real strategy through a real backtest, through real performance
  metrics, through a real SQLite save/load round trip, using the
  project's own bundled sample data. Every other test file is thorough
  at testing one layer in isolation with small hand-built fixtures; this
  file specifically checks that *real* output from one layer, fed
  unmodified into the next, still behaves correctly — including
  confirming that metrics recomputed on a *loaded* backtest still match
  the metrics computed before it was saved.
- **New: `tests/test_app.py`** — permanent UI regression tests using
  `streamlit.testing.v1.AppTest`, covering the full Run Backtest → Save
  → Load → Delete workflow against an isolated temporary database. Every
  prior phase verified this workflow manually during development; it was
  never previously saved as an actual, repeatable test.
- **New: `ARCHITECTURE.md`, `LEARNING_GUIDE.md`, `PROJECT_REPORT.md`** —
  see below.
- 190 automated tests total (171 from Phases 1–8, plus 8 integration
  tests and 11 UI regression tests) — no existing test was modified.

See `ARCHITECTURE.md` for how the codebase is structured and why,
`LEARNING_GUIDE.md` for the concepts explained across all 9 phases, and
`PROJECT_REPORT.md` for a portfolio-style summary of the whole project.

## Requirements

- Python 3.11 or 3.12
- No paid accounts, no API keys, no credit card — everything here is free
  and runs locally.

## Installation

```bash
# 1. Clone or download this project, then move into it
cd QuantSim

# 2. Create a virtual environment
python3 -m venv .venv

# 3. Activate it
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows (PowerShell: .venv\Scripts\Activate.ps1)

# 4. Install dependencies
pip install -r requirements.txt
```

## Running locally

```bash
streamlit run app.py
```

Streamlit will print a local URL (typically `http://localhost:8501`) —
open it in your browser.

## Running Tests

```bash
pytest tests/          # full suite (190 tests)
pytest tests/ -v       # verbose, one line per test
pytest tests/test_backtesting.py -v   # just one file
```

Tests are organized one file per layer, plus two cross-cutting files:

| File | What it covers |
|---|---|
| `test_indicators.py` | SMA, EMA, RSI, MACD (31 tests) |
| `test_strategies.py` | Moving Average, RSI, MACD strategies, incl. look-ahead-bias checks (36 tests) |
| `test_backtesting.py` | The backtesting engine, incl. transaction rollback and no-look-ahead checks (30 tests) |
| `test_performance_metrics.py` | Performance statistics, cross-checked against Python's `statistics` module (23 tests) |
| `test_database.py` | SQLite persistence, incl. cascading deletes and foreign-key enforcement (31 tests) |
| `test_charts.py` | Plotly chart-building functions, checking actual coordinates (20 tests) |
| `test_integration.py` | Real strategy → backtest → metrics → save/load, end-to-end (8 tests) |
| `test_app.py` | Permanent UI regression tests via `streamlit.testing.v1.AppTest` (11 tests) |

Every database-touching test uses its own temporary SQLite file — the
suite never reads or writes your real `quantsim.db`.

## Project structure (current)

```
QuantSim/
├── app.py                     # Streamlit entry point (Phase 1), 4-tab dashboard (Phase 8)
├── requirements.txt
├── README.md
├── ARCHITECTURE.md            # Phase 9: how the codebase is structured and why
├── LEARNING_GUIDE.md          # Phase 9: concepts explained across all 9 phases
├── PROJECT_REPORT.md          # Phase 9: portfolio-style project summary
├── .gitignore
├── data/
│   └── sample_market_data.csv # Synthetic offline fallback data
├── src/
│   ├── config.py              # App-wide constants, paths, defaults
│   ├── data/
│   │   └── market_data.py     # Phase 2: yfinance download + validation + cleaning + fallback
│   ├── indicators/
│   │   └── technical_indicators.py  # Phase 3: SMA, EMA, RSI, MACD
│   ├── strategies/
│   │   ├── base_strategy.py         # Signal enum, BaseStrategy, shared crossover logic
│   │   ├── moving_average_strategy.py
│   │   ├── rsi_strategy.py
│   │   └── macd_strategy.py
│   ├── backtesting/
│   │   ├── models.py                # Trade, OpenPosition, PortfolioSnapshot, BacktestResult
│   │   └── engine.py                # Phase 5: BacktestEngine, run_backtest()
│   ├── analytics/
│   │   └── performance_metrics.py   # Phase 6: PerformanceMetrics, calculate_performance_metrics()
│   ├── database/
│   │   └── repository.py            # Phase 7: SQLite schema, save/load/list/delete
│   └── ui/
│       └── charts.py                # Phase 8: pure Plotly chart-building functions
└── tests/
    ├── test_indicators.py     # Phase 3: 31 tests for SMA/EMA/RSI/MACD
    ├── test_strategies.py     # Phase 4: 36 tests for the strategy layer
    ├── test_backtesting.py    # Phase 5: 30 tests for the backtesting engine
    ├── test_performance_metrics.py  # Phase 6: 23 tests for performance analytics
    ├── test_database.py       # Phase 7: 31 tests for SQLite persistence
    ├── test_charts.py         # Phase 8: 20 tests for the Plotly chart builders
    ├── test_integration.py    # Phase 9: 8 real end-to-end pipeline tests
    └── test_app.py            # Phase 9: 11 permanent AppTest UI regression tests
```

## Screenshots

_(placeholder — add a screenshot of the running app here once you have one)_

## Limitations (current phase)

- Live data depends on `yfinance` reaching Yahoo Finance over the
  internet. If that's blocked or fails, you'll see synthetic sample data
  instead — clearly labeled, never presented as real.
- The bundled sample data represents one fixed, made-up price series; it
  does not adapt to the ticker you typed, only to the date range (where
  it overlaps).
- The backtesting engine is long-only, all-in/all-out, and executes at
  the same bar's closing price — no short selling, partial position
  sizing, or transaction costs by default (see `src/backtesting/engine.py`
  for the full list of documented assumptions).
- Performance metrics assume a 0% annual risk-free rate for the Sharpe
  Ratio by default (configurable, but not connected to any live rate
  source) and use a simple annual-to-daily conversion
  (`rate / TRADING_DAYS_PER_YEAR`), not compounding — a stated
  simplification, not a real-world-grade calculation.
- No benchmark comparison (e.g. vs. buy-and-hold) exists yet.
- Saved backtests live in a local SQLite file (`quantsim.db`, ignored by
  Git) — there is no cloud sync, multi-user support, or migration
  framework. If the schema ever needs to change, the documented approach
  for this local, disposable, educational database is to delete the
  `.db` file and let the app recreate it, not to migrate it in place.
- Every bar of a saved backtest's equity curve becomes one database row,
  so `portfolio_snapshots` grows with each save — fine at the "a student
  saves a handful of runs" scale this project targets, but not designed
  for large-scale or long-term production use.
- Saved `strategy_params` are stored as a JSON blob for display purposes
  only; they are never read back into strategy logic, so they can't
  silently affect how a loaded backtest behaves.
- The price-with-trade-markers chart only appears when the currently
  loaded market data (ticker + date range in the sidebar) matches the
  backtest being displayed — this is intentional, to avoid overlaying
  one ticker's trades on a different ticker's price chart, but it means
  loading a saved backtest for a different symbol than what's currently
  in the sidebar will show its equity curve and trade table without that
  particular chart until the sidebar is updated to match.

## Disclaimer

Educational simulator only. Backtested performance does not guarantee
future results. No real-money trading is performed. This is not financial
advice.

## Future improvements

See the phase list in this project's build plan — an optional/
experimental ML module is planned for a later phase.
