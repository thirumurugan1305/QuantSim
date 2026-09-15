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
phase lands. Right now, **Phases 1–7** are complete.

## Current status: Phase 7 — SQLite Persistence

What works today:
- The app boots with `streamlit run app.py`.
- The sidebar's **ticker** and **date range** controls drive a real
  request to `src/data/market_data.py`, which tries a live `yfinance`
  download and transparently falls back to bundled synthetic sample data
  if that fails.
- `src/indicators/technical_indicators.py` provides reusable, tested SMA,
  EMA, RSI, and MACD calculations, previewed on the main page.
- `src/strategies/` turns those indicators into standardized BUY / SELL /
  HOLD signals via Moving Average Crossover, RSI, and MACD strategies,
  live-configurable in the sidebar with a "Strategy Signals Preview".
- `src/backtesting/` simulates trades from those signals: long-only,
  all-in/all-out, whole shares only, executed at the same bar's closing
  price. Click **Run Backtest** to see final portfolio value, an equity
  curve, and a completed-trade table.
- `src/analytics/performance_metrics.py` computes Total Return, Win Rate,
  Max Drawdown, Sharpe Ratio, Annualized Volatility, and more, straight
  from that same backtest result — shown in a "Performance Metrics"
  section, with "N/A" wherever a metric genuinely isn't meaningful.
- **New in Phase 7:** `src/database/repository.py` persists a completed
  backtest (all its trades, its full equity curve, any open position,
  and its performance metrics) to a local SQLite file using Python's
  built-in `sqlite3` module — no external database, no ORM:
  - **Save Backtest** stores the currently displayed result in one
    all-or-nothing transaction; a "Saved Backtests" section lists every
    previously saved run (id, date saved, ticker, strategy, date range,
    total return%).
  - **Load Selected** reconstructs a full `BacktestResult` from storage
    and displays it through the exact same rendering code a live
    backtest uses — there is no separate "viewing a saved result" code
    path.
  - **Delete Selected** removes a saved backtest (with a simple two-step
    confirmation) and automatically removes its trades/snapshots/metrics
    via a cascading delete; deleting a result that's currently on screen
    never crashes the app, since the in-memory view and the database row
    are independent.
  - Derived values (`Trade.pnl`, `.pnl_pct`, `.is_win`) are never stored
    — they're recomputed after loading from the same raw numbers the
    `Trade` dataclass already uses, so a stored copy can never silently
    drift out of sync with that logic.
- 151 automated tests total across `tests/test_indicators.py`,
  `tests/test_strategies.py`, `tests/test_backtesting.py`,
  `tests/test_performance_metrics.py`, and `tests/test_database.py` —
  including round-trip persistence checks, cascading-delete checks, a
  direct proof that foreign-key enforcement is actually active (not just
  declared), and a transaction-rollback test confirming a failed save
  never leaves a partial record behind.

What is *not* built yet: the richer multi-tab UI (Phase 8). See
`LEARNING_GUIDE.md` and `ARCHITECTURE.md` (added as those phases land)
for details.

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

## Project structure (current)

```
QuantSim/
├── app.py                     # Streamlit entry point (Phase 1)
├── requirements.txt
├── README.md
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
│   └── ui/                    # (Phase 8) chart helpers
└── tests/
    ├── test_indicators.py     # Phase 3: 31 tests for SMA/EMA/RSI/MACD
    ├── test_strategies.py     # Phase 4: 36 tests for the strategy layer
    ├── test_backtesting.py    # Phase 5: 30 tests for the backtesting engine
    ├── test_performance_metrics.py  # Phase 6: 23 tests for performance analytics
    └── test_database.py       # Phase 7: 31 tests for SQLite persistence
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

## Disclaimer

Educational simulator only. Backtested performance does not guarantee
future results. No real-money trading is performed. This is not financial
advice.

## Future improvements

See the phase list in this project's build plan — a richer multi-tab UI
(Phase 8) and an optional/experimental ML module are planned for later
phases.
