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
phase lands. Right now, **Phases 1–8** are complete.

## Current status: Phase 8 — Professional UI/UX

What works today:
- The app boots with `streamlit run app.py`, now organized as a
  **4-tab dashboard** — "Market & Signals", "Backtest Results",
  "Performance", and "Saved Backtests" — instead of one long scrolling
  page. The sidebar's inputs are grouped under clear section labels
  (Market Data / Strategy / Backtest).
- The sidebar's **ticker** and **date range** controls drive a real
  request to `src/data/market_data.py`, which tries a live `yfinance`
  download and transparently falls back to bundled synthetic sample data
  if that fails; fetching and running a backtest now show a spinner.
- `src/indicators/technical_indicators.py` provides reusable, tested SMA,
  EMA, RSI, and MACD calculations, previewed with interactive Plotly
  charts (hover tooltips, RSI oversold/overbought bands, a MACD zero
  line).
- `src/strategies/` turns those indicators into standardized BUY / SELL /
  HOLD signals via Moving Average Crossover, RSI, and MACD strategies,
  live-configurable in the sidebar with a "Strategy Signals Preview".
- `src/backtesting/` simulates trades from those signals: long-only,
  all-in/all-out, whole shares only, executed at the same bar's closing
  price. **Run Backtest** now shows a price chart with actual ▲ BUY / ▼
  SELL markers at each trade's exact execution point, an equity curve
  with a cash/holdings hover breakdown, and a new drawdown-from-peak
  chart, alongside the completed-trade table.
- `src/analytics/performance_metrics.py` computes Total Return, Win Rate,
  Max Drawdown, Sharpe Ratio, Annualized Volatility, and more, now
  grouped into Returns / Risk / Trades sections on the "Performance" tab,
  with "N/A" wherever a metric genuinely isn't meaningful.
- `src/database/repository.py` persists a completed backtest to a local
  SQLite file — Save, Load, and the two-step Delete confirmation all work
  exactly as before, now on their own "Saved Backtests" tab.
- **New in Phase 8:** `src/ui/charts.py` — pure, dependency-free
  (no Streamlit import) Plotly chart-building functions, used across
  every tab. `plotly` has been listed in `requirements.txt` since Phase
  1; this is the first phase that actually uses it. No calculation
  logic changed anywhere — every chart is built from data Phases 2–7
  already compute (the drawdown chart reuses the exact same running-peak
  formula `PerformanceMetrics.max_drawdown_pct` uses internally, kept as
  its own small function so `src/analytics/` stays untouched).
- 171 automated tests total, including 20 new chart tests that check
  actual trace counts and marker coordinates against known input data
  (not just "it rendered without crashing").

See `LEARNING_GUIDE.md` and `ARCHITECTURE.md` (added as those phases
land) for details.

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
├── app.py                     # Streamlit entry point (Phase 1), 4-tab dashboard (Phase 8)
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
│   └── ui/
│       └── charts.py                # Phase 8: pure Plotly chart-building functions
└── tests/
    ├── test_indicators.py     # Phase 3: 31 tests for SMA/EMA/RSI/MACD
    ├── test_strategies.py     # Phase 4: 36 tests for the strategy layer
    ├── test_backtesting.py    # Phase 5: 30 tests for the backtesting engine
    ├── test_performance_metrics.py  # Phase 6: 23 tests for performance analytics
    ├── test_database.py       # Phase 7: 31 tests for SQLite persistence
    └── test_charts.py         # Phase 8: 20 tests for the Plotly chart builders
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
