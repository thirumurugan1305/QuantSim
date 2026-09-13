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
phase lands. Right now, **Phases 1–6** are complete.

## Current status: Phase 6 — Performance Analytics

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
- **New in Phase 6:** `src/analytics/performance_metrics.py` computes
  descriptive statistics straight from that same backtest result — no
  strategy or backtest is re-run:
  - Total Return, Absolute P&L, Number of Trades, Win Rate
  - Average Winning Trade, Average Losing Trade
  - Maximum Drawdown (reported as a negative percentage)
  - Sharpe Ratio (assumes a 0% annual risk-free rate by default — a
    stated simplifying assumption, not a live market rate) and
    Annualized Volatility
  - A metric that isn't meaningful for the current data (e.g. no
    completed trades yet, or a flat/too-short equity curve) always
    shows **N/A** — never a misleading 0 or a crash.
  - An open position still held at the end of the data is never counted
    as a completed trade, but its unrealized value is still reflected in
    Total Return/P&L, and the UI notes this explicitly.
- 120 automated tests total across `tests/test_indicators.py`,
  `tests/test_strategies.py`, `tests/test_backtesting.py`, and
  `tests/test_performance_metrics.py`, including dedicated
  look-ahead-bias checks and independently-verified Sharpe/volatility
  calculations (checked against Python's `statistics` module, not just
  against the implementation's own pandas logic).

What is *not* built yet (coming in later phases): the local SQLite
database (Phase 7) and the richer multi-tab UI (Phase 8). See
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
│   ├── database/              # (Phase 7) SQLite persistence
│   └── ui/                    # (Phase 8) chart helpers
└── tests/
    ├── test_indicators.py     # Phase 3: 31 tests for SMA/EMA/RSI/MACD
    ├── test_strategies.py     # Phase 4: 36 tests for the strategy layer
    ├── test_backtesting.py    # Phase 5: 30 tests for the backtesting engine
    └── test_performance_metrics.py  # Phase 6: 23 tests for performance analytics
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
- No benchmark comparison (e.g. vs. buy-and-hold) exists yet, and no
  database persistence exists yet — results are only kept for the
  current browser session.

## Disclaimer

Educational simulator only. Backtested performance does not guarantee
future results. No real-money trading is performed. This is not financial
advice.

## Future improvements

See the phase list in this project's build plan — local SQLite
persistence (Phase 7), a richer multi-tab UI (Phase 8), and an
optional/experimental ML module are all planned for later phases.
