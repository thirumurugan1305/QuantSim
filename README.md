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
phase lands. Right now, **Phases 1–5** are complete.

## Current status: Phase 5 — Backtesting Engine

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
- **New in Phase 5:** `src/backtesting/` actually simulates trades from
  those signals. Click **Run Backtest** in the sidebar to:
  - Walk through the loaded data bar-by-bar, chronologically.
  - BUY spends all available cash on whole shares (no fractional
    shares, no leverage); SELL closes the entire position. Long-only —
    no short selling.
  - Track cash, shares held, and total portfolio value at every bar.
  - Record every **completed** trade (entry/exit price, date, P&L), and
    separately report any position still **open** (unrealized) at the
    end of the data, without force-closing it.
  - Execution happens at the same bar's closing price the signal was
    generated from — a documented simplifying assumption for this
    version (see `src/backtesting/engine.py` for the full reasoning).
  - This is still a **preview**: no Sharpe ratio, max drawdown, win
    rate, or CAGR yet — those are Phase 6. Only raw results (final
    value, trade count, equity curve, trade table) are shown.
- 97 automated tests total across `tests/test_indicators.py`,
  `tests/test_strategies.py`, and `tests/test_backtesting.py`, including
  dedicated look-ahead-bias checks at both the strategy and backtest
  layers.

What is *not* built yet (coming in later phases): performance metrics
(Sharpe ratio, drawdown, win rate, CAGR), the local SQLite database, and
the full multi-tab UI. See `LEARNING_GUIDE.md` and `ARCHITECTURE.md`
(added as those phases land) for details.

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
│   ├── analytics/             # (Phase 6) performance metrics
│   ├── database/              # (Phase 7) SQLite persistence
│   └── ui/                    # (Phase 8) chart helpers
└── tests/
    ├── test_indicators.py     # Phase 3: 31 tests for SMA/EMA/RSI/MACD
    ├── test_strategies.py     # Phase 4: 36 tests for the strategy layer
    └── test_backtesting.py    # Phase 5: 30 tests for the backtesting engine
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
- No performance metrics (Sharpe ratio, max drawdown, win rate, CAGR) or
  database persistence exist yet.

## Disclaimer

Educational simulator only. Backtested performance does not guarantee
future results. No real-money trading is performed. This is not financial
advice.

## Future improvements

See the phase list in this project's build plan — performance analytics
(Phase 6), local SQLite persistence (Phase 7), a richer multi-tab UI
(Phase 8), and an optional/experimental ML module are all planned for
later phases.
