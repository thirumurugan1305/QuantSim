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
phase lands. Right now, **Phases 1–4** are complete.

## Current status: Phase 4 — Strategy Engine

What works today:
- The app boots with `streamlit run app.py`.
- The sidebar's **ticker** and **date range** controls drive a real
  request to `src/data/market_data.py`, which tries a live `yfinance`
  download and transparently falls back to bundled synthetic sample data
  if that fails.
- `src/indicators/technical_indicators.py` provides reusable, tested SMA,
  EMA, RSI, and MACD calculations, previewed on the main page.
- **New in Phase 4:** `src/strategies/` turns those indicators into
  standardized BUY / SELL / HOLD signals:
  - **Moving Average Crossover** — BUY when the fast SMA crosses above
    the slow SMA, SELL when it crosses below.
  - **RSI** — BUY while RSI is below the oversold threshold, SELL while
    it's above the overbought threshold.
  - **MACD** — BUY when the MACD line crosses above its signal line,
    SELL when it crosses below.
  The sidebar's **Strategy** dropdown is now live, with parameter inputs
  specific to whichever strategy is selected, and a "Strategy Signals
  Preview" section shows the resulting signal counts, a signal-events
  table, and a simple signal-pulse chart.
- This is **still a preview**: no trades are executed, no profit/loss is
  calculated, and no portfolio is tracked yet. The "Run Backtest" button
  stays disabled until Phase 5.
- 67 automated tests total (`tests/test_indicators.py` +
  `tests/test_strategies.py`), including a dedicated look-ahead-bias
  check that verifies no strategy's past signals change when future data
  is added.

What is *not* built yet (coming in later phases): the backtesting engine
(actually simulating trades and tracking a portfolio), performance
metrics, the local SQLite database, and the full multi-tab UI. See
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
│   ├── backtesting/           # (Phase 5) backtest + portfolio accounting
│   ├── analytics/             # (Phase 6) performance metrics
│   ├── database/              # (Phase 7) SQLite persistence
│   └── ui/                    # (Phase 8) chart helpers
└── tests/
    ├── test_indicators.py     # Phase 3: 31 tests for SMA/EMA/RSI/MACD
    └── test_strategies.py     # Phase 4: 36 tests for the strategy layer
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
- No indicators, strategies, backtesting, metrics, or database
  persistence exist yet.
- The "Run Backtest" button remains disabled until Phase 5.

## Disclaimer

Educational simulator only. Backtested performance does not guarantee
future results. No real-money trading is performed. This is not financial
advice.

## Future improvements

See the phase list in this project's build plan — indicators, strategies,
a full backtesting engine, performance analytics, local persistence, a
richer multi-tab UI, and an optional/experimental ML module are all
planned for later phases.
