"""
Application-wide configuration for QuantSim.

This module centralizes constants so that they live in ONE place instead of
being scattered (and duplicated) across the UI, data, and backtesting code.
It intentionally does NOT read any secrets or API keys — QuantSim's core
functionality works with zero credentials, using yfinance's free endpoints
and a local SQLite database.

If we ever add a genuinely optional feature that needs an environment
variable (for example, an optional alternate data provider), we will add a
`.env.example` at that point and load it here with something like
`os.getenv("SOME_KEY", default=None)`. Until then, there is nothing to
configure via environment variables, which keeps setup simple for a
beginner.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project metadata
# ---------------------------------------------------------------------------
APP_NAME: str = "QuantSim"
APP_TAGLINE: str = "Algorithmic Trading & Backtesting Platform (Educational Simulator)"

DISCLAIMER: str = (
    "Educational simulator only. Backtested performance does not guarantee "
    "future results. No real-money trading is performed."
)

# ---------------------------------------------------------------------------
# Filesystem paths
# ---------------------------------------------------------------------------
# BASE_DIR points at the QuantSim/ project root (two levels up from this file:
# src/config.py -> src/ -> QuantSim/).
BASE_DIR: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = BASE_DIR / "data"
SAMPLE_DATA_PATH: Path = DATA_DIR / "sample_market_data.csv"

# SQLite database file. Using a local file means zero setup and no external
# database service — everything runs on your machine.
DB_PATH: Path = BASE_DIR / "quantsim.db"

# ---------------------------------------------------------------------------
# Backtesting defaults
# ---------------------------------------------------------------------------
DEFAULT_INITIAL_CAPITAL: float = 10_000.00
DEFAULT_TRANSACTION_COST_PCT: float = 0.001  # 0.1% per trade, a common simple assumption

# Trading-day assumptions used later for annualizing metrics (Phase 6).
TRADING_DAYS_PER_YEAR: int = 252
