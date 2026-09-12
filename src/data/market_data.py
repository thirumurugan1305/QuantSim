"""
Market data service for QuantSim.

WHAT THIS MODULE DOES
----------------------
Given a ticker symbol and a date range, `fetch_market_data()` returns
historical daily OHLCV data (Open, High, Low, Close, Volume) as a clean
pandas DataFrame, along with metadata about where that data came from.

It tries a real download via `yfinance` first (Yahoo Finance's free,
no-API-key data feed). If that fails for ANY reason — no internet, an
invalid ticker, a Yahoo outage, a network policy blocking the request —
it falls back to the bundled synthetic sample CSV so the app can still be
demonstrated. The result always tells you which one you got; the app is
never allowed to silently pretend synthetic data is real.

KEY CONCEPTS (for the beginner-teaching goal of this project)
---------------------------------------------------------------
- OHLCV: shorthand for Open, High, Low, Close, Volume — the five numbers
  that describe one trading day (or other period) for a security. Open/
  Close are the first/last traded price of the day; High/Low are the
  extremes; Volume is how many shares changed hands.
- DataFrame: pandas' table structure — rows are observations (here, one
  row per trading day), columns are variables (Open, High, Low, ...).
- Time series: a sequence of data points indexed by time (here, by
  trading date). Order matters — you can't shuffle it without destroying
  its meaning, which becomes critical later when we build strategies
  (Phase 4) and must avoid "look-ahead bias".
- Adjusted vs. raw price: "Close" is the literal traded closing price.
  "Adj Close" (adjusted close) retroactively accounts for dividends and
  stock splits, so that a chart of Adj Close shows the *true* return an
  investor would have experienced. Raw Close is what actually traded
  that day. We keep both and let the caller decide which to use — mixing
  them up is a common beginner mistake that silently skews backtests.
- Why validation matters: bad input (a nonsense ticker, a start date
  after the end date, a request for dates in the future) doesn't always
  raise a clear error from yfinance — sometimes it just returns an empty
  or malformed DataFrame. Validating BEFORE we hit the network catches
  these problems early with a message the user can actually act on,
  instead of a confusing downstream crash or, worse, a silently empty
  backtest.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from functools import lru_cache

import pandas as pd
import yfinance as yf

from src.config import SAMPLE_DATA_PATH

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# A generous but sane pattern for stock tickers: letters, digits, dots, and
# hyphens (covers things like "BRK.B" or "RDS-A"), 1 to 10 characters.
_TICKER_PATTERN = re.compile(r"^[A-Za-z0-9.\-]{1,10}$")

# Columns every valid OHLCV row must have a usable value for.
_REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

# yfinance has data going back further for some tickers, but we don't want
# a beginner accidentally requesting 80 years and waiting forever / hitting
# odd edge cases. This is a sensible upper bound for an educational tool.
_MAX_RANGE_DAYS = 365 * 20

# Data sources reported back to the caller / UI.
SOURCE_LIVE = "live_yfinance"
SOURCE_FALLBACK = "fallback_sample"


@dataclass
class MarketDataResult:
    """Everything the UI needs to know about one data request.

    Keeping this as an explicit dataclass (instead of just returning a
    DataFrame) means the caller ALWAYS knows where the data came from and
    whether anything was skipped or adjusted — there is no way to
    accidentally treat fallback data as if it were real.
    """

    data: pd.DataFrame
    source: str  # SOURCE_LIVE or SOURCE_FALLBACK
    ticker: str
    warnings: list[str] = field(default_factory=list)


class MarketDataValidationError(ValueError):
    """Raised when the ticker/date-range input fails validation.

    This is a distinct exception type (rather than a bare ValueError) so
    calling code — like the Streamlit UI — can catch validation problems
    specifically and show a friendly message, without accidentally
    swallowing unrelated bugs.
    """


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_inputs(ticker: str, start_date: date, end_date: date) -> None:
    """Validate a ticker + date range before we touch the network.

    Raises MarketDataValidationError with a human-readable message on the
    first problem found. Returns None (no exception) if everything looks
    reasonable.
    """
    ticker = (ticker or "").strip()

    if not ticker:
        raise MarketDataValidationError("Please enter a ticker symbol.")

    if not _TICKER_PATTERN.match(ticker):
        raise MarketDataValidationError(
            f"'{ticker}' doesn't look like a valid ticker symbol. "
            "Use letters, digits, '.', or '-' only (e.g. AAPL, BRK.B)."
        )

    if start_date >= end_date:
        raise MarketDataValidationError(
            "Start date must be before end date."
        )

    if end_date > date.today():
        raise MarketDataValidationError(
            "End date can't be in the future — this is historical data, "
            "not a price prediction tool."
        )

    if (end_date - start_date) > timedelta(days=_MAX_RANGE_DAYS):
        raise MarketDataValidationError(
            f"Date range is too large (max {_MAX_RANGE_DAYS // 365} years). "
            "Pick a shorter window."
        )


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean_ohlcv(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Clean a raw OHLCV DataFrame and report what was removed/fixed.

    Cleaning rules (deliberately conservative — we drop bad rows rather
    than guess-fill them, since fabricating price data would violate the
    project's "never invent financial data" rule):
      - Drop rows missing any of Open/High/Low/Close/Adj Close/Volume.
      - Drop rows with non-positive prices (a $0 or negative price is
        not a real trading day).
      - Drop rows where High < Low (physically impossible).
      - Drop duplicate dates, keeping the first occurrence.
      - Sort chronologically ascending (oldest first) — required for any
        time-series/backtesting logic later.

    Returns the cleaned DataFrame plus a list of human-readable warning
    strings describing what was removed, so the UI can surface them.
    """
    warnings: list[str] = []
    original_len = len(df)

    # Some columns may be missing entirely (e.g. an unusual yfinance
    # response) — treat that as "nothing usable" rather than crashing.
    missing_cols = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        warnings.append(
            f"Response was missing expected column(s): {', '.join(missing_cols)}."
        )
        return df.iloc[0:0], warnings  # empty DataFrame, same shape

    cleaned = df.copy()

    before = len(cleaned)
    cleaned = cleaned.dropna(subset=_REQUIRED_COLUMNS)
    dropped_na = before - len(cleaned)

    before = len(cleaned)
    price_cols = ["Open", "High", "Low", "Close", "Adj Close"]
    positive_mask = (cleaned[price_cols] > 0).all(axis=1) & (cleaned["Volume"] >= 0)
    cleaned = cleaned[positive_mask]
    dropped_nonpositive = before - len(cleaned)

    before = len(cleaned)
    cleaned = cleaned[cleaned["High"] >= cleaned["Low"]]
    dropped_bad_range = before - len(cleaned)

    before = len(cleaned)
    cleaned = cleaned.drop_duplicates(subset=["Date"], keep="first")
    dropped_dupes = before - len(cleaned)

    cleaned = cleaned.sort_values("Date").reset_index(drop=True)

    if dropped_na:
        warnings.append(f"Dropped {dropped_na} row(s) with missing values.")
    if dropped_nonpositive:
        warnings.append(f"Dropped {dropped_nonpositive} row(s) with non-positive prices.")
    if dropped_bad_range:
        warnings.append(f"Dropped {dropped_bad_range} row(s) where High < Low.")
    if dropped_dupes:
        warnings.append(f"Dropped {dropped_dupes} duplicate-date row(s).")

    total_dropped = original_len - len(cleaned)
    if total_dropped:
        warnings.append(
            f"Total: removed {total_dropped} of {original_len} rows during cleaning "
            f"({len(cleaned)} rows remain)."
        )

    return cleaned, warnings


# ---------------------------------------------------------------------------
# Live download (cached)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=64)
def _download_from_yfinance(ticker: str, start_iso: str, end_iso: str) -> pd.DataFrame:
    """Download raw OHLCV data from yfinance for one ticker/date range.

    This is intentionally a thin, pure-ish function (string args in,
    DataFrame out) so it can be wrapped in `functools.lru_cache` — repeated
    requests for the same ticker/date range during a session are served
    from memory instead of re-hitting the network every time Streamlit
    reruns the script (which, as covered in Phase 1, happens on every
    widget interaction).

    Raises whatever exception yfinance/requests raises on failure; the
    caller (`fetch_market_data`) is responsible for catching it and
    falling back to sample data.
    """
    # auto_adjust=False keeps raw Close AND gives us a separate
    # "Adj Close" column — we want both, see module docstring.
    raw = yf.download(
        ticker,
        start=start_iso,
        end=end_iso,
        progress=False,
        auto_adjust=False,
    )

    if raw is None or raw.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'.")

    # yfinance can return MultiIndex columns (e.g. when batch-downloading);
    # for a single ticker we flatten that back down to simple column names.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw = raw.reset_index()  # 'Date' becomes a normal column, not the index
    return raw


# ---------------------------------------------------------------------------
# Fallback sample data
# ---------------------------------------------------------------------------

def load_fallback_sample(start_date: date, end_date: date) -> tuple[pd.DataFrame, list[str]]:
    """Load the bundled synthetic CSV, filtered to the requested range where possible.

    The synthetic file always represents the SAME made-up price series
    regardless of the ticker requested — it is not, and must never be
    presented as, real data for that ticker. Callers must label it
    clearly (the UI does this via `MarketDataResult.source`).
    """
    warnings: list[str] = []
    df = pd.read_csv(SAMPLE_DATA_PATH, parse_dates=["Date"])

    sample_min = df["Date"].min().date()
    sample_max = df["Date"].max().date()

    mask = (df["Date"].dt.date >= start_date) & (df["Date"].dt.date <= end_date)
    filtered = df[mask].reset_index(drop=True)

    if filtered.empty:
        warnings.append(
            f"Sample data covers {sample_min} to {sample_max}, which does not "
            f"overlap your requested range ({start_date} to {end_date}). "
            "Showing the full sample dataset instead."
        )
        filtered = df.reset_index(drop=True)
    elif start_date < sample_min or end_date > sample_max:
        warnings.append(
            f"Sample data only covers {sample_min} to {sample_max}; showing "
            "the overlapping portion of your requested range."
        )

    return filtered, warnings


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def fetch_market_data(ticker: str, start_date: date, end_date: date) -> MarketDataResult:
    """Main entry point: validate, try a live download, clean it, and fall
    back to sample data on any failure. Never fabricates live prices —
    on failure it always returns the clearly-labeled synthetic sample.
    """
    validate_inputs(ticker, start_date, end_date)
    ticker = ticker.strip().upper()

    try:
        raw = _download_from_yfinance(ticker, start_date.isoformat(), end_date.isoformat())
        cleaned, clean_warnings = clean_ohlcv(raw)

        if cleaned.empty:
            raise ValueError("Downloaded data was empty after cleaning.")

        return MarketDataResult(
            data=cleaned,
            source=SOURCE_LIVE,
            ticker=ticker,
            warnings=clean_warnings,
        )

    except Exception as exc:  # noqa: BLE001 - intentionally broad: any
        # download failure (network, invalid ticker, Yahoo outage, etc.)
        # should trigger the same graceful fallback, not a crash.
        fallback_df, fallback_warnings = load_fallback_sample(start_date, end_date)
        warnings = [
            f"Live data download failed ({exc.__class__.__name__}: {exc}). "
            "Using bundled synthetic sample data instead."
        ] + fallback_warnings

        return MarketDataResult(
            data=fallback_df,
            source=SOURCE_FALLBACK,
            ticker=ticker,
            warnings=warnings,
        )
