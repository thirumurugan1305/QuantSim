"""
Technical indicator calculations for QuantSim.

WHAT THIS MODULE DOES
----------------------
Pure, reusable functions that turn a raw price series into a technical
indicator series (or, for MACD, a small DataFrame of related series).
Nothing here knows about Streamlit, yfinance, or strategies — these
functions take a `pandas.Series` in and return a `pandas.Series` or
`pandas.DataFrame` out. That separation is deliberate: the strategy engine
(Phase 4) will import and reuse these exact functions, and so could a
future test, notebook, or CLI script, without dragging in the UI.

INDICATORS IMPLEMENTED
------------------------
- SMA  (Simple Moving Average)
- EMA  (Exponential Moving Average) — implemented because MACD is built
  directly from two EMAs, and it's a genuinely reusable building block
  for strategies later (e.g. an EMA-based crossover), not just an
  internal MACD detail.
- RSI  (Relative Strength Index)
- MACD (Moving Average Convergence Divergence, plus its signal line and
  histogram)

NO LOOK-AHEAD BIAS
--------------------
Every calculation here uses only `pandas.Series.rolling()` or
`pandas.Series.ewm()`, both of which are backward-looking by
construction: the value at row N is computed from row N and *earlier*
rows only, never from future rows. This property matters a lot once we
build strategies (Phase 4) — a signal generated from these indicators is
always safe to "act on" the next bar, because nothing here peeked ahead.

HOW MISSING/INSUFFICIENT DATA IS HANDLED
-------------------------------------------
- Invalid *parameters* (e.g. window=0, fast >= slow) raise
  `IndicatorInputError` immediately — that's a configuration mistake, not
  a data problem, and should fail loudly and early.
- Insufficient *data* (e.g. a 50-day SMA requested on 10 rows of prices)
  does NOT raise. It returns `NaN` for every row that doesn't yet have
  enough history — the same convention pandas itself uses. This lets
  calling code simply check `.isna()` rather than wrapping every call in
  a try/except.
- Missing values (NaN) already present in the input series are not
  silently dropped or filled — they propagate through the calculation
  the way pandas naturally handles them, so a gap in the input data
  shows up as a gap in the output rather than a fabricated number.
"""

from __future__ import annotations

import pandas as pd


class IndicatorInputError(ValueError):
    """Raised for invalid indicator *parameters* (not for insufficient data).

    A distinct exception type (rather than a bare ValueError) lets calling
    code — like a future strategy or the UI — catch configuration mistakes
    specifically, the same pattern used by `MarketDataValidationError` in
    the data layer.
    """


def _validate_series(series: pd.Series, name: str = "series") -> None:
    if not isinstance(series, pd.Series):
        raise IndicatorInputError(f"'{name}' must be a pandas Series, got {type(series).__name__}.")
    if not pd.api.types.is_numeric_dtype(series):
        raise IndicatorInputError(f"'{name}' must contain numeric values.")


def _validate_positive_int(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise IndicatorInputError(f"'{name}' must be an integer, got {type(value).__name__}.")
    if value < 1:
        raise IndicatorInputError(f"'{name}' must be a positive integer (got {value}).")


# ---------------------------------------------------------------------------
# SMA
# ---------------------------------------------------------------------------

def calculate_sma(series: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average.

    SMA[t] = mean(series[t-window+1 : t+1])

    i.e. the plain average of the last `window` prices, including the
    current one. Uses `min_periods=window` so the first `window - 1`
    values are NaN rather than an average of fewer points than requested
    — a partial-window "SMA" would be a different (and quietly
    misleading) number.

    Raises IndicatorInputError if `window` is not a positive integer or
    `series` is not a numeric pandas Series. Returns an all-NaN Series
    (not an error) if `series` has fewer rows than `window` — see module
    docstring.
    """
    _validate_series(series)
    _validate_positive_int(window, "window")

    return series.rolling(window=window, min_periods=window).mean().rename(f"SMA_{window}")


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average.

    Uses the standard recursive definition (pandas `adjust=False`):
        EMA[0] = series[0]
        EMA[t] = price[t] * alpha + EMA[t-1] * (1 - alpha),  alpha = 2 / (span + 1)

    This weights recent prices more heavily than older ones, unlike SMA
    which weights every price in the window equally. `min_periods=span`
    is applied so the first `span - 1` values are NaN, matching the SMA
    convention above and avoiding an unstable/undermatured early EMA.
    """
    _validate_series(series)
    _validate_positive_int(span, "span")

    return series.ewm(span=span, adjust=False, min_periods=span).mean().rename(f"EMA_{span}")


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder's smoothing method).

    Steps:
      1. delta[t]  = price[t] - price[t-1]
      2. gain[t]   = delta[t] if delta[t] > 0 else 0
         loss[t]   = -delta[t] if delta[t] < 0 else 0
      3. avg_gain, avg_loss = Wilder-smoothed averages of gain/loss,
         implemented as an EWM with alpha = 1/period (this IS Wilder's
         smoothing — it's mathematically an EMA with that specific alpha).
      4. RS  = avg_gain / avg_loss
      5. RSI = 100 - (100 / (1 + RS))

    RSI oscillates between 0 (relentless selling) and 100 (relentless
    buying); 50 means gains and losses have been balanced.

    Edge cases handled explicitly:
      - avg_loss == 0 and avg_gain > 0  -> RSI = 100 (all gains, no losses)
      - avg_gain == 0 and avg_loss == 0 -> RSI = 50  (no price movement at
        all — RS is mathematically undefined (0/0), so we define this as
        neutral rather than propagating NaN)
    """
    _validate_series(series)
    _validate_positive_int(period, "period")

    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # avg_gain == 0 and avg_loss == 0 produces 0/0 = NaN in `rs`, which
    # correctly stays NaN through the formula above by default. We instead
    # define that specific "no movement at all" case as neutral (50).
    no_movement = (avg_gain == 0) & (avg_loss == 0)
    rsi = rsi.mask(no_movement, 50.0)

    return rsi.rename(f"RSI_{period}")


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------

def calculate_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """Moving Average Convergence Divergence.

    macd_line   = EMA(series, fast) - EMA(series, slow)
    signal_line = EMA(macd_line, signal)
    histogram   = macd_line - signal_line

    The MACD line shows the relationship between a fast and a slow EMA;
    the signal line smooths the MACD line itself; the histogram shows the
    gap between them, which is what most "MACD crossover" strategies
    actually watch for a sign change.

    `fast` must be strictly less than `slow`, or the two EMAs would be
    equally- or perversely-weighted and the whole indicator would be
    meaningless. Returns a DataFrame with columns
    ['macd_line', 'signal_line', 'histogram'], indexed the same as the
    input series.
    """
    _validate_series(series)
    _validate_positive_int(fast, "fast")
    _validate_positive_int(slow, "slow")
    _validate_positive_int(signal, "signal")

    if fast >= slow:
        raise IndicatorInputError(
            f"'fast' ({fast}) must be smaller than 'slow' ({slow})."
        )

    ema_fast = calculate_ema(series, fast)
    ema_slow = calculate_ema(series, slow)
    macd_line = ema_fast - ema_slow

    # macd_line has leading NaNs (from ema_slow's warm-up period). ewm's
    # min_periods counts *non-null* observations, so signal_line correctly
    # waits for `signal` many valid macd_line values before producing a
    # result — it does not miscount the leading NaNs as real observations.
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line

    return pd.DataFrame(
        {
            "macd_line": macd_line,
            "signal_line": signal_line,
            "histogram": histogram,
        }
    )
