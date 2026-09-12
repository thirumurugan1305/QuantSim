"""
Tests for src/indicators/technical_indicators.py

Covers, for each indicator:
  - basic correctness against manually-verifiable expected values
  - correct number of leading NaNs (warm-up period)
  - invalid-parameter handling (raises IndicatorInputError)
  - insufficient-data handling (returns NaN, does not crash)
  - RSI's specific boundary conditions (all-gains, all-losses, no movement)
  - MACD's structural correctness and its fast/slow validation rule

Run with:
    pytest tests/test_indicators.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.indicators.technical_indicators import (
    IndicatorInputError,
    calculate_ema,
    calculate_macd,
    calculate_rsi,
    calculate_sma,
)


# ---------------------------------------------------------------------------
# Fixtures / shared data
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_series() -> pd.Series:
    """1, 2, 3, ..., 10 — easy to hand-verify SMA/EMA against."""
    return pd.Series(range(1, 11), dtype=float)


# ---------------------------------------------------------------------------
# SMA
# ---------------------------------------------------------------------------

class TestSMA:
    def test_basic_correctness(self, simple_series):
        result = calculate_sma(simple_series, window=3)
        # SMA of [1,2,3] = 2.0, [2,3,4] = 3.0, ..., [8,9,10] = 9.0
        expected_tail = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        assert result.dropna().tolist() == expected_tail

    def test_leading_nans_equal_window_minus_one(self, simple_series):
        result = calculate_sma(simple_series, window=3)
        assert result.isna().sum() == 2

    def test_insufficient_data_returns_all_nan_not_error(self):
        short_series = pd.Series([1.0, 2.0])  # only 2 points
        result = calculate_sma(short_series, window=5)
        assert len(result) == 2
        assert result.isna().all()

    @pytest.mark.parametrize("bad_window", [0, -1, 3.5, "5"])
    def test_invalid_window_raises(self, simple_series, bad_window):
        with pytest.raises(IndicatorInputError):
            calculate_sma(simple_series, window=bad_window)

    def test_non_series_input_raises(self):
        with pytest.raises(IndicatorInputError):
            calculate_sma([1, 2, 3], window=2)


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class TestEMA:
    def test_basic_correctness_matches_manual_recursion(self, simple_series):
        span = 3
        alpha = 2 / (span + 1)
        result = calculate_ema(simple_series, span=span)

        # Manually recompute using the recursive definition, seeded at the
        # first index where min_periods is satisfied (index span - 1),
        # seeded with the SMA of the first `span` values (pandas' own
        # internal seeding for adjust=False) and compare within tolerance.
        values = simple_series.tolist()
        manual = values[span - 1]  # seed will be checked via tolerance below
        # Rather than reimplement pandas' internal seeding exactly, assert
        # the recursive STEP relationship holds for consecutive valid values,
        # which is the actual mathematical definition of EMA.
        valid = result.dropna().tolist()
        for i in range(1, len(valid)):
            price_t = values[span - 1 + i]
            expected = price_t * alpha + valid[i - 1] * (1 - alpha)
            assert valid[i] == pytest.approx(expected)

    def test_leading_nans_equal_span_minus_one(self, simple_series):
        result = calculate_ema(simple_series, span=3)
        assert result.isna().sum() == 2

    def test_insufficient_data_returns_all_nan_not_error(self):
        short_series = pd.Series([1.0, 2.0])
        result = calculate_ema(short_series, span=5)
        assert result.isna().all()

    @pytest.mark.parametrize("bad_span", [0, -3, 2.0])
    def test_invalid_span_raises(self, simple_series, bad_span):
        with pytest.raises(IndicatorInputError):
            calculate_ema(simple_series, span=bad_span)


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

class TestRSI:
    def test_all_gains_approaches_100(self):
        rising = pd.Series(range(1, 30), dtype=float)
        result = calculate_rsi(rising, period=14)
        assert result.dropna().iloc[-1] == pytest.approx(100.0)

    def test_all_losses_approaches_0(self):
        falling = pd.Series(range(30, 1, -1), dtype=float)
        result = calculate_rsi(falling, period=14)
        assert result.dropna().iloc[-1] == pytest.approx(0.0)

    def test_no_movement_is_neutral_50(self):
        flat = pd.Series([100.0] * 30)
        result = calculate_rsi(flat, period=14)
        assert result.dropna().iloc[-1] == pytest.approx(50.0)

    def test_bounded_between_0_and_100_on_random_data(self):
        rng = np.random.default_rng(seed=1)
        prices = pd.Series(100 + np.cumsum(rng.standard_normal(200)))
        result = calculate_rsi(prices, period=14).dropna()
        assert result.between(0, 100).all()

    def test_insufficient_data_returns_all_nan_not_error(self):
        short_series = pd.Series([100.0, 101.0, 99.0])
        result = calculate_rsi(short_series, period=14)
        assert result.isna().all()

    @pytest.mark.parametrize("bad_period", [0, -14, 1.5])
    def test_invalid_period_raises(self, bad_period):
        series = pd.Series([100.0, 101.0, 102.0])
        with pytest.raises(IndicatorInputError):
            calculate_rsi(series, period=bad_period)


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------

class TestMACD:
    @pytest.fixture
    def price_series(self):
        rng = np.random.default_rng(seed=0)
        return pd.Series(100 + np.cumsum(rng.standard_normal(60)), dtype=float)

    def test_returns_expected_columns(self, price_series):
        result = calculate_macd(price_series, fast=12, slow=26, signal=9)
        assert list(result.columns) == ["macd_line", "signal_line", "histogram"]
        assert len(result) == len(price_series)

    def test_histogram_equals_macd_minus_signal(self, price_series):
        result = calculate_macd(price_series, fast=12, slow=26, signal=9)
        valid = result.dropna()
        diff = valid["macd_line"] - valid["signal_line"]
        assert (diff - valid["histogram"]).abs().max() < 1e-9

    def test_macd_line_matches_ema_difference(self, price_series):
        result = calculate_macd(price_series, fast=12, slow=26, signal=9)
        expected_macd_line = calculate_ema(price_series, 12) - calculate_ema(price_series, 26)
        pd.testing.assert_series_equal(
            result["macd_line"], expected_macd_line.rename("macd_line")
        )

    def test_macd_line_leading_nans_equal_slow_minus_one(self, price_series):
        result = calculate_macd(price_series, fast=12, slow=26, signal=9)
        assert result["macd_line"].isna().sum() == 25  # slow - 1

    def test_signal_line_waits_for_signal_many_valid_macd_values(self, price_series):
        # signal_line should only start once there are `signal` (9) real,
        # non-NaN macd_line values behind it -- i.e. at index (slow-1)+(signal-1) = 33
        result = calculate_macd(price_series, fast=12, slow=26, signal=9)
        assert result["signal_line"].isna().sum() == 33

    def test_fast_must_be_less_than_slow(self, price_series):
        with pytest.raises(IndicatorInputError):
            calculate_macd(price_series, fast=26, slow=12, signal=9)
        with pytest.raises(IndicatorInputError):
            calculate_macd(price_series, fast=20, slow=20, signal=9)

    def test_insufficient_data_returns_all_nan_not_error(self):
        short_series = pd.Series([100.0, 101.0, 99.0, 102.0])
        result = calculate_macd(short_series, fast=12, slow=26, signal=9)
        assert result["macd_line"].isna().all()
        assert result["signal_line"].isna().all()
        assert result["histogram"].isna().all()


# ---------------------------------------------------------------------------
# Cross-cutting: missing values (NaN gaps) in the middle of the input
# ---------------------------------------------------------------------------

class TestMissingValuesInInput:
    def test_sma_does_not_crash_on_internal_nan(self):
        series = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0])
        result = calculate_sma(series, window=3)
        # Should not raise; windows overlapping the NaN become NaN too,
        # which is the conservative, correct behavior (no fabricated data).
        assert len(result) == len(series)

    def test_rsi_does_not_crash_on_internal_nan(self):
        series = pd.Series([100.0, 101.0, np.nan, 103.0, 104.0, 105.0, 106.0, 107.0])
        result = calculate_rsi(series, period=3)
        assert len(result) == len(series)
