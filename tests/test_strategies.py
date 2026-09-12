"""
Tests for the strategy layer (src/strategies/).

Covers, across the shared crossover helper and all three strategies:
  - normal BUY / SELL / HOLD conditions
  - crossover detection correctness (constructed, hand-verifiable data)
  - invalid-parameter handling
  - insufficient-data handling (HOLD, never a crash)
  - that no strategy uses future data (look-ahead bias check)

Run with:
    pytest tests/test_strategies.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.strategies.base_strategy import (
    BaseStrategy,
    Signal,
    StrategyInputError,
    detect_crossover,
)
from src.strategies.macd_strategy import MACDStrategy
from src.strategies.moving_average_strategy import MovingAverageCrossoverStrategy
from src.strategies.rsi_strategy import RSIStrategy


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def make_price_df(prices: list[float]) -> pd.DataFrame:
    """Build a minimal price_data DataFrame with a Date + Adj Close column,
    the shape every strategy's `generate_signals()` expects."""
    dates = pd.date_range("2024-01-01", periods=len(prices), freq="D")
    return pd.DataFrame({"Date": dates, "Adj Close": prices})


@pytest.fixture
def trending_then_reversing_prices() -> pd.DataFrame:
    """A price path that settles first, then trends up, then trends down.

    The settling period at the start matters: without it, a series that
    trends up from row 0 has its fast SMA already leading the slow SMA
    before the slow SMA even finishes its warm-up, so there is no earlier
    "fast below slow" state left to cross up FROM — a BUY crossover
    becomes mathematically impossible to observe, even though the
    strategy logic is correct. This shape guarantees both a genuine BUY
    (uptrend begins) and a genuine SELL (downtrend begins) occur within
    the test window.
    """
    rng = np.random.default_rng(seed=42)
    flat = 100 + rng.normal(0, 0.2, 20)
    up = flat[-1] + np.arange(1, 41) * 0.8 + rng.normal(0, 0.3, 40)
    down = up[-1] - np.arange(1, 41) * 0.8 + rng.normal(0, 0.3, 40)
    prices = np.concatenate([flat, up, down])
    return make_price_df(prices.tolist())


# ---------------------------------------------------------------------------
# detect_crossover (shared helper)
# ---------------------------------------------------------------------------

class TestDetectCrossover:
    def test_buy_on_cross_above(self):
        fast = pd.Series([1, 2, 3, 5, 5], dtype=float)
        slow = pd.Series([4, 4, 4, 4, 4], dtype=float)
        result = detect_crossover(fast, slow)
        assert result.iloc[3] == Signal.BUY
        assert result.iloc[0] == Signal.HOLD

    def test_sell_on_cross_below(self):
        fast = pd.Series([5, 5, 4, 2, 2], dtype=float)
        slow = pd.Series([4, 4, 4, 4, 4], dtype=float)
        result = detect_crossover(fast, slow)
        # index 2: fast(4) == slow(4) is "touching", not yet below -> HOLD
        # index 3: fast(2) < slow(4) and fast_prev(4) >= slow_prev(4) -> SELL
        assert result.iloc[2] == Signal.HOLD
        assert result.iloc[3] == Signal.SELL

    def test_no_crossover_stays_hold(self):
        fast = pd.Series([1, 2, 3, 4, 5], dtype=float)
        slow = pd.Series([10, 10, 10, 10, 10], dtype=float)
        result = detect_crossover(fast, slow)
        assert (result == Signal.HOLD).all()

    def test_equal_then_diverge_is_not_a_false_signal(self):
        # touching exactly (fast == slow) should not itself count as a
        # cross in either direction until it actually goes past
        fast = pd.Series([3, 4, 4, 5], dtype=float)
        slow = pd.Series([4, 4, 4, 4], dtype=float)
        result = detect_crossover(fast, slow)
        # index 3: fast(5) > slow(4) and fast_prev(4) <= slow_prev(4) -> BUY
        assert result.iloc[3] == Signal.BUY
        assert result.iloc[1] == Signal.HOLD
        assert result.iloc[2] == Signal.HOLD

    def test_nan_rows_never_produce_a_signal(self):
        fast = pd.Series([np.nan, np.nan, 5.0, 6.0])
        slow = pd.Series([np.nan, np.nan, 4.0, 4.0])
        result = detect_crossover(fast, slow)
        assert result.iloc[0] == Signal.HOLD
        assert result.iloc[1] == Signal.HOLD

    def test_mismatched_length_raises(self):
        with pytest.raises(StrategyInputError):
            detect_crossover(pd.Series([1, 2, 3]), pd.Series([1, 2]))


# ---------------------------------------------------------------------------
# BaseStrategy contract
# ---------------------------------------------------------------------------

class TestBaseStrategy:
    def test_cannot_instantiate_abstract_base(self):
        with pytest.raises(TypeError):
            BaseStrategy()  # abstract method generate_signals not implemented

    def test_missing_price_column_raises(self):
        strat = MovingAverageCrossoverStrategy(fast_window=2, slow_window=3)
        bad_df = pd.DataFrame({"Date": pd.date_range("2024-01-01", periods=5), "Close": [1, 2, 3, 4, 5]})
        with pytest.raises(StrategyInputError):
            strat.generate_signals(bad_df)

    def test_non_dataframe_input_raises(self):
        strat = MovingAverageCrossoverStrategy(fast_window=2, slow_window=3)
        with pytest.raises(StrategyInputError):
            strat.generate_signals([1, 2, 3])

    def test_describe_reports_name_and_params(self):
        strat = RSIStrategy(period=10, oversold=25, overbought=75)
        info = strat.describe()
        assert info["name"] == "RSIStrategy"
        assert info["params"]["period"] == 10
        assert info["params"]["oversold"] == 25
        assert info["params"]["overbought"] == 75


# ---------------------------------------------------------------------------
# Moving Average Crossover Strategy
# ---------------------------------------------------------------------------

class TestMovingAverageCrossoverStrategy:
    def test_buy_and_sell_signals_appear_on_trend_reversal(self, trending_then_reversing_prices):
        strat = MovingAverageCrossoverStrategy(fast_window=5, slow_window=10)
        signals = strat.generate_signals(trending_then_reversing_prices)
        assert (signals == Signal.BUY).any()
        assert (signals == Signal.SELL).any()
        assert (signals == Signal.HOLD).any()

    def test_flat_price_series_never_signals(self):
        df = make_price_df([100.0] * 60)
        strat = MovingAverageCrossoverStrategy(fast_window=5, slow_window=10)
        signals = strat.generate_signals(df)
        assert (signals == Signal.HOLD).all()

    def test_insufficient_data_returns_all_hold_not_crash(self):
        df = make_price_df([100.0, 101.0, 102.0])  # far fewer than slow_window
        strat = MovingAverageCrossoverStrategy(fast_window=20, slow_window=50)
        signals = strat.generate_signals(df)
        assert len(signals) == 3
        assert (signals == Signal.HOLD).all()

    def test_fast_must_be_less_than_slow(self):
        with pytest.raises(StrategyInputError):
            MovingAverageCrossoverStrategy(fast_window=50, slow_window=20)
        with pytest.raises(StrategyInputError):
            MovingAverageCrossoverStrategy(fast_window=20, slow_window=20)

    @pytest.mark.parametrize("bad_value", [0, -5, 3.5])
    def test_invalid_window_raises(self, bad_value):
        with pytest.raises(StrategyInputError):
            MovingAverageCrossoverStrategy(fast_window=bad_value, slow_window=50)


# ---------------------------------------------------------------------------
# RSI Strategy
# ---------------------------------------------------------------------------

class TestRSIStrategy:
    def test_buy_when_relentlessly_oversold(self):
        # a strictly falling price series drives RSI toward 0, well below
        # the default oversold threshold of 30
        df = make_price_df(list(range(100, 60, -1)))
        strat = RSIStrategy(period=14, oversold=30, overbought=70)
        signals = strat.generate_signals(df)
        assert (signals.iloc[-5:] == Signal.BUY).all()

    def test_sell_when_relentlessly_overbought(self):
        df = make_price_df(list(range(60, 100)))
        strat = RSIStrategy(period=14, oversold=30, overbought=70)
        signals = strat.generate_signals(df)
        assert (signals.iloc[-5:] == Signal.SELL).all()

    def test_hold_in_neutral_zone(self):
        df = make_price_df([100.0] * 30)  # flat -> RSI == 50, neutral
        strat = RSIStrategy(period=14, oversold=30, overbought=70)
        signals = strat.generate_signals(df)
        assert (signals == Signal.HOLD).all()

    def test_insufficient_data_returns_all_hold_not_crash(self):
        df = make_price_df([100.0, 101.0, 99.0])
        strat = RSIStrategy(period=14)
        signals = strat.generate_signals(df)
        assert len(signals) == 3
        assert (signals == Signal.HOLD).all()

    def test_invalid_threshold_order_raises(self):
        with pytest.raises(StrategyInputError):
            RSIStrategy(oversold=70, overbought=30)  # swapped

    def test_threshold_out_of_range_raises(self):
        with pytest.raises(StrategyInputError):
            RSIStrategy(oversold=-10, overbought=70)
        with pytest.raises(StrategyInputError):
            RSIStrategy(oversold=30, overbought=150)

    @pytest.mark.parametrize("bad_period", [0, -14, 2.5])
    def test_invalid_period_raises(self, bad_period):
        with pytest.raises(StrategyInputError):
            RSIStrategy(period=bad_period)


# ---------------------------------------------------------------------------
# MACD Strategy
# ---------------------------------------------------------------------------

class TestMACDStrategy:
    def test_buy_and_sell_signals_appear_on_trend_reversal(self, trending_then_reversing_prices):
        strat = MACDStrategy(fast=5, slow=10, signal=3)
        signals = strat.generate_signals(trending_then_reversing_prices)
        assert (signals == Signal.BUY).any()
        assert (signals == Signal.SELL).any()

    def test_flat_price_series_never_signals(self):
        df = make_price_df([100.0] * 60)
        strat = MACDStrategy(fast=5, slow=10, signal=3)
        signals = strat.generate_signals(df)
        assert (signals == Signal.HOLD).all()

    def test_insufficient_data_returns_all_hold_not_crash(self):
        df = make_price_df([100.0, 101.0, 99.0, 102.0])
        strat = MACDStrategy(fast=12, slow=26, signal=9)
        signals = strat.generate_signals(df)
        assert len(signals) == 4
        assert (signals == Signal.HOLD).all()

    def test_fast_must_be_less_than_slow(self):
        with pytest.raises(StrategyInputError):
            MACDStrategy(fast=26, slow=12, signal=9)

    @pytest.mark.parametrize("bad_value", [0, -1, 1.5])
    def test_invalid_signal_period_raises(self, bad_value):
        with pytest.raises(StrategyInputError):
            MACDStrategy(fast=12, slow=26, signal=bad_value)


# ---------------------------------------------------------------------------
# Look-ahead bias: signals must not change when future rows are added
# ---------------------------------------------------------------------------

class TestNoLookAheadBias:
    """The core check: a signal at row i, computed on data truncated right
    after row i, must be IDENTICAL to that same row's signal when computed
    on the full dataset. If a strategy were peeking at future rows, cutting
    the data short would change historical signals -- it must not.
    """

    @pytest.mark.parametrize(
        "strategy_factory",
        [
            lambda: MovingAverageCrossoverStrategy(fast_window=5, slow_window=10),
            lambda: RSIStrategy(period=14, oversold=30, overbought=70),
            lambda: MACDStrategy(fast=5, slow=10, signal=3),
        ],
    )
    def test_truncated_data_yields_identical_past_signals(
        self, trending_then_reversing_prices, strategy_factory
    ):
        full_df = trending_then_reversing_prices
        strat_full = strategy_factory()
        full_signals = strat_full.generate_signals(full_df)

        check_points = [20, 40, 60, len(full_df) - 1]
        for cutoff in check_points:
            truncated_df = full_df.iloc[: cutoff + 1].reset_index(drop=True)
            strat_truncated = strategy_factory()
            truncated_signals = strat_truncated.generate_signals(truncated_df)

            assert truncated_signals.iloc[-1] == full_signals.iloc[cutoff], (
                f"Signal at row {cutoff} changed when future rows were added "
                f"-- possible look-ahead bias."
            )
