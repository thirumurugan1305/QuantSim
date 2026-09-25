"""
End-to-end integration tests for QuantSim (Phase 9).

WHY THIS FILE EXISTS
---------------------
Every other test file in this project (test_indicators.py,
test_strategies.py, test_backtesting.py, test_performance_metrics.py,
test_database.py) is excellent at testing ONE layer in isolation, using
small hand-built fixtures. That's the right way to test a single layer
-- but it leaves one real gap: nothing proves that REAL output from one
layer, fed unmodified into the next, still behaves correctly all the
way through.

Concretely, before this file existed, no test ever:
  - ran a real Strategy through a real BacktestEngine and then fed that
    exact BacktestResult into calculate_performance_metrics()
  - saved a REAL (not hand-built) BacktestResult to SQLite and loaded
    it back, then recomputed metrics on the LOADED result to check they
    still match what was computed before saving

Every test below uses the project's own real classes and the bundled
sample CSV (deterministic, checked into the repo) -- never a
hand-crafted BacktestResult standing in for a real one. If something
about how two modules pass data to each other is subtly wrong (a dtype
mismatch, a precision loss through SQLite, a field the save/load path
forgets), a test built from matching hand-built fixtures at each layer
could easily miss it; a test using one real, continuous pipeline run
cannot.

WHY tmp_path FOR THE DATABASE
----------------------------------
Same reasoning as test_database.py: every test here gets its own
temporary SQLite file via pytest's `tmp_path` fixture, so nothing here
ever touches the real project database.
"""

from datetime import date

import pandas as pd
import pytest

from src.analytics.performance_metrics import calculate_performance_metrics
from src.backtesting.engine import run_backtest
from src.config import DATA_DIR
from src.database.repository import init_db, load_backtest_result, save_backtest_result
from src.strategies import MACDStrategy, MovingAverageCrossoverStrategy, RSIStrategy


@pytest.fixture
def sample_price_data() -> pd.DataFrame:
    """The real bundled sample dataset (500 synthetic-but-deterministic
    trading days) -- the same file the app itself falls back to when
    yfinance is unavailable. Using the actual shipped file (not a
    hand-built substitute) means these tests exercise the exact data
    shape `run_backtest()` sees in practice."""
    path = DATA_DIR / "sample_market_data.csv"
    return pd.read_csv(path, parse_dates=["Date"])


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "integration_test.db"
    init_db(path)
    return path


# ---------------------------------------------------------------------------
# Strategy -> Backtest -> Metrics consistency
# ---------------------------------------------------------------------------

class TestStrategyToBacktestToMetrics:
    """A real strategy's real signals, run through the real engine,
    fed into the real metrics function -- no hand-built BacktestResult
    anywhere in this class."""

    @pytest.mark.parametrize(
        "strategy_factory",
        [
            lambda: MovingAverageCrossoverStrategy(fast_window=20, slow_window=50),
            lambda: RSIStrategy(period=14, oversold=30, overbought=70),
            lambda: MACDStrategy(fast=12, slow=26, signal=9),
        ],
        ids=["MovingAverageCrossover", "RSI", "MACD"],
    )
    def test_full_pipeline_is_internally_consistent(self, sample_price_data, strategy_factory):
        strategy = strategy_factory()
        result = run_backtest(sample_price_data, strategy, initial_capital=10_000.0)
        metrics = calculate_performance_metrics(result)

        # The metrics module's trade count must exactly match the
        # engine's own count of completed trades -- these are computed
        # by two different functions in two different files; nothing
        # forces them to agree except both reading the same result.trades.
        assert metrics.num_trades == result.num_trades == len(result.trades)

        # Total return, independently recomputed from the same two raw
        # numbers the metrics function itself uses, must match exactly.
        expected_return = (result.final_value / result.initial_capital - 1) * 100
        assert metrics.total_return_pct == pytest.approx(expected_return)

        # has_open_position must agree between the engine's own field
        # and the metrics module's derived flag.
        assert metrics.has_open_position == (result.open_position is not None)

        # A real backtest over 500 days of data should produce a
        # non-trivial equity curve -- a basic sanity check that the
        # pipeline actually did something, not a silent no-op.
        assert len(result.equity_curve) == len(sample_price_data)

    def test_different_strategies_produce_different_trade_histories(self, sample_price_data):
        """A regression-style sanity check: three different strategies
        run on the SAME data must not coincidentally produce identical
        trade counts -- if they did, it would suggest something is
        wrong with how signals are being generated or consumed."""
        ma_result = run_backtest(
            sample_price_data, MovingAverageCrossoverStrategy(fast_window=20, slow_window=50), initial_capital=10_000.0
        )
        rsi_result = run_backtest(
            sample_price_data, RSIStrategy(period=14, oversold=30, overbought=70), initial_capital=10_000.0
        )
        macd_result = run_backtest(
            sample_price_data, MACDStrategy(fast=12, slow=26, signal=9), initial_capital=10_000.0
        )

        trade_counts = {ma_result.num_trades, rsi_result.num_trades, macd_result.num_trades}
        assert len(trade_counts) > 1, "three different strategies produced identical trade counts"


# ---------------------------------------------------------------------------
# Determinism / regression
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_running_the_same_backtest_twice_is_byte_identical(self, sample_price_data):
        """No randomness anywhere in this pipeline -- running the exact
        same strategy over the exact same data must produce identical
        results every time. This is the kind of property that's easy to
        accidentally break (e.g. by introducing a set() somewhere, whose
        iteration order isn't guaranteed) without any single-layer test
        noticing."""
        strategy_a = MovingAverageCrossoverStrategy(fast_window=20, slow_window=50)
        strategy_b = MovingAverageCrossoverStrategy(fast_window=20, slow_window=50)

        result_a = run_backtest(sample_price_data, strategy_a, initial_capital=10_000.0)
        result_b = run_backtest(sample_price_data, strategy_b, initial_capital=10_000.0)

        assert result_a.final_value == result_b.final_value
        assert result_a.num_trades == result_b.num_trades
        assert [t.pnl for t in result_a.trades] == [t.pnl for t in result_b.trades]

        metrics_a = calculate_performance_metrics(result_a)
        metrics_b = calculate_performance_metrics(result_b)
        assert metrics_a == metrics_b


# ---------------------------------------------------------------------------
# Real pipeline -> SQLite round trip
# ---------------------------------------------------------------------------

class TestRealPipelineDatabaseRoundTrip:
    """Saves and loads an ACTUAL run_backtest() output -- not a
    hand-built BacktestResult standing in for one -- and confirms
    metrics recomputed on the loaded copy still match the metrics
    computed on the original, in-memory result."""

    def test_real_backtest_round_trips_and_metrics_still_match(self, sample_price_data, db_path):
        strategy = MovingAverageCrossoverStrategy(fast_window=20, slow_window=50)
        result = run_backtest(sample_price_data, strategy, initial_capital=10_000.0)
        original_metrics = calculate_performance_metrics(result)

        bt_id = save_backtest_result(
            result, original_metrics, "AAPL", date(2023, 1, 2), date(2024, 11, 29), db_path=db_path
        )
        loaded = load_backtest_result(bt_id, db_path=db_path)

        assert loaded is not None
        assert loaded.final_value == pytest.approx(result.final_value)
        assert loaded.initial_capital == result.initial_capital
        assert len(loaded.trades) == len(result.trades)
        assert len(loaded.equity_curve) == len(result.equity_curve)

        # The real test: recompute metrics on the LOADED result using
        # the exact same Phase 6 function, and confirm every field
        # still matches what was computed before saving -- proving the
        # save/load path lost nothing that matters to the calculation.
        reloaded_metrics = calculate_performance_metrics(loaded)
        assert reloaded_metrics.total_return_pct == pytest.approx(original_metrics.total_return_pct)
        assert reloaded_metrics.num_trades == original_metrics.num_trades
        assert reloaded_metrics.win_rate_pct == original_metrics.win_rate_pct
        assert reloaded_metrics.max_drawdown_pct == pytest.approx(original_metrics.max_drawdown_pct)
        assert reloaded_metrics.sharpe_ratio == pytest.approx(original_metrics.sharpe_ratio)
        assert reloaded_metrics.has_open_position == original_metrics.has_open_position

    def test_real_backtest_with_open_position_round_trips(self, sample_price_data, db_path):
        """A short date slice is likely to end mid-trade (bought but not
        yet sold) -- using the first 60 real rows to get a realistic
        open-position scenario instead of hand-constructing one."""
        short_data = sample_price_data.iloc[:60].reset_index(drop=True)
        strategy = MovingAverageCrossoverStrategy(fast_window=5, slow_window=20)
        result = run_backtest(short_data, strategy, initial_capital=10_000.0)
        metrics = calculate_performance_metrics(result)

        bt_id = save_backtest_result(
            result, metrics, "AAPL", date(2023, 1, 2), date(2023, 3, 29), db_path=db_path
        )
        loaded = load_backtest_result(bt_id, db_path=db_path)

        assert loaded is not None
        assert (loaded.open_position is not None) == (result.open_position is not None)
        if result.open_position is not None:
            assert loaded.open_position.quantity == result.open_position.quantity
            assert loaded.open_position.entry_price == pytest.approx(result.open_position.entry_price)

    def test_real_backtest_with_zero_trades_round_trips(self, db_path):
        """A flat/tiny dataset that can't produce any crossover signals
        -- a real (if degenerate) pipeline run with zero completed
        trades, saved and loaded like any other."""
        flat_data = pd.DataFrame(
            {
                "Date": pd.date_range("2024-01-01", periods=10, freq="D"),
                "Open": [100.0] * 10,
                "High": [100.0] * 10,
                "Low": [100.0] * 10,
                "Close": [100.0] * 10,
                "Adj Close": [100.0] * 10,
                "Volume": [1_000_000] * 10,
            }
        )
        strategy = MovingAverageCrossoverStrategy(fast_window=2, slow_window=5)
        result = run_backtest(flat_data, strategy, initial_capital=10_000.0)
        assert result.num_trades == 0  # confirms this really is the zero-trade case being tested

        metrics = calculate_performance_metrics(result)
        bt_id = save_backtest_result(
            result, metrics, "FLAT", date(2024, 1, 1), date(2024, 1, 10), db_path=db_path
        )
        loaded = load_backtest_result(bt_id, db_path=db_path)

        assert loaded is not None
        assert loaded.trades == []
        reloaded_metrics = calculate_performance_metrics(loaded)
        assert reloaded_metrics.num_trades == 0
        assert reloaded_metrics.win_rate_pct is None
