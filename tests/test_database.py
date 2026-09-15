"""
Tests for src/database/repository.py

TEST DATABASE ISOLATION -- WHY FILE PATHS, NOT ":memory:"
--------------------------------------------------------------
This module's functions each open a NEW connection per call (see
repository.py's module docstring for why) and close it before
returning. SQLite's ":memory:" databases are private to the connection
that created them -- opening ":memory:" a second time gives you a
brand new, empty database, not the one you just set up. Since every
function here (init_db, save_backtest_result, load_backtest_result,
...) opens its own connection, using ":memory:" would silently lose
all data between calls. Instead, every test gets a real temporary FILE
path via pytest's built-in `tmp_path` fixture, which persists correctly
across multiple connect()/close() cycles and is automatically cleaned
up by pytest afterward. This never touches the real project database
at DB_PATH.

Run with:
    pytest tests/test_database.py -v
"""

import sqlite3
from datetime import date

import pandas as pd
import pytest

from src.analytics.performance_metrics import calculate_performance_metrics
from src.backtesting.models import BacktestResult, OpenPosition, Trade
from src.database.repository import (
    delete_backtest_result,
    get_connection,
    init_db,
    list_saved_backtests,
    load_backtest_result,
    save_backtest_result,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def db_path(tmp_path):
    """A fresh, isolated SQLite file path for one test, with the schema
    already created."""
    path = tmp_path / "test_quantsim.db"
    init_db(path)
    return path


def make_equity_curve(total_values: list[float]) -> pd.DataFrame:
    n = len(total_values)
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="D") if n else pd.Series([], dtype="datetime64[ns]"),
            "price": total_values,
            "signal": ["HOLD"] * n,
            "cash": [0.0] * n,
            "shares_held": [0] * n,
            "holdings_value": [0.0] * n,
            "total_value": total_values,
        }
    )


def make_trade(entry_price=10.0, exit_price=20.0, quantity=100) -> Trade:
    return Trade(
        entry_date=pd.Timestamp("2024-01-01"),
        entry_price=entry_price,
        exit_date=pd.Timestamp("2024-01-02"),
        exit_price=exit_price,
        quantity=quantity,
        entry_signal="BUY",
        exit_signal="SELL",
    )


def make_result(
    initial_capital=1000.0,
    final_value=1500.0,
    trades=None,
    total_values=None,
    open_position=None,
    strategy_name="TestStrategy",
    strategy_params=None,
) -> BacktestResult:
    return BacktestResult(
        initial_capital=initial_capital,
        final_value=final_value,
        trades=trades or [],
        equity_curve=make_equity_curve(total_values if total_values is not None else [1000.0, 1500.0]),
        open_position=open_position,
        strategy_name=strategy_name,
        strategy_params=strategy_params or {},
    )


# ---------------------------------------------------------------------------
# Schema initialization
# ---------------------------------------------------------------------------

class TestSchemaInitialization:
    def test_all_tables_created(self, tmp_path):
        path = tmp_path / "fresh.db"
        init_db(path)
        conn = get_connection(path)
        tables = {
            r["name"]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        conn.close()
        assert {"backtests", "trades", "portfolio_snapshots", "performance_metrics", "schema_version"} <= tables

    def test_calling_init_db_twice_is_safe(self, tmp_path):
        path = tmp_path / "double_init.db"
        init_db(path)
        init_db(path)  # must not raise, must not duplicate schema_version rows
        conn = get_connection(path)
        count = conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0]
        conn.close()
        assert count == 1

    def test_init_db_does_not_wipe_existing_data(self, db_path):
        result = make_result()
        metrics = calculate_performance_metrics(result)
        bt_id = save_backtest_result(result, metrics, "AAPL", date(2024, 1, 1), date(2024, 1, 2), db_path=db_path)

        init_db(db_path)  # re-run initialization on a populated database

        assert load_backtest_result(bt_id, db_path=db_path) is not None

    def test_schema_version_seeded_to_1(self, db_path):
        conn = get_connection(db_path)
        version = conn.execute("SELECT version FROM schema_version").fetchone()[0]
        conn.close()
        assert version == 1


# ---------------------------------------------------------------------------
# Save / round-trip
# ---------------------------------------------------------------------------

class TestSaveAndLoadRoundTrip:
    def test_save_returns_an_id(self, db_path):
        result = make_result()
        metrics = calculate_performance_metrics(result)
        bt_id = save_backtest_result(result, metrics, "AAPL", date(2024, 1, 1), date(2024, 1, 2), db_path=db_path)
        assert isinstance(bt_id, int)
        assert bt_id > 0

    def test_round_trip_scalar_fields(self, db_path):
        result = make_result(initial_capital=2500.0, final_value=3000.0, strategy_name="MACDStrategy",
                              strategy_params={"fast": 12, "slow": 26, "signal": 9})
        metrics = calculate_performance_metrics(result)
        bt_id = save_backtest_result(result, metrics, "MSFT", date(2024, 1, 1), date(2024, 6, 1), db_path=db_path)

        loaded = load_backtest_result(bt_id, db_path=db_path)
        assert loaded.initial_capital == 2500.0
        assert loaded.final_value == 3000.0
        assert loaded.strategy_name == "MACDStrategy"

    def test_json_strategy_params_round_trip(self, db_path):
        params = {"fast_window": 20, "slow_window": 50, "price_column": "Adj Close"}
        result = make_result(strategy_params=params)
        metrics = calculate_performance_metrics(result)
        bt_id = save_backtest_result(result, metrics, "AAPL", date(2024, 1, 1), date(2024, 1, 2), db_path=db_path)

        loaded = load_backtest_result(bt_id, db_path=db_path)
        assert loaded.strategy_params == params

    def test_completed_trades_round_trip(self, db_path):
        trades = [make_trade(10, 20, 100), make_trade(20, 15, 50)]
        result = make_result(trades=trades)
        metrics = calculate_performance_metrics(result)
        bt_id = save_backtest_result(result, metrics, "AAPL", date(2024, 1, 1), date(2024, 1, 2), db_path=db_path)

        loaded = load_backtest_result(bt_id, db_path=db_path)
        assert len(loaded.trades) == 2
        assert loaded.trades[0].entry_price == 10.0
        assert loaded.trades[0].exit_price == 20.0
        assert loaded.trades[1].quantity == 50

    def test_derived_trade_properties_correct_after_reload(self, db_path):
        trades = [make_trade(10, 20, 100)]  # pnl should be +1000, a win
        result = make_result(trades=trades)
        metrics = calculate_performance_metrics(result)
        bt_id = save_backtest_result(result, metrics, "AAPL", date(2024, 1, 1), date(2024, 1, 2), db_path=db_path)

        loaded = load_backtest_result(bt_id, db_path=db_path)
        # pnl/pnl_pct/is_win are NOT stored -- these are recomputed fresh
        # from entry/exit price and quantity by the Trade dataclass itself.
        assert loaded.trades[0].pnl == 1000.0
        assert loaded.trades[0].pnl_pct == pytest.approx(100.0)
        assert loaded.trades[0].is_win is True

    def test_zero_trades_round_trips_to_empty_list_not_none(self, db_path):
        result = make_result(trades=[])
        metrics = calculate_performance_metrics(result)
        bt_id = save_backtest_result(result, metrics, "AAPL", date(2024, 1, 1), date(2024, 1, 2), db_path=db_path)

        loaded = load_backtest_result(bt_id, db_path=db_path)
        assert loaded.trades == []

    def test_equity_curve_round_trip(self, db_path):
        values = [1000.0, 1100.0, 1050.0, 1200.0]
        result = make_result(total_values=values)
        metrics = calculate_performance_metrics(result)
        bt_id = save_backtest_result(result, metrics, "AAPL", date(2024, 1, 1), date(2024, 1, 4), db_path=db_path)

        loaded = load_backtest_result(bt_id, db_path=db_path)
        assert loaded.equity_curve["total_value"].tolist() == values
        assert list(loaded.equity_curve.columns) == [
            "date", "price", "signal", "cash", "shares_held", "holdings_value", "total_value",
        ]

    def test_large_equity_curve_round_trip(self, db_path):
        # Matches the real sample dataset's scale (~500 trading days).
        values = [100.0 + i * 0.1 for i in range(500)]
        result = make_result(total_values=values)
        metrics = calculate_performance_metrics(result)
        bt_id = save_backtest_result(result, metrics, "AAPL", date(2023, 1, 1), date(2024, 11, 1), db_path=db_path)

        loaded = load_backtest_result(bt_id, db_path=db_path)
        assert len(loaded.equity_curve) == 500
        assert loaded.equity_curve["total_value"].iloc[-1] == pytest.approx(values[-1])

    def test_open_position_round_trip(self, db_path):
        op = OpenPosition(
            entry_date=pd.Timestamp("2024-01-05"),
            entry_price=42.5,
            quantity=17,
            entry_signal="BUY",
        )
        result = make_result(open_position=op)
        metrics = calculate_performance_metrics(result)
        bt_id = save_backtest_result(result, metrics, "AAPL", date(2024, 1, 1), date(2024, 1, 5), db_path=db_path)

        loaded = load_backtest_result(bt_id, db_path=db_path)
        assert loaded.open_position is not None
        assert loaded.open_position.entry_price == 42.5
        assert loaded.open_position.quantity == 17
        assert str(loaded.open_position.entry_signal) == "BUY"

    def test_no_open_position_round_trips_to_none(self, db_path):
        result = make_result(open_position=None)
        metrics = calculate_performance_metrics(result)
        bt_id = save_backtest_result(result, metrics, "AAPL", date(2024, 1, 1), date(2024, 1, 2), db_path=db_path)

        loaded = load_backtest_result(bt_id, db_path=db_path)
        assert loaded.open_position is None

    def test_stored_performance_metrics_are_correct(self, db_path):
        trades = [make_trade(10, 20, 100)]
        result = make_result(trades=trades, initial_capital=1000.0, final_value=2000.0)
        metrics = calculate_performance_metrics(result, risk_free_rate=0.03)
        bt_id = save_backtest_result(result, metrics, "AAPL", date(2024, 1, 1), date(2024, 1, 2), db_path=db_path)

        conn = get_connection(db_path)
        row = conn.execute("SELECT * FROM performance_metrics WHERE backtest_id = ?", (bt_id,)).fetchone()
        conn.close()
        assert row["total_return_pct"] == pytest.approx(metrics.total_return_pct)
        assert row["win_rate_pct"] == pytest.approx(100.0)
        assert row["risk_free_rate_annual"] == pytest.approx(0.03)

    def test_risk_free_rate_persists_correctly(self, db_path):
        result = make_result()
        metrics = calculate_performance_metrics(result, risk_free_rate=0.045)
        bt_id = save_backtest_result(result, metrics, "AAPL", date(2024, 1, 1), date(2024, 1, 2), db_path=db_path)

        conn = get_connection(db_path)
        row = conn.execute(
            "SELECT risk_free_rate_annual FROM performance_metrics WHERE backtest_id = ?", (bt_id,)
        ).fetchone()
        conn.close()
        assert row["risk_free_rate_annual"] == pytest.approx(0.045)


# ---------------------------------------------------------------------------
# Multiple backtests / isolation between them
# ---------------------------------------------------------------------------

class TestMultipleBacktestsNoCrossContamination:
    def test_multiple_backtests_dont_share_trades(self, db_path):
        result1 = make_result(trades=[make_trade(10, 20)], strategy_name="A")
        result2 = make_result(trades=[make_trade(5, 15), make_trade(15, 10)], strategy_name="B")
        m1 = calculate_performance_metrics(result1)
        m2 = calculate_performance_metrics(result2)

        id1 = save_backtest_result(result1, m1, "AAPL", date(2024, 1, 1), date(2024, 1, 2), db_path=db_path)
        id2 = save_backtest_result(result2, m2, "MSFT", date(2024, 1, 1), date(2024, 1, 2), db_path=db_path)

        loaded1 = load_backtest_result(id1, db_path=db_path)
        loaded2 = load_backtest_result(id2, db_path=db_path)

        assert len(loaded1.trades) == 1
        assert len(loaded2.trades) == 2
        assert loaded1.strategy_name == "A"
        assert loaded2.strategy_name == "B"

    def test_list_returns_all_saved_backtests(self, db_path):
        for i in range(3):
            result = make_result()
            metrics = calculate_performance_metrics(result)
            save_backtest_result(result, metrics, f"TICK{i}", date(2024, 1, 1), date(2024, 1, 2), db_path=db_path)

        listing = list_saved_backtests(db_path=db_path)
        assert len(listing) == 3
        tickers = {row["ticker"] for row in listing}
        assert tickers == {"TICK0", "TICK1", "TICK2"}

    def test_list_sorted_newest_first(self, db_path):
        ids = []
        for i in range(3):
            result = make_result()
            metrics = calculate_performance_metrics(result)
            bt_id = save_backtest_result(result, metrics, f"T{i}", date(2024, 1, 1), date(2024, 1, 2), db_path=db_path)
            ids.append(bt_id)

        listing = list_saved_backtests(db_path=db_path)
        listed_ids = [row["id"] for row in listing]
        assert listed_ids == list(reversed(ids))

    def test_list_includes_required_fields(self, db_path):
        result = make_result(strategy_name="RSIStrategy")
        metrics = calculate_performance_metrics(result)
        save_backtest_result(result, metrics, "AAPL", date(2024, 1, 1), date(2024, 3, 1), db_path=db_path)

        listing = list_saved_backtests(db_path=db_path)
        row = listing[0]
        for field in ["id", "created_at", "ticker", "strategy_name", "start_date", "end_date", "total_return_pct"]:
            assert field in row


# ---------------------------------------------------------------------------
# Empty database / missing records
# ---------------------------------------------------------------------------

class TestEmptyDatabaseAndMissingRecords:
    def test_list_on_empty_database_returns_empty_list(self, db_path):
        assert list_saved_backtests(db_path=db_path) == []

    def test_loading_nonexistent_id_returns_none(self, db_path):
        assert load_backtest_result(999, db_path=db_path) is None

    def test_deleting_nonexistent_id_returns_false(self, db_path):
        assert delete_backtest_result(999, db_path=db_path) is False


# ---------------------------------------------------------------------------
# Deletion / cascade
# ---------------------------------------------------------------------------

class TestDeletion:
    def test_deleting_existing_id_returns_true(self, db_path):
        result = make_result()
        metrics = calculate_performance_metrics(result)
        bt_id = save_backtest_result(result, metrics, "AAPL", date(2024, 1, 1), date(2024, 1, 2), db_path=db_path)
        assert delete_backtest_result(bt_id, db_path=db_path) is True

    def test_deleted_backtest_no_longer_loads(self, db_path):
        result = make_result()
        metrics = calculate_performance_metrics(result)
        bt_id = save_backtest_result(result, metrics, "AAPL", date(2024, 1, 1), date(2024, 1, 2), db_path=db_path)
        delete_backtest_result(bt_id, db_path=db_path)
        assert load_backtest_result(bt_id, db_path=db_path) is None

    def test_cascade_deletes_trades_snapshots_and_metrics(self, db_path):
        trades = [make_trade(10, 20), make_trade(20, 15)]
        result = make_result(trades=trades, total_values=[1000, 1100, 1050])
        metrics = calculate_performance_metrics(result)
        bt_id = save_backtest_result(result, metrics, "AAPL", date(2024, 1, 1), date(2024, 1, 3), db_path=db_path)

        delete_backtest_result(bt_id, db_path=db_path)

        conn = get_connection(db_path)
        trades_left = conn.execute("SELECT COUNT(*) FROM trades WHERE backtest_id = ?", (bt_id,)).fetchone()[0]
        snaps_left = conn.execute(
            "SELECT COUNT(*) FROM portfolio_snapshots WHERE backtest_id = ?", (bt_id,)
        ).fetchone()[0]
        metrics_left = conn.execute(
            "SELECT COUNT(*) FROM performance_metrics WHERE backtest_id = ?", (bt_id,)
        ).fetchone()[0]
        conn.close()
        assert trades_left == 0
        assert snaps_left == 0
        assert metrics_left == 0

    def test_deleting_one_backtest_does_not_affect_another(self, db_path):
        result1 = make_result(trades=[make_trade(10, 20)])
        result2 = make_result(trades=[make_trade(5, 15)])
        m1 = calculate_performance_metrics(result1)
        m2 = calculate_performance_metrics(result2)
        id1 = save_backtest_result(result1, m1, "AAPL", date(2024, 1, 1), date(2024, 1, 2), db_path=db_path)
        id2 = save_backtest_result(result2, m2, "MSFT", date(2024, 1, 1), date(2024, 1, 2), db_path=db_path)

        delete_backtest_result(id1, db_path=db_path)

        assert load_backtest_result(id1, db_path=db_path) is None
        still_there = load_backtest_result(id2, db_path=db_path)
        assert still_there is not None
        assert len(still_there.trades) == 1


# ---------------------------------------------------------------------------
# Foreign key enforcement / transaction rollback
# ---------------------------------------------------------------------------

class TestForeignKeysAndTransactions:
    def test_foreign_keys_pragma_is_active(self, db_path):
        """A direct proof that PRAGMA foreign_keys = ON is really in
        effect on connections from get_connection() -- inserting a trade
        referencing a backtest_id that doesn't exist must fail."""
        conn = get_connection(db_path)
        conn.execute(
            """INSERT INTO backtests (created_at, ticker, start_date, end_date,
               strategy_name, strategy_params, initial_capital, final_value,
               has_open_position) VALUES ('2024-01-01','AAPL','2024-01-01',
               '2024-01-02','X','{}',1000,1000,0)"""
        )
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """INSERT INTO trades (backtest_id, entry_date, entry_price,
                   exit_date, exit_price, quantity, entry_signal, exit_signal)
                   VALUES (999999, '2024-01-01', 10, '2024-01-02', 20, 100, 'BUY', 'SELL')"""
            )
        conn.close()

    def test_save_rolls_back_completely_on_failure(self, db_path):
        """A failure partway through save_backtest_result (here: a trade
        with quantity=None, violating trades.quantity NOT NULL, which
        only happens at the SQL layer -- entry/exit price stay valid so
        Trade.pnl/is_win compute fine and don't crash beforehand) must
        undo the backtests row that was already inserted in the same
        transaction -- proving this is genuinely one atomic transaction,
        not four independent inserts.
        """
        trade = make_trade(10, 20, 100)
        result = make_result(trades=[trade])
        metrics = calculate_performance_metrics(result)
        trade.quantity = None  # corrupt AFTER computing valid metrics

        with pytest.raises(sqlite3.IntegrityError):
            save_backtest_result(result, metrics, "AAPL", date(2024, 1, 1), date(2024, 1, 2), db_path=db_path)

        conn = get_connection(db_path)
        count = conn.execute("SELECT COUNT(*) FROM backtests").fetchone()[0]
        conn.close()
        assert count == 0, "the backtests row inserted before the failure must have been rolled back"


# ---------------------------------------------------------------------------
# Test isolation
# ---------------------------------------------------------------------------

class TestIsolation:
    def test_each_test_gets_a_fresh_database(self, db_path):
        """This test intentionally does nothing but assert the database
        is empty -- if a previous test's data ever leaked in here (e.g.
        from accidentally sharing a path or a module-level connection),
        this would fail."""
        assert list_saved_backtests(db_path=db_path) == []

    def test_two_explicit_temp_paths_are_independent(self, tmp_path):
        path_a = tmp_path / "a.db"
        path_b = tmp_path / "b.db"
        init_db(path_a)
        init_db(path_b)

        result = make_result()
        metrics = calculate_performance_metrics(result)
        save_backtest_result(result, metrics, "AAPL", date(2024, 1, 1), date(2024, 1, 2), db_path=path_a)

        assert len(list_saved_backtests(db_path=path_a)) == 1
        assert len(list_saved_backtests(db_path=path_b)) == 0
