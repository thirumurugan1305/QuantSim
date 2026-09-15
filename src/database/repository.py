"""
SQLite persistence layer for QuantSim.

WHAT THIS MODULE DOES
----------------------
Saves a completed `BacktestResult` (Phase 5) + its `PerformanceMetrics`
(Phase 6) to a local SQLite database, and loads them back. This is a
pure data-access layer: it has no Streamlit imports, no strategy logic,
and no backtesting logic — it only knows how to translate the existing
dataclasses to and from SQL rows. That separation means this whole
module can be tested (and IS tested, in tests/test_database.py) without
ever starting the Streamlit app.

WHY NO ORM
------------
This project deliberately uses Python's built-in `sqlite3` module with
hand-written SQL rather than SQLAlchemy or similar. For a project this
size, an ORM would add a whole new concept to learn (models, sessions,
migrations, query builders) for a benefit — abstracting over SQL — that
doesn't matter when there's exactly one supported database (SQLite) and
four small, stable tables. Plain SQL here is also more transparent for
learning: every column in the schema below is a column you can see.

CONNECTION LIFECYCLE
------------------------
Every public function in this module opens its own connection, does its
work, and closes the connection before returning — connections are
never held open across Streamlit reruns or cached in session state.
This is deliberate: Streamlit re-executes the whole script on every
interaction, and a long-lived connection sitting in `st.session_state`
would be easy to leak or accidentally share across reruns in confusing
ways. Opening a short-lived connection per operation is slightly less
"efficient" in the abstract, but SQLite connections are cheap to open,
and the simplicity is worth far more than the performance here.

WHY PRAGMA foreign_keys = ON MATTERS
----------------------------------------
SQLite does NOT enforce foreign key constraints by default, even though
the `REFERENCES ... ON DELETE CASCADE` syntax is right there in the
schema — silently ignoring it unless a session explicitly turns it on
with `PRAGMA foreign_keys = ON` is one of SQLite's best-known gotchas.
`get_connection()` below sets this on EVERY connection it creates, so
it's structurally impossible for some other code path to forget it.

DERIVED DATA IS NEVER STORED
--------------------------------
`Trade.pnl`, `Trade.pnl_pct`, and `Trade.is_win` are `@property`
computations on the `Trade` dataclass (see `src/backtesting/models.py`)
-- they are intentionally NOT columns in the `trades` table. Storing
them would duplicate business logic that already lives in `Trade`, and
risks the stored copy silently disagreeing with the dataclass if either
one ever changes. Recomputing them after loading (from the 4 raw numbers
that ARE stored: entry/exit price and quantity) is trivial and always
correct by construction.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

from src.analytics.performance_metrics import PerformanceMetrics
from src.backtesting.models import (
    EQUITY_CURVE_COLUMNS,
    BacktestResult,
    OpenPosition,
    Trade,
)
from src.config import DB_PATH
from src.strategies.base_strategy import Signal

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

# Every statement uses IF NOT EXISTS, so running this against an existing,
# populated database is always safe -- it creates what's missing and
# touches nothing that already exists. No migration framework: if the
# schema ever needs to change, the documented approach for this local,
# disposable, educational database is to delete the .db file and let
# init_db() recreate it from scratch (see README "Limitations").
_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS backtests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    ticker TEXT NOT NULL,
    start_date TEXT NOT NULL,
    end_date TEXT NOT NULL,
    strategy_name TEXT NOT NULL,
    strategy_params TEXT NOT NULL,
    initial_capital REAL NOT NULL,
    final_value REAL NOT NULL,
    has_open_position INTEGER NOT NULL,
    open_position_entry_date TEXT,
    open_position_entry_price REAL,
    open_position_quantity INTEGER,
    open_position_entry_signal TEXT
);

CREATE INDEX IF NOT EXISTS idx_backtests_created_at ON backtests(created_at);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    backtest_id INTEGER NOT NULL REFERENCES backtests(id) ON DELETE CASCADE,
    entry_date TEXT NOT NULL,
    entry_price REAL NOT NULL,
    exit_date TEXT NOT NULL,
    exit_price REAL NOT NULL,
    quantity INTEGER NOT NULL,
    entry_signal TEXT NOT NULL,
    exit_signal TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_trades_backtest_id ON trades(backtest_id);

CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    backtest_id INTEGER NOT NULL REFERENCES backtests(id) ON DELETE CASCADE,
    date TEXT NOT NULL,
    price REAL NOT NULL,
    signal TEXT NOT NULL,
    cash REAL NOT NULL,
    shares_held INTEGER NOT NULL,
    holdings_value REAL NOT NULL,
    total_value REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_snapshots_backtest_id ON portfolio_snapshots(backtest_id);

CREATE TABLE IF NOT EXISTS performance_metrics (
    backtest_id INTEGER PRIMARY KEY REFERENCES backtests(id) ON DELETE CASCADE,
    total_return_pct REAL NOT NULL,
    absolute_pnl REAL NOT NULL,
    num_trades INTEGER NOT NULL,
    win_rate_pct REAL,
    average_win REAL,
    average_loss REAL,
    max_drawdown_pct REAL,
    sharpe_ratio REAL,
    annualized_volatility_pct REAL,
    best_period_return_pct REAL,
    worst_period_return_pct REAL,
    has_open_position INTEGER NOT NULL,
    risk_free_rate_annual REAL NOT NULL
);
"""


# ---------------------------------------------------------------------------
# Connection management + schema initialization
# ---------------------------------------------------------------------------

def get_connection(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    """Open a new SQLite connection with foreign keys enforced.

    Callers are responsible for closing the returned connection (every
    function in this module does so in a `finally` block). A fresh
    connection is intentionally cheap here -- see module docstring for
    why this module never holds one open across calls.
    """
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path | str = DB_PATH) -> None:
    """Create every table/index this module needs, if they don't already
    exist, and seed `schema_version` to 1 if it's empty.

    Safe to call every time the app starts, and safe to call more than
    once in a row -- it never drops or alters an existing table, so it
    can never destroy data that's already there.
    """
    conn = get_connection(db_path)
    try:
        conn.executescript(_SCHEMA_SQL)
        existing_version = conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0]
        if existing_version == 0:
            conn.execute("INSERT INTO schema_version (version) VALUES (1)")
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------

def _to_date_str(value) -> str | None:
    """Normalize a date/datetime/Timestamp/string to an ISO-8601 string.

    All dates are stored as plain ISO text (SQLite has no native date
    type) and kept naive/timezone-free throughout, consistent with how
    the rest of this project already handles dates from pandas/yfinance.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _to_signal_str(value) -> str | None:
    """Normalize a Signal (or None) to its plain string value.

    `Trade.entry_signal`/`exit_signal` are always real `Signal` values
    in practice -- the backtesting engine never produces a `None`
    signal -- but this stays explicit about `None` rather than calling
    `str()` unconditionally, which would otherwise silently turn a
    missing value into the literal text "None" instead of a real SQL
    NULL. Mirrors `_to_date_str`'s same None-handling shape.
    """
    if value is None:
        return None
    return str(value)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_backtest_result(
    result: BacktestResult,
    metrics: PerformanceMetrics,
    ticker: str,
    start_date,
    end_date,
    db_path: Path | str = DB_PATH,
) -> int:
    """Persist one completed backtest (scalar metadata, every completed
    trade, the full equity curve, any open position, and its already-
    computed performance metrics) in a single transaction.

    `result.strategy_name`/`result.strategy_params` are read directly
    from the BacktestResult (the source of truth for what actually ran)
    -- only `ticker`/`start_date`/`end_date` are passed in separately,
    since a BacktestResult has no idea what data it was run on; that's
    the data layer's concern, not the backtesting engine's.

    `metrics` must already be computed (e.g. by
    `calculate_performance_metrics()`) -- this function only stores it,
    it never calls that function itself, so Phase 6's calculation logic
    is never duplicated here.

    Uses ONE transaction for all four tables: if anything fails partway
    through (a bad value, a constraint violation), everything inserted
    so far in this call is rolled back -- there is no way to end up with
    a `backtests` row that has no matching trades/snapshots/metrics, or
    vice versa.

    Returns the new backtest's id.
    """
    conn = get_connection(db_path)
    try:
        # Python's sqlite3 module automatically opens an implicit
        # transaction before the first data-modifying statement (INSERT/
        # UPDATE/DELETE) and keeps every statement after it in that same
        # transaction until an explicit commit() or rollback() -- so the
        # four inserts below are already ONE transaction without needing
        # a manual "BEGIN".
        open_position = result.open_position
        cursor = conn.execute(
            """
            INSERT INTO backtests (
                created_at, ticker, start_date, end_date,
                strategy_name, strategy_params,
                initial_capital, final_value, has_open_position,
                open_position_entry_date, open_position_entry_price,
                open_position_quantity, open_position_entry_signal
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                pd.Timestamp.now("UTC").isoformat(),
                ticker,
                _to_date_str(start_date),
                _to_date_str(end_date),
                result.strategy_name,
                json.dumps(result.strategy_params),
                result.initial_capital,
                result.final_value,
                1 if open_position is not None else 0,
                _to_date_str(open_position.entry_date) if open_position else None,
                open_position.entry_price if open_position else None,
                open_position.quantity if open_position else None,
                _to_signal_str(open_position.entry_signal) if open_position else None,
            ),
        )
        backtest_id = cursor.lastrowid

        if result.trades:
            conn.executemany(
                """
                INSERT INTO trades (
                    backtest_id, entry_date, entry_price, exit_date,
                    exit_price, quantity, entry_signal, exit_signal
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        backtest_id,
                        _to_date_str(t.entry_date),
                        t.entry_price,
                        _to_date_str(t.exit_date),
                        t.exit_price,
                        t.quantity,
                        _to_signal_str(t.entry_signal),
                        _to_signal_str(t.exit_signal),
                    )
                    for t in result.trades
                ],
            )

        equity_curve = result.equity_curve
        if equity_curve is not None and len(equity_curve) > 0:
            conn.executemany(
                """
                INSERT INTO portfolio_snapshots (
                    backtest_id, date, price, signal, cash, shares_held,
                    holdings_value, total_value
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        backtest_id,
                        _to_date_str(row.date),
                        float(row.price),
                        _to_signal_str(row.signal),
                        float(row.cash),
                        int(row.shares_held),
                        float(row.holdings_value),
                        float(row.total_value),
                    )
                    for row in equity_curve.itertuples(index=False)
                ],
            )

        conn.execute(
            """
            INSERT INTO performance_metrics (
                backtest_id, total_return_pct, absolute_pnl, num_trades,
                win_rate_pct, average_win, average_loss, max_drawdown_pct,
                sharpe_ratio, annualized_volatility_pct,
                best_period_return_pct, worst_period_return_pct,
                has_open_position, risk_free_rate_annual
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                backtest_id,
                metrics.total_return_pct,
                metrics.absolute_pnl,
                metrics.num_trades,
                metrics.win_rate_pct,
                metrics.average_win,
                metrics.average_loss,
                metrics.max_drawdown_pct,
                metrics.sharpe_ratio,
                metrics.annualized_volatility_pct,
                metrics.best_period_return_pct,
                metrics.worst_period_return_pct,
                1 if metrics.has_open_position else 0,
                metrics.risk_free_rate_annual,
            ),
        )

        conn.commit()
        return backtest_id
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_backtest_result(backtest_id: int, db_path: Path | str = DB_PATH) -> BacktestResult | None:
    """Reconstruct a full BacktestResult from a saved backtest_id.

    Returns None if the id doesn't exist -- never raises for a missing
    id, so calling code can treat "not found" as an ordinary case.

    The returned BacktestResult is built entirely from Trade,
    OpenPosition, and pandas primitives already defined in
    `src.backtesting.models` -- nothing database-specific (no
    sqlite3.Row, no raw SQL types) leaks into it, so it is
    indistinguishable from a BacktestResult produced by a live
    `run_backtest()` call.
    """
    conn = get_connection(db_path)
    try:
        row = conn.execute("SELECT * FROM backtests WHERE id = ?", (backtest_id,)).fetchone()
        if row is None:
            return None

        trade_rows = conn.execute(
            "SELECT * FROM trades WHERE backtest_id = ? ORDER BY id", (backtest_id,)
        ).fetchall()
        trades = [
            Trade(
                entry_date=pd.Timestamp(r["entry_date"]),
                entry_price=r["entry_price"],
                exit_date=pd.Timestamp(r["exit_date"]),
                exit_price=r["exit_price"],
                quantity=r["quantity"],
                entry_signal=Signal(r["entry_signal"]),
                exit_signal=Signal(r["exit_signal"]),
            )
            for r in trade_rows
        ]

        snapshot_rows = conn.execute(
            "SELECT * FROM portfolio_snapshots WHERE backtest_id = ? ORDER BY id",
            (backtest_id,),
        ).fetchall()
        if snapshot_rows:
            equity_curve = pd.DataFrame(
                {
                    "date": [pd.Timestamp(r["date"]) for r in snapshot_rows],
                    "price": [r["price"] for r in snapshot_rows],
                    "signal": [Signal(r["signal"]) for r in snapshot_rows],
                    "cash": [r["cash"] for r in snapshot_rows],
                    "shares_held": [r["shares_held"] for r in snapshot_rows],
                    "holdings_value": [r["holdings_value"] for r in snapshot_rows],
                    "total_value": [r["total_value"] for r in snapshot_rows],
                }
            )
        else:
            # Same empty-shape convention Phase 5 itself uses -- see
            # EQUITY_CURVE_COLUMNS in src/backtesting/models.py.
            equity_curve = pd.DataFrame(columns=EQUITY_CURVE_COLUMNS)

        open_position = None
        if row["has_open_position"]:
            open_position = OpenPosition(
                entry_date=pd.Timestamp(row["open_position_entry_date"]),
                entry_price=row["open_position_entry_price"],
                quantity=row["open_position_quantity"],
                entry_signal=Signal(row["open_position_entry_signal"]),
            )

        return BacktestResult(
            initial_capital=row["initial_capital"],
            final_value=row["final_value"],
            trades=trades,
            equity_curve=equity_curve,
            open_position=open_position,
            strategy_name=row["strategy_name"],
            strategy_params=json.loads(row["strategy_params"]),
        )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# List (lightweight) and delete
# ---------------------------------------------------------------------------

def list_saved_backtests(db_path: Path | str = DB_PATH) -> list[dict]:
    """Return one lightweight summary dict per saved backtest, newest
    first -- id, when it was saved, ticker, strategy, date range, and
    the STORED total return percentage.

    Deliberately does NOT load trades or portfolio_snapshots -- a
    "Saved Backtests" list showing many rows shouldn't have to pull in
    every bar of every backtest's equity curve just to render a table.
    Uses the metrics stored at save time (the "fast list view" half of
    the approved store/recalculate hybrid) rather than reconstructing
    a full BacktestResult per row.
    """
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            """
            SELECT b.id, b.created_at, b.ticker, b.strategy_name,
                   b.start_date, b.end_date, m.total_return_pct
            FROM backtests b
            LEFT JOIN performance_metrics m ON m.backtest_id = b.id
            ORDER BY b.created_at DESC, b.id DESC
            """
        ).fetchall()
        return [
            {
                "id": r["id"],
                "created_at": r["created_at"],
                "ticker": r["ticker"],
                "strategy_name": r["strategy_name"],
                "start_date": r["start_date"],
                "end_date": r["end_date"],
                "total_return_pct": r["total_return_pct"],
            }
            for r in rows
        ]
    finally:
        conn.close()


def delete_backtest_result(backtest_id: int, db_path: Path | str = DB_PATH) -> bool:
    """Delete one saved backtest and everything that belongs to it.

    Returns True if a row was actually deleted, False if the id didn't
    exist (never raises for a missing id). Child rows in `trades`,
    `portfolio_snapshots`, and `performance_metrics` are removed
    automatically via `ON DELETE CASCADE` -- this function issues exactly
    one DELETE statement and lets the schema's foreign keys handle the
    rest, rather than manually deleting from four tables in application
    code (which would be duplicated logic the schema already expresses).
    """
    conn = get_connection(db_path)
    try:
        cursor = conn.execute("DELETE FROM backtests WHERE id = ?", (backtest_id,))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()
