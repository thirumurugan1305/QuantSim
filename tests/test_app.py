"""
Permanent Streamlit UI regression tests for QuantSim (Phase 9).

WHY THIS FILE EXISTS
---------------------
Every phase since Phase 1 has been verified by actually running the app
-- first via a real headless `streamlit run` boot, and via
`streamlit.testing.v1.AppTest` scripts that clicked through Run Backtest,
Save, Load, and Delete. Those verifications were real and thorough at
the time, but every single one of them was a throwaway script: run once
during development, then discarded. None of them were ever saved as an
actual test file, so none of them run automatically, and none of them
would catch a future change that silently breaks the app's wiring (a
renamed session_state key, a button whose on_click logic stops firing,
a tab that stops rendering).

This file makes that verification permanent and repeatable. It uses the
exact same `AppTest` technique used interactively throughout
development -- nothing new is being introduced, only checked in.

DATABASE ISOLATION
----------------------
`app.py` calls every `src.database.repository` function (`init_db()`,
`save_backtest_result()`, `list_saved_backtests()`, etc.) with NO
explicit `db_path` argument, relying entirely on each function's own
default parameter value (`db_path: Path | str = DB_PATH`). That default
is bound to the real `DB_PATH` once, the first time
`src.database.repository` is imported anywhere in the whole pytest
session -- which may happen well before this file's fixture runs (e.g.
via `test_database.py`). Because of that, simply monkeypatching
`src.config.DB_PATH` (or even `src.database.repository.DB_PATH`) after
that point has NO effect on those already-bound defaults -- a real,
easy-to-miss Python gotcha, confirmed empirically before writing this
fixture the way it's written below.

The correct fix: each repository function is a singleton object living
in `sys.modules['src.database.repository'].__dict__`, shared no matter
how many times `app.py`'s source gets re-executed by `AppTest`. So this
fixture patches each function's `__defaults__` tuple directly, pointing
every one of them at a fresh temporary database BEFORE `AppTest.from_file()`
triggers `app.py`'s module-level `init_db()` call. Nothing here ever
reads or writes the real `quantsim.db`.

NETWORK
----------
This project's test/CI environment has no route to Yahoo Finance
(confirmed throughout development), so `fetch_market_data()` always
falls back to the bundled synthetic sample data here. Every assertion
below is written to hold true regardless of whether the live or
fallback data path is taken, exactly as the app itself already handles
both cases identically from the UI's point of view.
"""

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

# AppTest.from_file() resolves a relative path against the directory of
# the file that CALLS it (i.e. this tests/ directory), not the current
# working directory pytest was invoked from -- so a bare "app.py" would
# incorrectly look for tests/app.py. Resolving explicitly to the real
# app.py one directory up avoids that.
_APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture
def isolated_app(tmp_path, monkeypatch):
    """An AppTest instance wired to a temporary, empty SQLite database
    instead of the real project database.

    Patches `__defaults__` directly on each `src.database.repository`
    function `app.py` calls without an explicit `db_path` -- see the
    module docstring's "DATABASE ISOLATION" section for why patching the
    `DB_PATH` constant itself does not work here.
    """
    from src.database import repository

    db_path = tmp_path / "app_test.db"
    patched_functions = [
        repository.init_db,
        repository.save_backtest_result,
        repository.load_backtest_result,
        repository.list_saved_backtests,
        repository.delete_backtest_result,
    ]
    for func in patched_functions:
        monkeypatch.setattr(func, "__defaults__", (db_path,))

    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)
    return at


def _click(at: AppTest, label: str):
    """Find and click the first button with this exact label, then rerun."""
    matches = [b for b in at.button if b.label == label]
    assert matches, f"no button labeled {label!r} found"
    matches[0].click().run(timeout=30)


class TestAppBootsCleanly:
    def test_no_exception_on_load(self, isolated_app):
        assert isolated_app.exception == []

    def test_all_four_tabs_present(self, isolated_app):
        tab_labels = [t.label for t in isolated_app.tabs]
        assert tab_labels == [
            "📊 Market & Signals",
            "📈 Backtest Results",
            "🎯 Performance",
            "💾 Saved Backtests",
        ]

    def test_saved_backtests_section_renders_even_with_empty_database(self, isolated_app):
        subheaders = [sh.value for sh in isolated_app.subheader]
        assert "Save & Load Backtests" in subheaders


class TestRunBacktestWorkflow:
    def test_run_backtest_produces_results_with_no_exception(self, isolated_app):
        _click(isolated_app, "Run Backtest")
        assert isolated_app.exception == []

        metric_labels = {m.label for m in isolated_app.metric}
        assert "Final Portfolio Value" in metric_labels
        assert "Completed Trades" in metric_labels

    def test_run_backtest_populates_performance_tab(self, isolated_app):
        _click(isolated_app, "Run Backtest")
        metric_labels = {m.label for m in isolated_app.metric}
        assert "Total Return" in metric_labels
        assert "Sharpe Ratio" in metric_labels
        assert "Max Drawdown" in metric_labels


class TestSaveLoadDeleteWorkflow:
    def test_save_backtest_succeeds_and_disables_further_saves(self, isolated_app):
        _click(isolated_app, "Run Backtest")
        _click(isolated_app, "Save Backtest")

        assert isolated_app.exception == []
        assert any("Saved as Backtest #" in s.value for s in isolated_app.success)

        save_btn = [b for b in isolated_app.button if b.label == "Save Backtest"][0]
        assert save_btn.disabled is True

    def test_saved_backtest_appears_in_saved_list(self, isolated_app):
        _click(isolated_app, "Run Backtest")
        _click(isolated_app, "Save Backtest")

        assert len(isolated_app.dataframe) >= 1

    def test_load_selected_reconstructs_result_with_no_exception(self, isolated_app):
        _click(isolated_app, "Run Backtest")
        _click(isolated_app, "Save Backtest")

        load_btns = [b for b in isolated_app.button if b.label == "Load Selected"]
        assert len(load_btns) == 1
        load_btns[0].click().run(timeout=30)

        assert isolated_app.exception == []
        final_value_metrics = [m for m in isolated_app.metric if m.label == "Final Portfolio Value"]
        assert len(final_value_metrics) == 1

    def test_delete_requires_two_clicks_to_confirm(self, isolated_app):
        _click(isolated_app, "Run Backtest")
        _click(isolated_app, "Save Backtest")

        # First click only arms the confirmation -- must not delete yet.
        delete_btns = [b for b in isolated_app.button if b.label == "Delete Selected"]
        delete_btns[0].click().run(timeout=30)
        assert isolated_app.exception == []
        assert any("Delete" in w.value for w in isolated_app.warning)

        confirm_btns = [b for b in isolated_app.button if b.label == "Confirm Delete"]
        assert len(confirm_btns) == 1

    def test_confirmed_delete_removes_the_saved_backtest(self, isolated_app):
        _click(isolated_app, "Run Backtest")
        _click(isolated_app, "Save Backtest")
        _click(isolated_app, "Delete Selected")

        confirm_btns = [b for b in isolated_app.button if b.label == "Confirm Delete"]
        confirm_btns[0].click().run(timeout=30)

        assert isolated_app.exception == []
        load_btns_after_delete = [b for b in isolated_app.button if b.label == "Load Selected"]
        assert len(load_btns_after_delete) == 0  # nothing left to load

    def test_deleting_the_currently_loaded_result_does_not_crash(self, isolated_app):
        _click(isolated_app, "Run Backtest")
        _click(isolated_app, "Save Backtest")
        _click(isolated_app, "Delete Selected")

        confirm_btns = [b for b in isolated_app.button if b.label == "Confirm Delete"]
        confirm_btns[0].click().run(timeout=30)

        assert isolated_app.exception == []
        # The in-memory result (still in session_state) keeps rendering
        # even though its database row is gone -- this is the exact
        # behavior documented in render_backtest_section()'s docstring.
        final_value_metrics = [m for m in isolated_app.metric if m.label == "Final Portfolio Value"]
        assert len(final_value_metrics) == 1
