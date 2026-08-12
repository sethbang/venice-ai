"""Tests for OutputManager rich and plain mode output."""

from __future__ import annotations

# `import venice_ai.cli.utils.console as x` resolves to the Console *object*
# because the module has `console = Console()` at the top level.
# We need the actual module to access the `_plain_mode` flag.
import importlib as _il
import sys as _sys
from unittest.mock import patch

import pytest

from venice_ai.cli.utils.output import OutputManager

_il.import_module("venice_ai.cli.utils.console")
_console_mod = _sys.modules["venice_ai.cli.utils.console"]


def _set_plain_mode(value: bool) -> None:
    """Set the module-level _plain_mode flag."""
    _console_mod._plain_mode = value  # type: ignore[attr-defined]


def _get_plain_mode() -> bool:
    return _console_mod._plain_mode  # type: ignore[return-value]


@pytest.fixture(autouse=True)
def _reset_plain_mode():
    """Ensure plain mode state is reset between tests."""
    original = _get_plain_mode()
    yield
    _set_plain_mode(original)


class TestOutputManagerPlainMode:
    """Test OutputManager methods in plain mode."""

    @pytest.fixture(autouse=True)
    def _enable_plain(self):
        _set_plain_mode(True)

    def test_info(self, capsys):
        OutputManager.info("test message")
        assert "INFO: test message" in capsys.readouterr().out

    def test_success(self, capsys):
        OutputManager.success("done")
        assert "SUCCESS: done" in capsys.readouterr().out

    def test_warning(self, capsys):
        OutputManager.warning("careful")
        assert "WARNING: careful" in capsys.readouterr().out

    def test_error(self, capsys):
        OutputManager.error("bad")
        assert "ERROR: bad" in capsys.readouterr().err

    def test_echo(self, capsys):
        OutputManager.echo("hello")
        assert "hello" in capsys.readouterr().out

    def test_table_with_title(self, capsys):
        OutputManager.table(["Name", "Age"], [["Alice", "30"], ["Bob", "25"]], title="People")
        out = capsys.readouterr().out
        assert "People" in out
        assert "Alice" in out
        assert "Total: 2" in out

    def test_table_without_title(self, capsys):
        OutputManager.table(["A", "B"], [["1", "2"]])
        out = capsys.readouterr().out
        assert "1" in out
        assert "Total: 1" in out

    def test_table_row_padding(self, capsys):
        """Rows shorter than headers should be padded."""
        OutputManager.table(["A", "B", "C"], [["1"]])
        out = capsys.readouterr().out
        assert "1" in out

    def test_panel_with_title(self, capsys):
        OutputManager.panel("content", title="Title")
        out = capsys.readouterr().out
        assert "=== Title ===" in out
        assert "content" in out

    def test_panel_without_title(self, capsys):
        OutputManager.panel("just content")
        out = capsys.readouterr().out
        assert "just content" in out
        assert "===" not in out

    def test_progress_with_pct(self, capsys):
        OutputManager.progress("loading", pct=50.0)
        assert "[50%] loading" in capsys.readouterr().out

    def test_progress_without_pct(self, capsys):
        OutputManager.progress("working")
        out = capsys.readouterr().out
        assert "working" in out
        assert "%" not in out


class TestOutputManagerRichMode:
    """Test OutputManager methods in rich (non-plain) mode."""

    @pytest.fixture(autouse=True)
    def _disable_plain(self):
        _set_plain_mode(False)

    @patch("venice_ai.cli.utils.output.console")
    def test_info(self, mock_console):
        OutputManager.info("test message")
        mock_console.print.assert_called_once()
        call_arg = mock_console.print.call_args[0][0]
        assert "test message" in call_arg

    @patch("venice_ai.cli.utils.output.console")
    def test_success(self, mock_console):
        OutputManager.success("done")
        mock_console.print.assert_called_once()
        call_arg = mock_console.print.call_args[0][0]
        assert "done" in call_arg

    @patch("venice_ai.cli.utils.output.console")
    def test_warning(self, mock_console):
        OutputManager.warning("careful")
        mock_console.print.assert_called_once()
        call_arg = mock_console.print.call_args[0][0]
        assert "careful" in call_arg

    @patch("venice_ai.cli.utils.output.console")
    def test_error(self, mock_console):
        OutputManager.error("bad")
        mock_console.print.assert_called_once()
        call_arg = mock_console.print.call_args[0][0]
        assert "bad" in call_arg

    @patch("venice_ai.cli.utils.output.console")
    def test_echo(self, mock_console):
        OutputManager.echo("hello")
        mock_console.print.assert_called_once_with("hello")

    @patch("venice_ai.cli.utils.output.console")
    def test_table(self, mock_console):
        OutputManager.table(
            ["Name", "Age"],
            [["Alice", "30"]],
            title="People",
            col_styles=["bold", "dim"],
        )
        # Should call console.print twice (table + total)
        assert mock_console.print.call_count == 2

    @patch("venice_ai.cli.utils.output.console")
    def test_table_without_col_styles(self, mock_console):
        OutputManager.table(["A"], [["1"]])
        assert mock_console.print.call_count == 2

    @patch("venice_ai.cli.utils.output.console")
    def test_panel_with_title(self, mock_console):
        OutputManager.panel("content", title="Title", style="green")
        mock_console.print.assert_called_once()

    @patch("venice_ai.cli.utils.output.console")
    def test_panel_without_title(self, mock_console):
        OutputManager.panel("content")
        mock_console.print.assert_called_once()

    @patch("venice_ai.cli.utils.output.console")
    def test_progress_with_pct(self, mock_console):
        OutputManager.progress("loading", pct=75.0)
        mock_console.print.assert_called_once()
        call_arg = mock_console.print.call_args[0][0]
        assert "75%" in call_arg

    @patch("venice_ai.cli.utils.output.console")
    def test_progress_without_pct(self, mock_console):
        OutputManager.progress("working")
        mock_console.print.assert_called_once()
        call_arg = mock_console.print.call_args[0][0]
        assert "working" in call_arg
