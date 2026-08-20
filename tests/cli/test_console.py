"""
Tests for console utilities
"""

import sys
from io import StringIO
from unittest.mock import patch

import pytest
from rich.console import Console

from venice_ai.cli.utils.console import (
    console,
    enable_plain_mode,
    is_plain_mode,
    print_code,
    print_error,
    print_info,
    print_markdown,
    print_panel,
    print_success,
    print_version_info,
    print_warning,
)


def get_console_module():
    """Get the console module from sys.modules to avoid import shadowing."""
    return sys.modules["venice_ai.cli.utils.console"]


@pytest.fixture
def capture_console():
    """Fixture to capture console output"""
    string_io = StringIO()
    test_console = Console(file=string_io, force_terminal=True, width=80)
    return test_console, string_io


class TestPrintFunctions:
    """Test console print utility functions"""

    @pytest.fixture(autouse=True)
    def reset_plain_mode(self):
        """Ensure plain mode is off for rich-output tests."""
        console_mod = get_console_module()
        original_plain_mode = console_mod._plain_mode
        original_console = console_mod.console
        console_mod._plain_mode = False  # type: ignore
        yield
        console_mod._plain_mode = original_plain_mode  # type: ignore
        console_mod.console = original_console  # type: ignore

    def test_print_error(self, capture_console):
        """Test error message printing"""
        test_console, output = capture_console

        with patch("venice_ai.cli.utils.console.console", test_console):
            print_error("Test error message")

        result = output.getvalue()
        assert "error" in result.lower()
        assert "Test error message" in result

    def test_print_success(self, capture_console):
        """Test success message printing"""
        test_console, output = capture_console

        with patch("venice_ai.cli.utils.console.console", test_console):
            print_success("Operation completed")

        result = output.getvalue()
        assert "Success" in result
        assert "Operation completed" in result

    def test_print_info(self, capture_console):
        """Test info message printing"""
        test_console, output = capture_console

        with patch("venice_ai.cli.utils.console.console", test_console):
            print_info("Information message")

        result = output.getvalue()
        assert "Info" in result
        assert "Information message" in result

    def test_print_warning(self, capture_console):
        """Test warning message printing"""
        test_console, output = capture_console

        with patch("venice_ai.cli.utils.console.console", test_console):
            print_warning("Warning message")

        result = output.getvalue()
        assert "Warning" in result
        assert "Warning message" in result


class TestVersionInfo:
    """Test version information printing"""

    @pytest.fixture(autouse=True)
    def reset_plain_mode(self):
        """Ensure plain mode is off for rich-output tests."""
        console_mod = get_console_module()
        original_plain_mode = console_mod._plain_mode
        original_console = console_mod.console
        console_mod._plain_mode = False  # type: ignore
        yield
        console_mod._plain_mode = original_plain_mode  # type: ignore
        console_mod.console = original_console  # type: ignore

    def test_print_version_info(self, capture_console):
        """Test version info displays correctly"""
        test_console, output = capture_console

        with (
            patch("venice_ai.cli.utils.console.console", test_console),
            patch("venice_ai.cli.__version__", "1.0.0"),
        ):
            print_version_info()

        result = output.getvalue()
        assert "venice-py" in result
        assert "1.0.0" in result


class TestMarkdownAndCode:
    """Test markdown and code printing"""

    def test_print_markdown(self, capture_console):
        """Test markdown rendering"""
        test_console, output = capture_console

        with patch("venice_ai.cli.utils.console.console", test_console):
            print_markdown("# Test Heading\n\nSome **bold** text")

        result = output.getvalue()
        assert "Test Heading" in result

    def test_print_code(self, capture_console):
        """Test code syntax highlighting"""
        test_console, output = capture_console

        with patch("venice_ai.cli.utils.console.console", test_console):
            print_code("def hello():\n    print('world')", language="python")

        result = output.getvalue()
        assert "hello" in result
        assert "print" in result

    def test_print_code_different_language(self, capture_console):
        """Test code printing with different language"""
        test_console, output = capture_console

        with patch("venice_ai.cli.utils.console.console", test_console):
            print_code("const x = 42;", language="javascript")

        result = output.getvalue()
        assert "const" in result or "x = 42" in result


class TestPanel:
    """Test panel printing"""

    def test_print_panel_with_title(self, capture_console):
        """Test panel with title"""
        test_console, output = capture_console

        with patch("venice_ai.cli.utils.console.console", test_console):
            print_panel("Panel content", title="Test Title", style="cyan")

        result = output.getvalue()
        assert "Panel content" in result
        assert "Test Title" in result

    def test_print_panel_without_title(self, capture_console):
        """Test panel without title"""
        test_console, output = capture_console

        with patch("venice_ai.cli.utils.console.console", test_console):
            print_panel("Panel content")

        result = output.getvalue()
        assert "Panel content" in result

    def test_print_panel_custom_style(self, capture_console):
        """Test panel with custom style"""
        test_console, output = capture_console

        with patch("venice_ai.cli.utils.console.console", test_console):
            print_panel("Content", title="Title", style="red")

        result = output.getvalue()
        assert "Content" in result


class TestGlobalConsole:
    """Test global console instance"""

    def test_console_is_console_instance(self):
        """Test global console is a Console instance"""

        assert isinstance(console, Console)

    def test_console_can_print(self):
        """Test global console can print"""

        # Should not raise
        with patch.object(console, "print") as mock_print:
            console.print("test")
            mock_print.assert_called_once_with("test")


class TestPlainMode:
    """Test plain mode functionality"""

    @pytest.fixture(autouse=True)
    def reset_plain_mode(self):
        """Reset plain mode state before and after each test"""
        console_mod = get_console_module()

        # Store original state
        original_plain_mode = console_mod._plain_mode
        original_console = console_mod.console
        yield
        # Restore original state
        console_mod._plain_mode = original_plain_mode  # type: ignore
        console_mod.console = original_console  # type: ignore

    def test_enable_plain_mode(self):
        """Test that enable_plain_mode sets the correct state"""
        console_mod = get_console_module()

        # Ensure we start in non-plain mode
        console_mod._plain_mode = False  # type: ignore

        enable_plain_mode()

        assert console_mod._plain_mode is True  # type: ignore
        # Check that the console is reconfigured
        assert console_mod.console.no_color is True  # type: ignore

    def test_is_plain_mode_returns_false_by_default(self):
        """Test is_plain_mode returns False when not in plain mode"""
        with patch("venice_ai.cli.utils.console._plain_mode", False):
            assert is_plain_mode() is False

    def test_is_plain_mode_returns_true_when_enabled(self):
        """Test is_plain_mode returns True after enabling plain mode"""
        enable_plain_mode()
        assert is_plain_mode() is True


class TestPlainModePrintFunctions:
    """Test print functions in plain mode"""

    @pytest.fixture(autouse=True)
    def setup_plain_mode(self):
        """Set up plain mode and capture console for each test"""
        console_mod = get_console_module()

        # Store original state
        original_plain_mode = console_mod._plain_mode  # type: ignore
        original_console = console_mod.console  # type: ignore

        # Enable plain mode
        enable_plain_mode()

        yield

        # Restore original state
        console_mod._plain_mode = original_plain_mode  # type: ignore
        console_mod.console = original_console  # type: ignore

    @pytest.fixture
    def capture_plain_console(self):
        """Fixture to capture plain console output"""
        string_io = StringIO()
        test_console = Console(file=string_io, force_terminal=False, no_color=True, highlight=False)
        return test_console, string_io

    def test_print_error_plain_mode(self, capture_plain_console):
        """Test error message in plain mode"""
        test_console, output = capture_plain_console

        with patch("venice_ai.cli.utils.console.console", test_console):
            print_error("Test error")

        result = output.getvalue()
        assert "ERROR: Test error" in result

    def test_print_success_plain_mode(self, capture_plain_console):
        """Test success message in plain mode"""
        test_console, output = capture_plain_console

        with patch("venice_ai.cli.utils.console.console", test_console):
            print_success("Operation successful")

        result = output.getvalue()
        assert "SUCCESS: Operation successful" in result

    def test_print_info_plain_mode(self, capture_plain_console):
        """Test info message in plain mode"""
        test_console, output = capture_plain_console

        with patch("venice_ai.cli.utils.console.console", test_console):
            print_info("Info message")

        result = output.getvalue()
        assert "INFO: Info message" in result

    def test_print_warning_plain_mode(self, capture_plain_console):
        """Test warning message in plain mode"""
        test_console, output = capture_plain_console

        with patch("venice_ai.cli.utils.console.console", test_console):
            print_warning("Warning message")

        result = output.getvalue()
        assert "WARNING: Warning message" in result

    def test_print_version_info_plain_mode(self, capture_plain_console):
        """Test version info in plain mode"""
        test_console, output = capture_plain_console

        with (
            patch("venice_ai.cli.utils.console.console", test_console),
            patch("venice_ai.cli.__version__", "2.0.0"),
        ):
            print_version_info()

        result = output.getvalue()
        assert "venice-py v2.0.0" in result

    def test_print_markdown_plain_mode(self, capture_plain_console):
        """Test markdown printing in plain mode (prints raw text)"""
        test_console, output = capture_plain_console

        with patch("venice_ai.cli.utils.console.console", test_console):
            print_markdown("# Heading\nSome text")

        result = output.getvalue()
        assert "# Heading" in result
        assert "Some text" in result

    def test_print_code_plain_mode(self, capture_plain_console):
        """Test code printing in plain mode (no syntax highlighting)"""
        test_console, output = capture_plain_console

        with patch("venice_ai.cli.utils.console.console", test_console):
            print_code("def foo(): pass", language="python")

        result = output.getvalue()
        assert "def foo(): pass" in result

    def test_print_panel_plain_mode_with_title(self, capture_plain_console):
        """Test panel printing in plain mode with title"""
        test_console, output = capture_plain_console

        with patch("venice_ai.cli.utils.console.console", test_console):
            print_panel("Panel content here", title="My Title")

        result = output.getvalue()
        assert "=== My Title ===" in result
        assert "Panel content here" in result

    def test_print_panel_plain_mode_without_title(self, capture_plain_console):
        """Test panel printing in plain mode without title"""
        test_console, output = capture_plain_console

        with patch("venice_ai.cli.utils.console.console", test_console):
            print_panel("Just content")

        result = output.getvalue()
        assert "Just content" in result
        # Should not contain title separator
        assert "===" not in result

    def test_print_panel_plain_mode_empty_title(self, capture_plain_console):
        """Test panel printing in plain mode with empty title string"""
        test_console, output = capture_plain_console

        with patch("venice_ai.cli.utils.console.console", test_console):
            print_panel("Content", title="")

        result = output.getvalue()
        assert "Content" in result
        # Empty title should not print separator
        assert "===" not in result
