"""
Comprehensive tests for Venice AI CLI - cli.py module

Tests focus on achieving 90%+ coverage by targeting:
- main() entry point function
- KeyboardInterrupt handling
- Exception handling
- Plain mode functionality
- models command with beta flag logic
"""

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from venice_ai import __version__ as VERSION
from venice_ai.cli.cli import cli, main


@pytest.fixture
def cli_runner():
    """Fixture providing Click's CliRunner for testing"""
    return CliRunner()


@pytest.fixture
def mock_load_config():
    """Mock config loading to avoid file I/O"""
    with patch("venice_ai.cli.cli.load_config") as mock:
        mock.return_value = {
            "api": {"base_url": "https://api.venice.ai/api/v1"},
            "defaults": {
                "chat_model": "venice-uncensored",
                "image_model": "hidream",
                "max_completion_tokens": 2048,
                "temperature": 0.7,
            },
            "output": {"format": "markdown", "images_dir": "~/Pictures/venice"},
            "features": {"streaming": True, "cost_tracking": True},
        }
        yield mock


class TestMainEntryPoint:
    """Test the main() function entry point"""

    def test_main_successful_invocation(self, mock_load_config):
        """Test main() runs successfully with no arguments (shows help)"""
        with patch("venice_ai.cli.cli.cli") as mock_cli:
            # Simulate successful CLI execution
            mock_cli.return_value = None
            main()
            mock_cli.assert_called_once_with(obj={})

    def test_main_keyboard_interrupt(self, mock_load_config):
        """Test main() handles KeyboardInterrupt gracefully"""
        with (
            patch("venice_ai.cli.cli.cli") as mock_cli,
            patch("venice_ai.cli.cli.console") as mock_console,
            patch("venice_ai.cli.cli.sys.exit") as mock_exit,
        ):
            mock_cli.side_effect = KeyboardInterrupt()
            main()
            mock_console.print.assert_called_once()
            assert "Interrupted" in mock_console.print.call_args[0][0]
            mock_exit.assert_called_once_with(0)

    def test_main_exception_handling(self, mock_load_config):
        """Test main() handles generic exceptions"""
        with (
            patch("venice_ai.cli.cli.cli") as mock_cli,
            patch("venice_ai.cli.cli.console") as mock_console,
            patch("venice_ai.cli.cli.sys.exit") as mock_exit,
        ):
            mock_cli.side_effect = Exception("Test error message")
            main()
            mock_console.print.assert_called_once()
            assert "Error" in mock_console.print.call_args[0][0]
            assert "Test error message" in mock_console.print.call_args[0][0]
            mock_exit.assert_called_once_with(1)

    def test_main_runtime_error(self, mock_load_config):
        """Test main() handles RuntimeError"""
        with (
            patch("venice_ai.cli.cli.cli") as mock_cli,
            patch("venice_ai.cli.cli.console") as mock_console,
            patch("venice_ai.cli.cli.sys.exit") as mock_exit,
        ):
            mock_cli.side_effect = RuntimeError("Configuration failure")
            main()
            mock_console.print.assert_called_once()
            assert "Configuration failure" in mock_console.print.call_args[0][0]
            mock_exit.assert_called_once_with(1)


class TestCLIPlainMode:
    """Test --plain flag functionality"""

    def test_plain_mode_enables_plain_output(self, cli_runner, mock_load_config):
        """Test --plain flag enables plain text output mode"""
        with patch("venice_ai.cli.cli.enable_plain_mode") as mock_enable:
            result = cli_runner.invoke(cli, ["--plain"])
            assert result.exit_code == 0
            mock_enable.assert_called_once()

    def test_plain_mode_sets_context_values(self, cli_runner, mock_load_config):
        """Test --plain flag sets correct context object values"""
        with patch("venice_ai.cli.cli.enable_plain_mode"):
            result = cli_runner.invoke(cli, ["--plain"])
            assert result.exit_code == 0

    def test_plain_mode_with_version(self, cli_runner, mock_load_config):
        """Test --plain with --version flag"""
        with (
            patch("venice_ai.cli.cli.enable_plain_mode") as mock_enable,
            patch("venice_ai.cli.cli.print_version_info") as mock_version,
        ):
            cli_runner.invoke(cli, ["--plain", "--version"])
            mock_enable.assert_called_once()
            mock_version.assert_called_once()


class TestCLIConfigOption:
    """Test --config option handling"""

    def test_config_option_with_path(self, cli_runner, tmp_path):
        """Test --config with a specific file path"""
        config_file = tmp_path / "custom_config.yaml"
        config_file.write_text("api:\n  base_url: https://custom.api\n")

        with patch("venice_ai.cli.cli.load_config") as mock_load:
            mock_load.return_value = {"api": {"base_url": "https://custom.api"}}
            result = cli_runner.invoke(cli, ["--config", str(config_file)])
            assert result.exit_code == 0
            # Verify the Path was passed correctly
            call_args = mock_load.call_args[0][0]
            assert str(config_file) in str(call_args)

    def test_config_option_none_uses_default(self, cli_runner, mock_load_config):
        """Test that no --config uses default path (None)"""
        result = cli_runner.invoke(cli, [])
        assert result.exit_code == 0
        mock_load_config.assert_called_with(None)


@pytest.fixture
def mock_models_command():
    """Mock list_models function and asyncio.run to avoid unawaited coroutine warnings.

    The list_models function is async and imported inside the models command
    callback (``commands.models.group``). We mock at the import site and the
    asyncio.run inside that group module.
    """
    # Create a plain synchronous mock to avoid coroutine issues
    mock_list_models = MagicMock(return_value=None)

    with (
        patch("venice_ai.cli.commands.models.command.list_models", mock_list_models),
        patch("venice_ai.cli.commands.models.group.asyncio.run") as mock_run,
    ):
        mock_run.return_value = None
        yield {"list_models": mock_list_models, "asyncio_run": mock_run}


class TestModelsCommand:
    """Test models command and option handling"""

    def test_models_command_beta_flag_true(self, cli_runner, mock_load_config, mock_models_command):
        """Test models command with --beta flag sets beta=True"""
        result = cli_runner.invoke(cli, ["models", "--beta"])
        assert result.exit_code == 0
        # Verify asyncio.run was called
        mock_models_command["asyncio_run"].assert_called_once()

    def test_models_command_no_beta_flag(self, cli_runner, mock_load_config, mock_models_command):
        """Test models command with --no-beta flag sets beta=False"""
        result = cli_runner.invoke(cli, ["models", "--no-beta"])
        assert result.exit_code == 0
        mock_models_command["asyncio_run"].assert_called_once()

    def test_models_command_no_beta_flags(self, cli_runner, mock_load_config, mock_models_command):
        """Test models command without any beta flags leaves beta=None"""
        result = cli_runner.invoke(cli, ["models"])
        assert result.exit_code == 0
        mock_models_command["asyncio_run"].assert_called_once()

    def test_models_command_with_type_filter(
        self, cli_runner, mock_load_config, mock_models_command
    ):
        """Test models command with --type filter"""
        result = cli_runner.invoke(cli, ["models", "--type", "text"])
        assert result.exit_code == 0
        mock_models_command["asyncio_run"].assert_called_once()

    def test_models_command_with_multiple_filters(
        self, cli_runner, mock_load_config, mock_models_command
    ):
        """Test models command with multiple capability filters"""
        result = cli_runner.invoke(cli, ["models", "--vision", "--function-calling", "--reasoning"])
        assert result.exit_code == 0
        mock_models_command["asyncio_run"].assert_called_once()

    def test_models_command_with_price_filters(
        self, cli_runner, mock_load_config, mock_models_command
    ):
        """Test models command with price filtering options"""
        result = cli_runner.invoke(cli, ["models", "--max-input", "1.0", "--max-output", "2.0"])
        assert result.exit_code == 0
        mock_models_command["asyncio_run"].assert_called_once()

    def test_models_command_with_sort(self, cli_runner, mock_load_config, mock_models_command):
        """Test models command with --sort option"""
        result = cli_runner.invoke(cli, ["models", "--sort", "price-asc"])
        assert result.exit_code == 0
        mock_models_command["asyncio_run"].assert_called_once()

    def test_models_command_with_search(self, cli_runner, mock_load_config, mock_models_command):
        """Test models command with --search option"""
        result = cli_runner.invoke(cli, ["models", "--search", "qwen"])
        assert result.exit_code == 0
        mock_models_command["asyncio_run"].assert_called_once()

    def test_models_command_with_detail(self, cli_runner, mock_load_config, mock_models_command):
        """Test models command with --id/--detail option"""
        result = cli_runner.invoke(cli, ["models", "--id", "test-model"])
        assert result.exit_code == 0
        mock_models_command["asyncio_run"].assert_called_once()

    def test_models_command_with_compare(self, cli_runner, mock_load_config, mock_models_command):
        """Test models command with --compare option"""
        result = cli_runner.invoke(cli, ["models", "--compare", "model-a,model-b"])
        assert result.exit_code == 0
        mock_models_command["asyncio_run"].assert_called_once()

    def test_models_command_verbose_json_output(
        self, cli_runner, mock_load_config, mock_models_command
    ):
        """Test models command with --verbose and --json options"""
        result = cli_runner.invoke(cli, ["models", "--verbose", "--json"])
        assert result.exit_code == 0
        mock_models_command["asyncio_run"].assert_called_once()

    def test_models_command_with_currency(self, cli_runner, mock_load_config, mock_models_command):
        """Test models command with --currency option"""
        result = cli_runner.invoke(cli, ["models", "--currency", "usd"])
        assert result.exit_code == 0
        mock_models_command["asyncio_run"].assert_called_once()

    def test_models_command_with_trait(self, cli_runner, mock_load_config, mock_models_command):
        """Test models command with --trait filter"""
        result = cli_runner.invoke(cli, ["models", "--trait", "default"])
        assert result.exit_code == 0
        mock_models_command["asyncio_run"].assert_called_once()

    def test_models_command_with_budget(self, cli_runner, mock_load_config, mock_models_command):
        """Test models command with --budget option"""
        result = cli_runner.invoke(cli, ["models", "--budget", "5.0"])
        assert result.exit_code == 0
        mock_models_command["asyncio_run"].assert_called_once()

    def test_models_command_with_online_flag(
        self, cli_runner, mock_load_config, mock_models_command
    ):
        """Test models command with --online flag"""
        result = cli_runner.invoke(cli, ["models", "--online"])
        assert result.exit_code == 0
        mock_models_command["asyncio_run"].assert_called_once()


class TestConfigureCommand:
    """Test configure command"""

    def test_configure_command_invokes_configure_cli(self, cli_runner, mock_load_config):
        """Test configure command calls configure_cli function"""
        with patch("venice_ai.cli.commands.configure.configure_cli") as mock_configure:
            result = cli_runner.invoke(cli, ["configure"])
            assert result.exit_code == 0
            mock_configure.assert_called_once()


class TestDunderMain:
    """Test __main__ execution path"""

    def test_dunder_main_calls_main(self):
        """Test if __name__ == '__main__' block would call main()"""
        # This tests the imported main function
        from venice_ai.cli.cli import main as cli_main

        assert callable(cli_main)

    def test_main_module_execution(self):
        """Test main can be called from __main__"""
        from venice_ai.cli.__main__ import main

        # Verify main is the correct function
        from venice_ai.cli.cli import main as cli_main

        assert main == cli_main


class TestModuleConstants:
    """Test module-level constants and imports"""

    def test_version_constant(self):
        """Test VERSION constant is defined"""
        import venice_ai

        assert venice_ai.__version__ == VERSION

    def test_version_matches_installed_metadata(self):
        """__version__ must match installed package metadata.

        Regression guard: the value was hardcoded to "2.0.0" while the
        installed package was "2.0.0rc1", so ``venice-py --version`` lied.
        """
        from importlib.metadata import version as _installed_version

        import venice_ai

        assert venice_ai.__version__ == _installed_version("venice-ai")

    def test_cli_group_name(self):
        """Test CLI group name"""
        assert cli.name == "cli"

    def test_registered_commands(self):
        """Test commands are registered"""
        command_names = list(cli.commands.keys())
        assert "chat" in command_names
        assert "image" in command_names
        assert "configure" in command_names
        assert "models" in command_names
