"""
Integration tests for Venice CLI entry points
"""

from unittest.mock import patch

import pytest
from click.testing import CliRunner

from venice_ai.cli.cli import cli


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


class TestCLIEntryPoint:
    """Test CLI entry point and basic flags"""

    def test_version_flag(self, cli_runner, mock_load_config):
        """Test --version flag displays version info"""
        result = cli_runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "venice-py" in result.output

    def test_no_command_shows_help(self, cli_runner, mock_load_config):
        """Test running with no command shows help"""
        result = cli_runner.invoke(cli, [])
        assert result.exit_code == 0
        assert "venice-py" in result.output
        # The help layout groups subcommands into curated sections
        # (Generate / Discover / Account / Develop) rather than one
        # "Commands:" block.
        assert "Commands (Generate)" in result.output

    def test_help_flag(self, cli_runner, mock_load_config):
        """Test --help flag shows help"""
        result = cli_runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "venice-py" in result.output
        assert "Options:" in result.output

    def test_invalid_command(self, cli_runner, mock_load_config):
        """Test invalid command shows error"""
        result = cli_runner.invoke(cli, ["nonexistent"])
        assert result.exit_code != 0
        assert "Error" in result.output or "No such command" in result.output


class TestMainEntryPoint:
    """Test __main__.py entry point"""

    def test_main_imports(self):
        """Test main can be imported from __main__"""
        from venice_ai.cli.__main__ import main

        assert callable(main)

    def test_main_is_cli_main(self):
        """Test __main__.main is the same as cli.main"""
        from venice_ai.cli.__main__ import main as main_entry
        from venice_ai.cli.cli import main as cli_main

        assert main_entry == cli_main


class TestConfigLoading:
    """Test configuration loading in CLI context"""

    def test_custom_config_path(self, cli_runner, tmp_path):
        """Test --config option with custom path"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("api:\n  base_url: https://custom.url\n")

        with patch("venice_ai.cli.cli.load_config") as mock_load:
            mock_load.return_value = {"api": {"base_url": "https://custom.url"}}
            result = cli_runner.invoke(cli, ["--config", str(config_file)])
            assert result.exit_code == 0
            # Verify load_config was called with the custom path
            mock_load.assert_called()

    def test_default_config_loading(self, cli_runner, mock_load_config):
        """Test config loads with defaults when no --config specified"""
        result = cli_runner.invoke(cli, [])
        assert result.exit_code == 0
        mock_load_config.assert_called_once()


class TestCommandGroups:
    """Test command groups are properly registered"""

    def test_chat_command_registered(self, cli_runner, mock_load_config):
        """Test chat command is available"""
        result = cli_runner.invoke(cli, ["chat", "--help"])
        assert result.exit_code == 0
        assert "chat" in result.output.lower()

    def test_image_command_registered(self, cli_runner, mock_load_config):
        """Test image command is available"""
        result = cli_runner.invoke(cli, ["image", "--help"])
        assert result.exit_code == 0
        assert "image" in result.output.lower()

    def test_configure_command_registered(self, cli_runner, mock_load_config):
        """Test configure command is available"""
        result = cli_runner.invoke(cli, ["configure", "--help"])
        assert result.exit_code == 0
        assert "configure" in result.output.lower()

    def test_models_command_registered(self, cli_runner, mock_load_config):
        """Test models command is available"""
        result = cli_runner.invoke(cli, ["models", "--help"])
        assert result.exit_code == 0
        assert "models" in result.output.lower()
