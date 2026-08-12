"""
Tests for image command - simplified to focus on testable components
"""

import pytest
from click.testing import CliRunner

from venice_ai.cli.cli import cli


@pytest.fixture
def cli_runner():
    """Fixture providing Click's CliRunner"""
    return CliRunner()


class TestImageCommand:
    """Test image command structure"""

    def test_image_help(self, cli_runner):
        """Test image command help"""
        result = cli_runner.invoke(cli, ["image", "--help"])
        assert result.exit_code == 0
        assert "image" in result.output.lower()

    def test_image_has_subcommands(self, cli_runner):
        """Test image has generate and batch subcommands"""
        result = cli_runner.invoke(cli, ["image", "--help"])
        assert result.exit_code == 0
        assert "generate" in result.output.lower()
        assert "batch" in result.output.lower()

    def test_image_command_registered(self, cli_runner):
        """Test image command is registered in main CLI"""
        result = cli_runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "image" in result.output.lower()


class TestImageGenerate:
    """Test image generate command"""

    def test_generate_help(self, cli_runner):
        """Test generate command help"""
        result = cli_runner.invoke(cli, ["image", "generate", "--help"])
        assert result.exit_code == 0
        assert "generate" in result.output.lower() or "prompt" in result.output.lower()

    def test_generate_requires_prompt(self, cli_runner):
        """Test generate requires prompt argument"""
        result = cli_runner.invoke(cli, ["image", "generate"])
        # Should fail or show error about missing prompt
        assert result.exit_code != 0 or "prompt" in result.output.lower()

    def test_generate_has_model_option(self, cli_runner):
        """Test generate has --model option"""
        result = cli_runner.invoke(cli, ["image", "generate", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.output or "-m" in result.output

    def test_generate_has_size_option(self, cli_runner):
        """Test generate has --size option"""
        result = cli_runner.invoke(cli, ["image", "generate", "--help"])
        assert result.exit_code == 0
        assert "--size" in result.output or "-s" in result.output

    def test_generate_size_choices(self, cli_runner):
        """Test generate size option has valid choices"""
        result = cli_runner.invoke(cli, ["image", "generate", "--help"])
        assert result.exit_code == 0
        # Check for at least some size options
        assert "1024" in result.output or "512" in result.output

    def test_generate_has_num_images_option(self, cli_runner):
        """Test generate has --num-images option"""
        result = cli_runner.invoke(cli, ["image", "generate", "--help"])
        assert result.exit_code == 0
        assert "num" in result.output.lower() and "image" in result.output.lower()


class TestImageBatch:
    """Test image batch command"""

    def test_batch_help(self, cli_runner):
        """Test batch command help"""
        result = cli_runner.invoke(cli, ["image", "batch", "--help"])
        assert result.exit_code == 0
        assert "batch" in result.output.lower() or "prompt" in result.output.lower()

    def test_batch_requires_prompts_file(self, cli_runner):
        """Test batch requires --prompts-file"""
        result = cli_runner.invoke(cli, ["image", "batch"])
        # Should fail or show error about missing prompts file
        assert result.exit_code != 0 or "prompt" in result.output.lower()

    def test_batch_has_prompts_file_option(self, cli_runner):
        """Test batch has --prompts-file option"""
        result = cli_runner.invoke(cli, ["image", "batch", "--help"])
        assert result.exit_code == 0
        assert "prompts-file" in result.output or "prompts" in result.output.lower()

    def test_batch_has_model_option(self, cli_runner):
        """Test batch has --model option"""
        result = cli_runner.invoke(cli, ["image", "batch", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.output or "-m" in result.output

    def test_batch_missing_file_error(self, cli_runner):
        """Test batch handles missing prompts file"""
        result = cli_runner.invoke(
            cli, ["image", "batch", "--prompts-file", "/nonexistent/file.txt"]
        )
        # Should fail due to missing file
        assert result.exit_code != 0


class TestImageOptions:
    """Test image command option validation"""

    def test_generate_invalid_size_rejected(self, cli_runner, monkeypatch):
        """Test generate rejects invalid size"""
        # Resolve a key explicitly so the command reaches size validation.
        # ``cli.config`` calls ``load_dotenv()`` at import, so without this the
        # result depends on whether the working tree has a local ``.env``.
        monkeypatch.setenv("VENICE_API_KEY", "vn_test_key_abc12345")
        result = cli_runner.invoke(
            cli,
            [
                "image",
                "generate",
                "test prompt",
                "--size",
                "999x999",  # Invalid size - not divisible by 8
            ],
        )
        # Should fail or display a validation error message
        output_lower = result.output.lower()
        assert (
            result.exit_code != 0
            or "invalid" in output_lower
            or "validation failed" in output_lower
            or "divisible" in output_lower
        )

    def test_generate_output_option_exists(self, cli_runner):
        """Test generate has --output option"""
        result = cli_runner.invoke(cli, ["image", "generate", "--help"])
        assert result.exit_code == 0
        assert "--output" in result.output or "-o" in result.output

    def test_generate_save_dir_option_exists(self, cli_runner):
        """Test generate has --save-dir option"""
        result = cli_runner.invoke(cli, ["image", "generate", "--help"])
        assert result.exit_code == 0
        assert "save" in result.output.lower() and "dir" in result.output.lower()
