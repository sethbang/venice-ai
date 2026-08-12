"""
Tests for configure command - comprehensive coverage for cli/commands/configure.py
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from venice_ai.cli.cli import cli
from venice_ai.cli.commands.configure import (
    _fetch_models_by_type,
    _fetch_models_sync,
    configure_cli,
)


@pytest.fixture
def cli_runner():
    """Fixture providing Click's CliRunner"""
    return CliRunner()


@pytest.fixture
def mock_config():
    """Fixture providing default config"""
    return {
        "api": {"key": "test-api-key-12345678"},
        "defaults": {
            "chat_model": "llama-3.3-70b",
            "image_model": "flux-2-pro",
            "temperature": 0.7,
            "max_completion_tokens": 2048,
        },
        "output": {"images_dir": "/tmp/venice"},
        "features": {"streaming": True},
    }


@pytest.fixture
def mock_questionary():
    """Fixture providing a mock questionary module"""
    mock = MagicMock()
    return mock


class TestConfigureCommand:
    """Test configure command"""

    def test_configure_help(self, cli_runner):
        """Test configure command help text"""
        result = cli_runner.invoke(cli, ["configure", "--help"])
        assert result.exit_code == 0
        assert "configure" in result.output.lower()

    def test_configure_command_registered(self, cli_runner):
        """Test configure command is registered in main CLI"""
        result = cli_runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "configure" in result.output.lower()


class TestConfigureAsync:
    """Test configure async function"""

    def test_configure_cli_function_exists(self):
        """Test configure_cli function exists and is callable"""
        import inspect

        assert callable(configure_cli)
        # Note: configure_cli is synchronous as it uses synchronous questionary calls
        assert inspect.isfunction(configure_cli)


class TestFetchModelsByType:
    """Test async model fetching function"""

    @pytest.mark.asyncio
    async def test_fetch_models_by_type_success(self):
        """Test successful model fetching"""
        mock_model1 = SimpleNamespace(id="model-1")
        mock_model2 = SimpleNamespace(id="model-2")
        mock_response = SimpleNamespace(data=[mock_model1, mock_model2])

        mock_client = AsyncMock()
        mock_client.models.list = AsyncMock(return_value=mock_response)

        with patch("venice_ai.cli.commands.configure.VeniceClient") as mock_venice_client:
            mock_venice_client.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_venice_client.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await _fetch_models_by_type("test-key", "text")
            assert result == ["model-1", "model-2"]

    @pytest.mark.asyncio
    async def test_fetch_models_by_type_empty_response(self):
        """Test model fetching with empty response"""
        mock_response = SimpleNamespace(data=None)

        mock_client = AsyncMock()
        mock_client.models.list = AsyncMock(return_value=mock_response)

        with patch("venice_ai.cli.commands.configure.VeniceClient") as mock_venice_client:
            mock_venice_client.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_venice_client.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await _fetch_models_by_type("test-key", "text")
            assert result == []

    @pytest.mark.asyncio
    async def test_fetch_models_by_type_empty_data_list(self):
        """Test model fetching with empty data list"""
        mock_response = SimpleNamespace(data=[])

        mock_client = AsyncMock()
        mock_client.models.list = AsyncMock(return_value=mock_response)

        with patch("venice_ai.cli.commands.configure.VeniceClient") as mock_venice_client:
            mock_venice_client.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_venice_client.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await _fetch_models_by_type("test-key", "text")
            assert result == []

    @pytest.mark.asyncio
    async def test_fetch_models_by_type_exception(self):
        """Test model fetching handles exceptions"""
        with patch("venice_ai.cli.commands.configure.VeniceClient") as mock_venice_client:
            mock_venice_client.return_value.__aenter__ = AsyncMock(
                side_effect=Exception("API Error")
            )
            mock_venice_client.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await _fetch_models_by_type("test-key", "text")
            assert result == []


class TestFetchModelsSync:
    """Test synchronous model fetching wrapper"""

    def test_fetch_models_sync_success(self):
        """Test sync wrapper calls async function correctly"""

        # Mock asyncio.run to return the expected result directly
        # We need to close the coroutine that was passed to avoid warnings
        def run_mock(coro):
            # Close the coroutine to prevent "unawaited" warnings
            coro.close()
            return ["model-1", "model-2"]

        with patch("venice_ai.cli.commands.configure.asyncio.run", side_effect=run_mock):
            result = _fetch_models_sync("test-key", "text")
            assert result == ["model-1", "model-2"]


class TestConfigureConfigPath:
    """configure honors --config for both reading and writing."""

    def test_configure_reads_and_writes_given_config_path(self, cli_runner, tmp_path, monkeypatch):
        import yaml

        from venice_ai.cli import config as config_mod

        monkeypatch.delenv("VENICE_API_KEY", raising=False)
        # Reset active path so other tests / state don't leak in.
        config_mod.set_active_config_path(None)

        custom_path = tmp_path / "myconf" / "config.yaml"
        # Seed an existing config at the custom path so we can verify it is READ.
        custom_path.parent.mkdir(parents=True, exist_ok=True)
        custom_path.write_text("api:\n  key: seeded-existing-key-1234\n")

        # confirm order: update key? No -> models? No -> gen? No -> output? No
        #                -> streaming? Yes -> save? Yes -> save-to-path? Yes
        confirm_responses = iter([False, False, False, False, True, True, True])

        def confirm_mock(*args, **kwargs):
            m = MagicMock()
            m.ask = MagicMock(return_value=next(confirm_responses, False))
            return m

        try:
            with patch("venice_ai.cli.commands.configure.questionary") as mock_q:
                mock_q.confirm = MagicMock(side_effect=confirm_mock)
                # No password prompt expected (existing key, user declines update).
                mock_q.password = MagicMock(
                    side_effect=lambda *a, **k: MagicMock(ask=MagicMock(return_value=None))
                )
                with patch("venice_ai.cli.commands.configure.console"):
                    result = cli_runner.invoke(cli, ["--config", str(custom_path), "configure"])
        finally:
            config_mod.set_active_config_path(None)

        assert result.exit_code == 0, result.output
        # Written back to the SAME custom path, and the seeded key was preserved.
        assert custom_path.exists()
        written = yaml.safe_load(custom_path.read_text())
        assert written["api"]["key"] == "seeded-existing-key-1234"
        assert written["features"]["streaming"] is True
        # No stray default config file was created.
        assert not (tmp_path / "config.yaml").exists()


class TestConfigureCli:
    """Test main configure_cli function with various scenarios"""

    def _create_questionary_mock(self, responses):
        """Helper to create questionary mock with multiple responses"""
        mock = MagicMock()

        def create_prompt_mock(response):
            prompt_mock = MagicMock()
            prompt_mock.ask = MagicMock(return_value=response)
            return prompt_mock

        # Track call counts for each prompt type
        confirm_calls = []
        password_calls = []
        select_calls = []
        text_calls = []
        path_calls = []

        # Extract responses for each type
        for key, value in responses.items():
            if key.startswith("confirm"):
                confirm_calls.append(value)
            elif key.startswith("password"):
                password_calls.append(value)
            elif key.startswith("select"):
                select_calls.append(value)
            elif key.startswith("text"):
                text_calls.append(value)
            elif key.startswith("path"):
                path_calls.append(value)

        # Create side_effect generators
        confirm_iter = iter(confirm_calls) if confirm_calls else iter([False])
        password_iter = iter(password_calls) if password_calls else iter([None])
        select_iter = iter(select_calls) if select_calls else iter([None])
        text_iter = iter(text_calls) if text_calls else iter([None])
        path_iter = iter(path_calls) if path_calls else iter([None])

        def confirm_side_effect(*args, **kwargs):
            prompt_mock = MagicMock()
            try:
                value = next(confirm_iter)
            except StopIteration:
                value = False
            prompt_mock.ask = MagicMock(return_value=value)
            return prompt_mock

        def password_side_effect(*args, **kwargs):
            prompt_mock = MagicMock()
            try:
                value = next(password_iter)
            except StopIteration:
                value = None
            prompt_mock.ask = MagicMock(return_value=value)
            return prompt_mock

        def select_side_effect(*args, **kwargs):
            prompt_mock = MagicMock()
            try:
                value = next(select_iter)
            except StopIteration:
                value = None
            prompt_mock.ask = MagicMock(return_value=value)
            return prompt_mock

        def text_side_effect(*args, **kwargs):
            prompt_mock = MagicMock()
            try:
                value = next(text_iter)
            except StopIteration:
                value = None
            prompt_mock.ask = MagicMock(return_value=value)
            return prompt_mock

        def path_side_effect(*args, **kwargs):
            prompt_mock = MagicMock()
            try:
                value = next(path_iter)
            except StopIteration:
                value = None
            prompt_mock.ask = MagicMock(return_value=value)
            return prompt_mock

        mock.confirm = MagicMock(side_effect=confirm_side_effect)
        mock.password = MagicMock(side_effect=password_side_effect)
        mock.select = MagicMock(side_effect=select_side_effect)
        mock.text = MagicMock(side_effect=text_side_effect)
        mock.path = MagicMock(side_effect=path_side_effect)

        return mock

    def test_configure_cli_no_existing_key_new_key_provided(self):
        """Test configuration with no existing key and user provides new key"""
        # All confirms in order:
        # 1. "Do you want to update the API key?" - not asked since no key
        # 2. "Configure default models?" - True
        # 3. "Configure generation parameters?" - False
        # 4. "Configure output settings?" - False
        # 5. "Enable response streaming?" - True
        # 6. "Save configuration?" - False (to avoid file writes)

        with patch("venice_ai.cli.commands.configure.load_config") as mock_load:
            mock_load.return_value = {}

            with patch("venice_ai.cli.commands.configure.os.getenv") as mock_getenv:
                mock_getenv.return_value = None

                with patch("venice_ai.cli.commands.configure._fetch_models_sync") as mock_fetch:
                    # configure now pins all six model types; return a non-empty
                    # list for every /models type so each prompt is offered.
                    mock_fetch.return_value = [
                        "llama-3.3-70b",
                        "qwen3-235b",
                        "flux-2-pro",
                        "flux-2-max",
                        "venice-sd35",
                    ]

                    with patch("venice_ai.cli.commands.configure.questionary") as mock_q:
                        # Setup confirm responses
                        confirm_responses = iter(
                            [True, False, False, True, False]
                        )  # models, gen params, output, streaming, save
                        password_responses = iter(["new-test-api-key-12345"])
                        select_responses = iter(["llama-3.3-70b", "flux-2-pro"])

                        def confirm_mock(*args, **kwargs):
                            mock_prompt = MagicMock()
                            mock_prompt.ask = MagicMock(return_value=next(confirm_responses, False))
                            return mock_prompt

                        def password_mock(*args, **kwargs):
                            mock_prompt = MagicMock()
                            mock_prompt.ask = MagicMock(return_value=next(password_responses, None))
                            return mock_prompt

                        def select_mock(*args, **kwargs):
                            mock_prompt = MagicMock()
                            mock_prompt.ask = MagicMock(return_value=next(select_responses, None))
                            return mock_prompt

                        mock_q.confirm = MagicMock(side_effect=confirm_mock)
                        mock_q.password = MagicMock(side_effect=password_mock)
                        mock_q.select = MagicMock(side_effect=select_mock)

                        with patch("venice_ai.cli.commands.configure.console"):
                            configure_cli({})

    def test_configure_cli_existing_key_no_update(self):
        """Test configuration with existing key and user declines update"""
        with patch("venice_ai.cli.commands.configure.load_config") as mock_load:
            mock_load.return_value = {"api": {"key": "existing-api-key-12345678"}}

            with patch("venice_ai.cli.commands.configure.os.getenv") as mock_getenv:
                mock_getenv.return_value = None

                with patch("venice_ai.cli.commands.configure.questionary") as mock_q:
                    # Responses: update key? No, configure models? No, streaming? Yes, save? No
                    confirm_responses = iter([False, False, False, False, True, False])

                    def confirm_mock(*args, **kwargs):
                        mock_prompt = MagicMock()
                        mock_prompt.ask = MagicMock(return_value=next(confirm_responses, False))
                        return mock_prompt

                    mock_q.confirm = MagicMock(side_effect=confirm_mock)

                    with patch("venice_ai.cli.commands.configure.console"):
                        configure_cli({})

    def test_configure_cli_existing_key_from_env(self):
        """Test configuration with existing key from environment variable"""
        with patch("venice_ai.cli.commands.configure.load_config") as mock_load:
            mock_load.return_value = {}

            with patch("venice_ai.cli.commands.configure.os.getenv") as mock_getenv:
                mock_getenv.return_value = "env-api-key-12345678901234"

                with patch("venice_ai.cli.commands.configure.questionary") as mock_q:
                    # Responses: update key? No, configure models? No, streaming? Yes, save? No
                    confirm_responses = iter([False, False, False, False, True, False])

                    def confirm_mock(*args, **kwargs):
                        mock_prompt = MagicMock()
                        mock_prompt.ask = MagicMock(return_value=next(confirm_responses, False))
                        return mock_prompt

                    mock_q.confirm = MagicMock(side_effect=confirm_mock)

                    with patch("venice_ai.cli.commands.configure.console"):
                        configure_cli({})

    def test_configure_cli_existing_key_update(self):
        """Test configuration with existing key and user updates it"""
        with patch("venice_ai.cli.commands.configure.load_config") as mock_load:
            mock_load.return_value = {"api": {"key": "old-api-key-12345678"}}

            with patch("venice_ai.cli.commands.configure.os.getenv") as mock_getenv:
                mock_getenv.return_value = None

                with patch("venice_ai.cli.commands.configure.questionary") as mock_q:
                    # Responses: update key? Yes, configure models? No, streaming? Yes, save? No
                    confirm_responses = iter([True, False, False, False, True, False])
                    password_responses = iter(["new-updated-key"])

                    def confirm_mock(*args, **kwargs):
                        mock_prompt = MagicMock()
                        mock_prompt.ask = MagicMock(return_value=next(confirm_responses, False))
                        return mock_prompt

                    def password_mock(*args, **kwargs):
                        mock_prompt = MagicMock()
                        mock_prompt.ask = MagicMock(return_value=next(password_responses, None))
                        return mock_prompt

                    mock_q.confirm = MagicMock(side_effect=confirm_mock)
                    mock_q.password = MagicMock(side_effect=password_mock)

                    with patch("venice_ai.cli.commands.configure.console"):
                        configure_cli({})

    def test_configure_cli_short_existing_key_masking(self):
        """Test API key masking for short keys (< 12 chars)"""
        with patch("venice_ai.cli.commands.configure.load_config") as mock_load:
            # Short key - will show as "***"
            mock_load.return_value = {"api": {"key": "short"}}

            with patch("venice_ai.cli.commands.configure.os.getenv") as mock_getenv:
                mock_getenv.return_value = None

                with patch("venice_ai.cli.commands.configure.questionary") as mock_q:
                    confirm_responses = iter([False, False, False, False, True, False])

                    def confirm_mock(*args, **kwargs):
                        mock_prompt = MagicMock()
                        mock_prompt.ask = MagicMock(return_value=next(confirm_responses, False))
                        return mock_prompt

                    mock_q.confirm = MagicMock(side_effect=confirm_mock)

                    with patch("venice_ai.cli.commands.configure.console"):
                        configure_cli({})

    def test_configure_cli_configure_models_no_api_key(self):
        """Test configuring models without an API key uses defaults"""
        with patch("venice_ai.cli.commands.configure.load_config") as mock_load:
            mock_load.return_value = {}

            with patch("venice_ai.cli.commands.configure.os.getenv") as mock_getenv:
                mock_getenv.return_value = None

                with patch("venice_ai.cli.commands.configure.questionary") as mock_q:
                    # No password provided, configure models, streaming, no save
                    confirm_responses = iter([True, False, False, True, False])
                    password_responses = iter([None])  # No API key provided
                    select_responses = iter(["llama-3.3-70b", "venice-sd35"])

                    def confirm_mock(*args, **kwargs):
                        mock_prompt = MagicMock()
                        mock_prompt.ask = MagicMock(return_value=next(confirm_responses, False))
                        return mock_prompt

                    def password_mock(*args, **kwargs):
                        mock_prompt = MagicMock()
                        mock_prompt.ask = MagicMock(return_value=next(password_responses, None))
                        return mock_prompt

                    def select_mock(*args, **kwargs):
                        mock_prompt = MagicMock()
                        mock_prompt.ask = MagicMock(return_value=next(select_responses, None))
                        return mock_prompt

                    mock_q.confirm = MagicMock(side_effect=confirm_mock)
                    mock_q.password = MagicMock(side_effect=password_mock)
                    mock_q.select = MagicMock(side_effect=select_mock)

                    with (
                        patch("venice_ai.cli.commands.configure.console"),
                        patch("venice_ai.cli.commands.configure.print_error"),
                    ):
                        configure_cli({})

    def test_configure_cli_configure_models_fetch_fails(self):
        """Test configuring models when API fetch fails"""
        with patch("venice_ai.cli.commands.configure.load_config") as mock_load:
            mock_load.return_value = {"api": {"key": "test-key-123456789"}}

            with patch("venice_ai.cli.commands.configure.os.getenv") as mock_getenv:
                mock_getenv.return_value = None

                with patch("venice_ai.cli.commands.configure._fetch_models_sync") as mock_fetch:
                    # Return empty lists for every type to trigger graceful skip
                    mock_fetch.return_value = []

                    with patch("venice_ai.cli.commands.configure.questionary") as mock_q:
                        # No update key, configure models, no gen params, no output, streaming, no save
                        confirm_responses = iter([False, True, False, False, True, False])
                        select_responses = iter(["llama-3.3-70b", "venice-sd35"])

                        def confirm_mock(*args, **kwargs):
                            mock_prompt = MagicMock()
                            mock_prompt.ask = MagicMock(return_value=next(confirm_responses, False))
                            return mock_prompt

                        def select_mock(*args, **kwargs):
                            mock_prompt = MagicMock()
                            mock_prompt.ask = MagicMock(return_value=next(select_responses, None))
                            return mock_prompt

                        mock_q.confirm = MagicMock(side_effect=confirm_mock)
                        mock_q.select = MagicMock(side_effect=select_mock)

                        with (
                            patch("venice_ai.cli.commands.configure.console"),
                            patch("venice_ai.cli.commands.configure.print_error"),
                            patch("venice_ai.cli.commands.configure.print_info"),
                        ):
                            configure_cli({})

    def test_configure_cli_model_selection_current_not_in_choices(self):
        """Test model selection when current model is not in fetched choices"""
        with patch("venice_ai.cli.commands.configure.load_config") as mock_load:
            mock_load.return_value = {
                "api": {"key": "test-key-123456789"},
                "defaults": {
                    "chat_model": "old-model-not-in-list",
                    "image_model": "old-image-not-in-list",
                },
            }

            with patch("venice_ai.cli.commands.configure.os.getenv") as mock_getenv:
                mock_getenv.return_value = None

                with patch("venice_ai.cli.commands.configure._fetch_models_sync") as mock_fetch:
                    # configure now pins all six model types; return a non-empty
                    # list for every /models type so each prompt is offered.
                    mock_fetch.return_value = [
                        "llama-3.3-70b",
                        "qwen3-235b",
                        "flux-2-pro",
                        "flux-2-max",
                        "venice-sd35",
                    ]

                    with patch("venice_ai.cli.commands.configure.questionary") as mock_q:
                        confirm_responses = iter([False, True, False, False, True, False])
                        select_responses = iter(["llama-3.3-70b", "flux-2-pro"])

                        def confirm_mock(*args, **kwargs):
                            mock_prompt = MagicMock()
                            mock_prompt.ask = MagicMock(return_value=next(confirm_responses, False))
                            return mock_prompt

                        def select_mock(*args, **kwargs):
                            mock_prompt = MagicMock()
                            mock_prompt.ask = MagicMock(return_value=next(select_responses, None))
                            return mock_prompt

                        mock_q.confirm = MagicMock(side_effect=confirm_mock)
                        mock_q.select = MagicMock(side_effect=select_mock)

                        with (
                            patch("venice_ai.cli.commands.configure.console"),
                            patch("venice_ai.cli.commands.configure.print_info"),
                        ):
                            configure_cli({})

    def test_configure_cli_model_selection_returns_none(self):
        """Test model selection when user cancels selection"""
        with patch("venice_ai.cli.commands.configure.load_config") as mock_load:
            mock_load.return_value = {"api": {"key": "test-key-123456789"}}

            with patch("venice_ai.cli.commands.configure.os.getenv") as mock_getenv:
                mock_getenv.return_value = None

                with patch("venice_ai.cli.commands.configure._fetch_models_sync") as mock_fetch:
                    # configure now pins all six model types; return a non-empty
                    # list for every /models type so each prompt is offered.
                    mock_fetch.return_value = [
                        "llama-3.3-70b",
                        "qwen3-235b",
                        "flux-2-pro",
                        "flux-2-max",
                        "venice-sd35",
                    ]

                    with patch("venice_ai.cli.commands.configure.questionary") as mock_q:
                        confirm_responses = iter([False, True, False, False, True, False])
                        select_responses = iter([None, None])  # User cancels

                        def confirm_mock(*args, **kwargs):
                            mock_prompt = MagicMock()
                            mock_prompt.ask = MagicMock(return_value=next(confirm_responses, False))
                            return mock_prompt

                        def select_mock(*args, **kwargs):
                            mock_prompt = MagicMock()
                            mock_prompt.ask = MagicMock(return_value=next(select_responses, None))
                            return mock_prompt

                        mock_q.confirm = MagicMock(side_effect=confirm_mock)
                        mock_q.select = MagicMock(side_effect=select_mock)

                        with (
                            patch("venice_ai.cli.commands.configure.console"),
                            patch("venice_ai.cli.commands.configure.print_info"),
                        ):
                            configure_cli({})

    def test_configure_cli_generation_parameters_valid(self):
        """Test configuring generation parameters with valid values"""
        with patch("venice_ai.cli.commands.configure.load_config") as mock_load:
            mock_load.return_value = {"api": {"key": "test-key-123456789"}}

            with patch("venice_ai.cli.commands.configure.os.getenv") as mock_getenv:
                mock_getenv.return_value = None

                with patch("venice_ai.cli.commands.configure.questionary") as mock_q:
                    # update? No, models? No, gen params? Yes, output? No, streaming, save? No
                    confirm_responses = iter([False, False, True, False, True, False])
                    text_responses = iter(["0.8", "4096"])  # temperature, max_tokens

                    def confirm_mock(*args, **kwargs):
                        mock_prompt = MagicMock()
                        mock_prompt.ask = MagicMock(return_value=next(confirm_responses, False))
                        return mock_prompt

                    def text_mock(*args, **kwargs):
                        mock_prompt = MagicMock()
                        mock_prompt.ask = MagicMock(return_value=next(text_responses, None))
                        return mock_prompt

                    mock_q.confirm = MagicMock(side_effect=confirm_mock)
                    mock_q.text = MagicMock(side_effect=text_mock)

                    with patch("venice_ai.cli.commands.configure.console"):
                        configure_cli({})

    def test_configure_cli_generation_parameters_invalid_temperature(self):
        """Test configuring generation parameters with invalid temperature"""
        with patch("venice_ai.cli.commands.configure.load_config") as mock_load:
            mock_load.return_value = {"api": {"key": "test-key-123456789"}}

            with patch("venice_ai.cli.commands.configure.os.getenv") as mock_getenv:
                mock_getenv.return_value = None

                with patch("venice_ai.cli.commands.configure.questionary") as mock_q:
                    confirm_responses = iter([False, False, True, False, True, False])
                    text_responses = iter(["3.5", "4096"])  # temperature out of range

                    def confirm_mock(*args, **kwargs):
                        mock_prompt = MagicMock()
                        mock_prompt.ask = MagicMock(return_value=next(confirm_responses, False))
                        return mock_prompt

                    def text_mock(*args, **kwargs):
                        mock_prompt = MagicMock()
                        mock_prompt.ask = MagicMock(return_value=next(text_responses, None))
                        return mock_prompt

                    mock_q.confirm = MagicMock(side_effect=confirm_mock)
                    mock_q.text = MagicMock(side_effect=text_mock)

                    with (
                        patch("venice_ai.cli.commands.configure.console"),
                        patch("venice_ai.cli.commands.configure.print_error"),
                    ):
                        configure_cli({})
                        # Should have called print_error for out of range temp

    def test_configure_cli_generation_parameters_non_numeric_temperature(self):
        """Test configuring generation parameters with non-numeric temperature"""
        with patch("venice_ai.cli.commands.configure.load_config") as mock_load:
            mock_load.return_value = {"api": {"key": "test-key-123456789"}}

            with patch("venice_ai.cli.commands.configure.os.getenv") as mock_getenv:
                mock_getenv.return_value = None

                with patch("venice_ai.cli.commands.configure.questionary") as mock_q:
                    confirm_responses = iter([False, False, True, False, True, False])
                    text_responses = iter(["not-a-number", "4096"])  # invalid temperature

                    def confirm_mock(*args, **kwargs):
                        mock_prompt = MagicMock()
                        mock_prompt.ask = MagicMock(return_value=next(confirm_responses, False))
                        return mock_prompt

                    def text_mock(*args, **kwargs):
                        mock_prompt = MagicMock()
                        mock_prompt.ask = MagicMock(return_value=next(text_responses, None))
                        return mock_prompt

                    mock_q.confirm = MagicMock(side_effect=confirm_mock)
                    mock_q.text = MagicMock(side_effect=text_mock)

                    with (
                        patch("venice_ai.cli.commands.configure.console"),
                        patch("venice_ai.cli.commands.configure.print_error"),
                    ):
                        configure_cli({})

    def test_configure_cli_generation_parameters_negative_max_tokens(self):
        """Test configuring generation parameters with negative max tokens"""
        with patch("venice_ai.cli.commands.configure.load_config") as mock_load:
            mock_load.return_value = {"api": {"key": "test-key-123456789"}}

            with patch("venice_ai.cli.commands.configure.os.getenv") as mock_getenv:
                mock_getenv.return_value = None

                with patch("venice_ai.cli.commands.configure.questionary") as mock_q:
                    confirm_responses = iter([False, False, True, False, True, False])
                    text_responses = iter(["0.7", "-100"])  # valid temp, negative tokens

                    def confirm_mock(*args, **kwargs):
                        mock_prompt = MagicMock()
                        mock_prompt.ask = MagicMock(return_value=next(confirm_responses, False))
                        return mock_prompt

                    def text_mock(*args, **kwargs):
                        mock_prompt = MagicMock()
                        mock_prompt.ask = MagicMock(return_value=next(text_responses, None))
                        return mock_prompt

                    mock_q.confirm = MagicMock(side_effect=confirm_mock)
                    mock_q.text = MagicMock(side_effect=text_mock)

                    with (
                        patch("venice_ai.cli.commands.configure.console"),
                        patch("venice_ai.cli.commands.configure.print_error"),
                    ):
                        configure_cli({})

    def test_configure_cli_generation_parameters_non_numeric_tokens(self):
        """Test configuring generation parameters with non-numeric max tokens"""
        with patch("venice_ai.cli.commands.configure.load_config") as mock_load:
            mock_load.return_value = {"api": {"key": "test-key-123456789"}}

            with patch("venice_ai.cli.commands.configure.os.getenv") as mock_getenv:
                mock_getenv.return_value = None

                with patch("venice_ai.cli.commands.configure.questionary") as mock_q:
                    confirm_responses = iter([False, False, True, False, True, False])
                    text_responses = iter(["0.7", "abc"])  # valid temp, invalid tokens

                    def confirm_mock(*args, **kwargs):
                        mock_prompt = MagicMock()
                        mock_prompt.ask = MagicMock(return_value=next(confirm_responses, False))
                        return mock_prompt

                    def text_mock(*args, **kwargs):
                        mock_prompt = MagicMock()
                        mock_prompt.ask = MagicMock(return_value=next(text_responses, None))
                        return mock_prompt

                    mock_q.confirm = MagicMock(side_effect=confirm_mock)
                    mock_q.text = MagicMock(side_effect=text_mock)

                    with (
                        patch("venice_ai.cli.commands.configure.console"),
                        patch("venice_ai.cli.commands.configure.print_error"),
                    ):
                        configure_cli({})

    def test_configure_cli_output_settings(self):
        """Test configuring output settings"""
        with patch("venice_ai.cli.commands.configure.load_config") as mock_load:
            mock_load.return_value = {"api": {"key": "test-key-123456789"}}

            with patch("venice_ai.cli.commands.configure.os.getenv") as mock_getenv:
                mock_getenv.return_value = None

                with patch("venice_ai.cli.commands.configure.questionary") as mock_q:
                    # update? No, models? No, gen? No, output? Yes, streaming, save? No
                    confirm_responses = iter([False, False, False, True, True, False])
                    path_responses = iter(["/tmp/my-venice-images"])

                    def confirm_mock(*args, **kwargs):
                        mock_prompt = MagicMock()
                        mock_prompt.ask = MagicMock(return_value=next(confirm_responses, False))
                        return mock_prompt

                    def path_mock(*args, **kwargs):
                        mock_prompt = MagicMock()
                        mock_prompt.ask = MagicMock(return_value=next(path_responses, None))
                        return mock_prompt

                    mock_q.confirm = MagicMock(side_effect=confirm_mock)
                    mock_q.path = MagicMock(side_effect=path_mock)

                    with patch("venice_ai.cli.commands.configure.console"):
                        configure_cli({})

    def test_configure_cli_output_settings_none_path(self):
        """Test configuring output settings when path selection is cancelled"""
        with patch("venice_ai.cli.commands.configure.load_config") as mock_load:
            mock_load.return_value = {"api": {"key": "test-key-123456789"}}

            with patch("venice_ai.cli.commands.configure.os.getenv") as mock_getenv:
                mock_getenv.return_value = None

                with patch("venice_ai.cli.commands.configure.questionary") as mock_q:
                    confirm_responses = iter([False, False, False, True, True, False])
                    path_responses = iter([None])  # User cancels path selection

                    def confirm_mock(*args, **kwargs):
                        mock_prompt = MagicMock()
                        mock_prompt.ask = MagicMock(return_value=next(confirm_responses, False))
                        return mock_prompt

                    def path_mock(*args, **kwargs):
                        mock_prompt = MagicMock()
                        mock_prompt.ask = MagicMock(return_value=next(path_responses, None))
                        return mock_prompt

                    mock_q.confirm = MagicMock(side_effect=confirm_mock)
                    mock_q.path = MagicMock(side_effect=path_mock)

                    with patch("venice_ai.cli.commands.configure.console"):
                        configure_cli({})

    def test_configure_cli_streaming_disabled(self):
        """Test configuring streaming to be disabled"""
        with patch("venice_ai.cli.commands.configure.load_config") as mock_load:
            mock_load.return_value = {"api": {"key": "test-key-123456789"}}

            with patch("venice_ai.cli.commands.configure.os.getenv") as mock_getenv:
                mock_getenv.return_value = None

                with patch("venice_ai.cli.commands.configure.questionary") as mock_q:
                    # update? No, models? No, gen? No, output? No, streaming? No, save? No
                    confirm_responses = iter([False, False, False, False, False, False])

                    def confirm_mock(*args, **kwargs):
                        mock_prompt = MagicMock()
                        mock_prompt.ask = MagicMock(return_value=next(confirm_responses, False))
                        return mock_prompt

                    mock_q.confirm = MagicMock(side_effect=confirm_mock)

                    with patch("venice_ai.cli.commands.configure.console"):
                        configure_cli({})

    def test_configure_cli_save_to_default_path_success(self):
        """Test saving configuration to default path successfully"""
        with patch("venice_ai.cli.commands.configure.load_config") as mock_load:
            mock_load.return_value = {"api": {"key": "test-key-123456789"}}

            with patch("venice_ai.cli.commands.configure.os.getenv") as mock_getenv:
                mock_getenv.return_value = None

                with patch("venice_ai.cli.commands.configure.questionary") as mock_q:
                    # update? No, models? No, gen? No, output? No, streaming? Yes, save? Yes, default path? Yes
                    confirm_responses = iter([False, False, False, False, True, True, True])

                    def confirm_mock(*args, **kwargs):
                        mock_prompt = MagicMock()
                        mock_prompt.ask = MagicMock(return_value=next(confirm_responses, False))
                        return mock_prompt

                    mock_q.confirm = MagicMock(side_effect=confirm_mock)

                    with (
                        patch("venice_ai.cli.commands.configure.console"),
                        patch("venice_ai.cli.commands.configure.save_config") as mock_save,
                        patch("venice_ai.cli.commands.configure.print_success"),
                    ):
                        configure_cli({})
                        mock_save.assert_called_once()

    def test_configure_cli_save_to_default_path_failure(self):
        """Test saving configuration to default path with failure"""
        with patch("venice_ai.cli.commands.configure.load_config") as mock_load:
            mock_load.return_value = {"api": {"key": "test-key-123456789"}}

            with patch("venice_ai.cli.commands.configure.os.getenv") as mock_getenv:
                mock_getenv.return_value = None

                with patch("venice_ai.cli.commands.configure.questionary") as mock_q:
                    confirm_responses = iter([False, False, False, False, True, True, True])

                    def confirm_mock(*args, **kwargs):
                        mock_prompt = MagicMock()
                        mock_prompt.ask = MagicMock(return_value=next(confirm_responses, False))
                        return mock_prompt

                    mock_q.confirm = MagicMock(side_effect=confirm_mock)

                    with (
                        patch("venice_ai.cli.commands.configure.console"),
                        patch("venice_ai.cli.commands.configure.save_config") as mock_save,
                    ):
                        mock_save.side_effect = Exception("Permission denied")
                        with patch("venice_ai.cli.commands.configure.print_error") as mock_error:
                            configure_cli({})
                            mock_error.assert_called()

    def test_configure_cli_save_to_custom_path_success(self):
        """Test saving configuration to custom path successfully"""
        with patch("venice_ai.cli.commands.configure.load_config") as mock_load:
            mock_load.return_value = {"api": {"key": "test-key-123456789"}}

            with patch("venice_ai.cli.commands.configure.os.getenv") as mock_getenv:
                mock_getenv.return_value = None

                with patch("venice_ai.cli.commands.configure.questionary") as mock_q:
                    # update? No, models? No, gen? No, output? No, streaming? Yes, save? Yes, default? No
                    confirm_responses = iter([False, False, False, False, True, True, False])
                    path_responses = iter(["/tmp/custom-venice-config.yaml"])

                    def confirm_mock(*args, **kwargs):
                        mock_prompt = MagicMock()
                        mock_prompt.ask = MagicMock(return_value=next(confirm_responses, False))
                        return mock_prompt

                    def path_mock(*args, **kwargs):
                        mock_prompt = MagicMock()
                        mock_prompt.ask = MagicMock(return_value=next(path_responses, None))
                        return mock_prompt

                    mock_q.confirm = MagicMock(side_effect=confirm_mock)
                    mock_q.path = MagicMock(side_effect=path_mock)

                    with (
                        patch("venice_ai.cli.commands.configure.console"),
                        patch("venice_ai.cli.commands.configure.save_config") as mock_save,
                        patch("venice_ai.cli.commands.configure.print_success"),
                    ):
                        configure_cli({})
                        mock_save.assert_called_once()

    def test_configure_cli_save_to_custom_path_none(self):
        """Test when user cancels custom path input"""
        with patch("venice_ai.cli.commands.configure.load_config") as mock_load:
            mock_load.return_value = {"api": {"key": "test-key-123456789"}}

            with patch("venice_ai.cli.commands.configure.os.getenv") as mock_getenv:
                mock_getenv.return_value = None

                with patch("venice_ai.cli.commands.configure.questionary") as mock_q:
                    confirm_responses = iter([False, False, False, False, True, True, False])
                    path_responses = iter([None])  # User cancels path input

                    def confirm_mock(*args, **kwargs):
                        mock_prompt = MagicMock()
                        mock_prompt.ask = MagicMock(return_value=next(confirm_responses, False))
                        return mock_prompt

                    def path_mock(*args, **kwargs):
                        mock_prompt = MagicMock()
                        mock_prompt.ask = MagicMock(return_value=next(path_responses, None))
                        return mock_prompt

                    mock_q.confirm = MagicMock(side_effect=confirm_mock)
                    mock_q.path = MagicMock(side_effect=path_mock)

                    with (
                        patch("venice_ai.cli.commands.configure.console"),
                        patch("venice_ai.cli.commands.configure.save_config") as mock_save,
                    ):
                        configure_cli({})
                        mock_save.assert_not_called()

    def test_configure_cli_save_to_custom_path_failure(self):
        """Test saving configuration to custom path with failure"""
        with patch("venice_ai.cli.commands.configure.load_config") as mock_load:
            mock_load.return_value = {"api": {"key": "test-key-123456789"}}

            with patch("venice_ai.cli.commands.configure.os.getenv") as mock_getenv:
                mock_getenv.return_value = None

                with patch("venice_ai.cli.commands.configure.questionary") as mock_q:
                    confirm_responses = iter([False, False, False, False, True, True, False])
                    path_responses = iter(["/tmp/custom-venice-config.yaml"])

                    def confirm_mock(*args, **kwargs):
                        mock_prompt = MagicMock()
                        mock_prompt.ask = MagicMock(return_value=next(confirm_responses, False))
                        return mock_prompt

                    def path_mock(*args, **kwargs):
                        mock_prompt = MagicMock()
                        mock_prompt.ask = MagicMock(return_value=next(path_responses, None))
                        return mock_prompt

                    mock_q.confirm = MagicMock(side_effect=confirm_mock)
                    mock_q.path = MagicMock(side_effect=path_mock)

                    with (
                        patch("venice_ai.cli.commands.configure.console"),
                        patch("venice_ai.cli.commands.configure.save_config") as mock_save,
                    ):
                        mock_save.side_effect = Exception("Permission denied")
                        with patch("venice_ai.cli.commands.configure.print_error") as mock_error:
                            configure_cli({})
                            mock_error.assert_called()

    def test_configure_cli_skip_save(self):
        """Test skipping configuration save"""
        with patch("venice_ai.cli.commands.configure.load_config") as mock_load:
            mock_load.return_value = {"api": {"key": "test-key-123456789"}}

            with patch("venice_ai.cli.commands.configure.os.getenv") as mock_getenv:
                mock_getenv.return_value = None

                with patch("venice_ai.cli.commands.configure.questionary") as mock_q:
                    # update? No, models? No, gen? No, output? No, streaming? Yes, save? No
                    confirm_responses = iter([False, False, False, False, True, False])

                    def confirm_mock(*args, **kwargs):
                        mock_prompt = MagicMock()
                        mock_prompt.ask = MagicMock(return_value=next(confirm_responses, False))
                        return mock_prompt

                    mock_q.confirm = MagicMock(side_effect=confirm_mock)

                    with (
                        patch("venice_ai.cli.commands.configure.console"),
                        patch("venice_ai.cli.commands.configure.save_config") as mock_save,
                        patch("venice_ai.cli.commands.configure.print_info"),
                    ):
                        configure_cli({})
                        mock_save.assert_not_called()

    def test_configure_cli_full_flow_with_models_fetch(self):
        """Test complete configuration flow with successful model fetching"""
        with patch("venice_ai.cli.commands.configure.load_config") as mock_load:
            mock_load.return_value = {}

            with patch("venice_ai.cli.commands.configure.os.getenv") as mock_getenv:
                mock_getenv.return_value = None

                with patch("venice_ai.cli.commands.configure._fetch_models_sync") as mock_fetch:
                    mock_fetch.return_value = [
                        "llama-3.3-70b",
                        "qwen3-235b",
                        "deepseek-v3",
                        "flux-2-pro",
                        "flux-2-max",
                        "venice-sd35",
                    ]

                    with patch("venice_ai.cli.commands.configure.questionary") as mock_q:
                        # password for API key, then various confirms/selects
                        password_responses = iter(["new-api-key-for-testing"])
                        confirm_responses = iter(
                            [True, False, False, True, True, True]
                        )  # models, gen, output, streaming, save, default
                        select_responses = iter(["llama-3.3-70b", "flux-2-pro"])

                        def confirm_mock(*args, **kwargs):
                            mock_prompt = MagicMock()
                            mock_prompt.ask = MagicMock(return_value=next(confirm_responses, False))
                            return mock_prompt

                        def password_mock(*args, **kwargs):
                            mock_prompt = MagicMock()
                            mock_prompt.ask = MagicMock(return_value=next(password_responses, None))
                            return mock_prompt

                        def select_mock(*args, **kwargs):
                            mock_prompt = MagicMock()
                            mock_prompt.ask = MagicMock(return_value=next(select_responses, None))
                            return mock_prompt

                        mock_q.confirm = MagicMock(side_effect=confirm_mock)
                        mock_q.password = MagicMock(side_effect=password_mock)
                        mock_q.select = MagicMock(side_effect=select_mock)

                        with (
                            patch("venice_ai.cli.commands.configure.console"),
                            patch("venice_ai.cli.commands.configure.save_config"),
                            patch("venice_ai.cli.commands.configure.print_success"),
                            patch("venice_ai.cli.commands.configure.print_info"),
                        ):
                            configure_cli({})

    def test_configure_cli_pins_all_six_model_types(self):
        """configure pins all six model types (chat/image/tts/stt/embedding/video×2)."""
        with patch("venice_ai.cli.commands.configure.load_config") as mock_load:
            mock_load.return_value = {"api": {"key": "test-key-123456789"}}

            with patch("venice_ai.cli.commands.configure.os.getenv") as mock_getenv:
                mock_getenv.return_value = None

                with patch("venice_ai.cli.commands.configure._fetch_models_sync") as mock_fetch:
                    # Every /models type returns a list whose first entry we'll select.
                    mock_fetch.return_value = ["picked-model", "other-model"]

                    with patch("venice_ai.cli.commands.configure.questionary") as mock_q:
                        # update key? False, models? True, gen? False, output? False,
                        # streaming? True, save? True, save-to-path? True
                        confirm_responses = iter([False, True, False, False, True, True, True])
                        # Each of the 7 select prompts returns the model id.
                        select_responses = iter(["picked-model"] * 7)

                        def confirm_mock(*args, **kwargs):
                            p = MagicMock()
                            p.ask = MagicMock(return_value=next(confirm_responses, False))
                            return p

                        def select_mock(*args, **kwargs):
                            p = MagicMock()
                            p.ask = MagicMock(return_value=next(select_responses, None))
                            return p

                        mock_q.confirm = MagicMock(side_effect=confirm_mock)
                        mock_q.select = MagicMock(side_effect=select_mock)

                        with (
                            patch("venice_ai.cli.commands.configure.console"),
                            patch("venice_ai.cli.commands.configure.save_config") as mock_save,
                            patch("venice_ai.cli.commands.configure.print_success"),
                            patch("venice_ai.cli.commands.configure.print_info"),
                        ):
                            configure_cli({})

                        # The config handed to save_config must pin all six types.
                        saved_config = mock_save.call_args[0][0]
                        defaults = saved_config["defaults"]
                        for key in (
                            "chat_model",
                            "image_model",
                            "tts_model",
                            "stt_model",
                            "embedding_model",
                            "video_t2v_model",
                            "video_i2v_model",
                        ):
                            assert defaults[key] == "picked-model", f"{key} not pinned"

                        # Fetch was called for every distinct /models type (video once).
                        fetched_types = {c.args[1] for c in mock_fetch.call_args_list}
                        assert fetched_types == {
                            "text",
                            "image",
                            "tts",
                            "asr",
                            "embedding",
                            "video",
                        }


class TestConfigureCliEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_configure_cli_empty_new_api_key(self):
        """Test when user provides empty API key"""
        with patch("venice_ai.cli.commands.configure.load_config") as mock_load:
            mock_load.return_value = {}

            with patch("venice_ai.cli.commands.configure.os.getenv") as mock_getenv:
                mock_getenv.return_value = None

                with patch("venice_ai.cli.commands.configure.questionary") as mock_q:
                    password_responses = iter([""])  # Empty API key
                    confirm_responses = iter([False, False, False, True, False])

                    def confirm_mock(*args, **kwargs):
                        mock_prompt = MagicMock()
                        mock_prompt.ask = MagicMock(return_value=next(confirm_responses, False))
                        return mock_prompt

                    def password_mock(*args, **kwargs):
                        mock_prompt = MagicMock()
                        mock_prompt.ask = MagicMock(return_value=next(password_responses, None))
                        return mock_prompt

                    mock_q.confirm = MagicMock(side_effect=confirm_mock)
                    mock_q.password = MagicMock(side_effect=password_mock)

                    with patch("venice_ai.cli.commands.configure.console"):
                        configure_cli({})

    def test_configure_cli_model_current_in_fetched_list(self):
        """Test when current model is in the fetched list"""
        with patch("venice_ai.cli.commands.configure.load_config") as mock_load:
            mock_load.return_value = {
                "api": {"key": "test-key-123456789"},
                "defaults": {"chat_model": "qwen3-235b", "image_model": "flux-2-max"},
            }

            with patch("venice_ai.cli.commands.configure.os.getenv") as mock_getenv:
                mock_getenv.return_value = None

                with patch("venice_ai.cli.commands.configure._fetch_models_sync") as mock_fetch:
                    # Lists contain the current chat/image models (qwen3-235b,
                    # flux-2-max) to exercise the "current in choices" path.
                    mock_fetch.return_value = [
                        "llama-3.3-70b",
                        "qwen3-235b",
                        "flux-2-pro",
                        "flux-2-max",
                    ]

                    with patch("venice_ai.cli.commands.configure.questionary") as mock_q:
                        confirm_responses = iter([False, True, False, False, True, False])
                        select_responses = iter(["qwen3-235b", "flux-2-max"])

                        def confirm_mock(*args, **kwargs):
                            mock_prompt = MagicMock()
                            mock_prompt.ask = MagicMock(return_value=next(confirm_responses, False))
                            return mock_prompt

                        def select_mock(*args, **kwargs):
                            mock_prompt = MagicMock()
                            mock_prompt.ask = MagicMock(return_value=next(select_responses, None))
                            return mock_prompt

                        mock_q.confirm = MagicMock(side_effect=confirm_mock)
                        mock_q.select = MagicMock(side_effect=select_mock)

                        with (
                            patch("venice_ai.cli.commands.configure.console"),
                            patch("venice_ai.cli.commands.configure.print_info"),
                        ):
                            configure_cli({})

    def test_configure_cli_zero_tokens(self):
        """Test when max tokens is set to zero"""
        with patch("venice_ai.cli.commands.configure.load_config") as mock_load:
            mock_load.return_value = {"api": {"key": "test-key-123456789"}}

            with patch("venice_ai.cli.commands.configure.os.getenv") as mock_getenv:
                mock_getenv.return_value = None

                with patch("venice_ai.cli.commands.configure.questionary") as mock_q:
                    confirm_responses = iter([False, False, True, False, True, False])
                    text_responses = iter(["0.7", "0"])  # Zero tokens

                    def confirm_mock(*args, **kwargs):
                        mock_prompt = MagicMock()
                        mock_prompt.ask = MagicMock(return_value=next(confirm_responses, False))
                        return mock_prompt

                    def text_mock(*args, **kwargs):
                        mock_prompt = MagicMock()
                        mock_prompt.ask = MagicMock(return_value=next(text_responses, None))
                        return mock_prompt

                    mock_q.confirm = MagicMock(side_effect=confirm_mock)
                    mock_q.text = MagicMock(side_effect=text_mock)

                    with (
                        patch("venice_ai.cli.commands.configure.console"),
                        patch("venice_ai.cli.commands.configure.print_error"),
                    ):
                        configure_cli({})


class TestSaveConfigPermissions:
    """save_config must not leave the API key world-readable."""

    def test_save_config_sets_owner_only_permissions(self, tmp_path):
        """The written config.yaml must have mode 0o600 (owner read/write only).

        Regression guard: previously save_config wrote the file with the
        process umask default (typically 0o644 — world-readable), exposing the
        plaintext API key.
        """
        import os
        import stat

        from venice_ai.cli.config import save_config

        config_path = tmp_path / "config.yaml"
        save_config({"api": {"key": "sk-secret-dummy-key"}}, config_path)

        mode = stat.S_IMODE(os.stat(config_path).st_mode)
        assert mode == 0o600, f"expected 0o600, got {oct(mode)}"
