"""
Tests for configuration management
"""

from unittest.mock import patch

import pytest

from venice_ai.cli.config import (
    DEFAULT_CONFIG,
    ensure_api_key,
    get_active_config_path,
    get_api_key,
    get_base_url,
    get_client_kwargs,
    load_config,
    save_config,
    set_active_config_path,
)


@pytest.fixture(autouse=True)
def _reset_active_config_path():
    """Reset the module-global active config path between tests.

    ``_ACTIVE_CONFIG_PATH`` persists within an xdist worker, so leaving it set
    would make order-dependent tests (e.g. the env/config priority tests that
    patch ``DEFAULT_CONFIG_PATH``) read a stale path.
    """
    set_active_config_path(None)
    yield
    set_active_config_path(None)


class TestLoadConfig:
    """Test configuration loading"""

    def test_load_default_config_when_file_missing(self, tmp_path):
        """Test loading default config when file doesn't exist"""
        non_existent = tmp_path / "nonexistent.yaml"
        config = load_config(non_existent)

        assert config == DEFAULT_CONFIG
        assert "api" in config
        assert "defaults" in config
        assert "output" in config
        assert "features" in config

    def test_load_config_from_file(self, tmp_path):
        """Test loading config from existing file"""
        config_file = tmp_path / "config.yaml"
        test_config = """
api:
  base_url: https://custom.api.url
  key: test-api-key
defaults:
  chat_model: custom-model
"""
        config_file.write_text(test_config)

        config = load_config(config_file)
        assert config["api"]["base_url"] == "https://custom.api.url"
        assert config["api"]["key"] == "test-api-key"
        assert config["defaults"]["chat_model"] == "custom-model"

    def test_load_config_merges_with_defaults(self, tmp_path):
        """Test loaded config merges with defaults"""
        config_file = tmp_path / "config.yaml"
        # Only override one setting
        config_file.write_text("api:\n  base_url: https://custom.url\n")

        config = load_config(config_file)
        # Custom value should be present
        assert config["api"]["base_url"] == "https://custom.url"
        # Default values should still be present
        assert "defaults" in config
        assert "output" in config

    def test_load_config_handles_empty_file(self, tmp_path):
        """Test loading empty config file returns defaults"""
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")

        config = load_config(config_file)
        assert config == DEFAULT_CONFIG

    def test_load_config_handles_invalid_yaml(self, tmp_path, capsys):
        """Test loading invalid YAML prints warning and returns defaults"""
        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("invalid: yaml: content: ::::")

        config = load_config(config_file)
        # Should return defaults on error
        assert config == DEFAULT_CONFIG

    def test_load_config_none_path_uses_default(self):
        """Test None path uses DEFAULT_CONFIG_PATH"""
        with patch("venice_ai.cli.config.DEFAULT_CONFIG_PATH") as mock_path:
            mock_path.exists.return_value = False
            config = load_config(None)
            assert config == DEFAULT_CONFIG


class TestLegacyKeyMigration:
    """Test migration of legacy max_tokens key to max_completion_tokens"""

    def test_migrate_max_tokens_to_max_completion_tokens(self, tmp_path):
        """Test config with legacy max_tokens is migrated to max_completion_tokens"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("defaults:\n  max_tokens: 4096\n  chat_model: venice-uncensored\n")

        config = load_config(config_file)
        assert config["defaults"]["max_completion_tokens"] == 4096
        assert "max_tokens" not in config["defaults"]

    def test_cleanup_stale_max_tokens_when_both_present(self, tmp_path):
        """Test config with both keys keeps max_completion_tokens and removes stale max_tokens"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("defaults:\n  max_tokens: 1024\n  max_completion_tokens: 8192\n")

        config = load_config(config_file)
        assert config["defaults"]["max_completion_tokens"] == 8192
        # Stale max_tokens should be cleaned up
        assert "max_tokens" not in config["defaults"]

    def test_no_migration_when_only_max_completion_tokens(self, tmp_path):
        """Test config with only max_completion_tokens is unchanged"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("defaults:\n  max_completion_tokens: 4096\n")

        config = load_config(config_file)
        assert config["defaults"]["max_completion_tokens"] == 4096
        assert "max_tokens" not in config["defaults"]


class TestSaveConfig:
    """Test configuration saving"""

    def test_save_config_creates_file(self, tmp_path):
        """Test save_config creates file if it doesn't exist"""
        config_file = tmp_path / "config.yaml"
        test_config = {
            "api": {"key": "test-key"},
            "defaults": {"chat_model": "test-model"},
        }

        save_config(test_config, config_file)

        assert config_file.exists()
        loaded = load_config(config_file)
        assert loaded["api"]["key"] == "test-key"

    def test_save_config_creates_directory(self, tmp_path):
        """Test save_config creates parent directories"""
        nested_path = tmp_path / "nested" / "dirs" / "config.yaml"
        test_config = {"api": {"key": "test"}}

        save_config(test_config, nested_path)

        assert nested_path.exists()
        assert nested_path.parent.exists()

    def test_save_config_overwrites_existing(self, tmp_path):
        """Test save_config overwrites existing file"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("old: data\n")

        new_config = {"api": {"key": "new-key"}}
        save_config(new_config, config_file)

        loaded = load_config(config_file)
        assert loaded["api"]["key"] == "new-key"
        assert "old" not in loaded

    def test_save_config_none_path_uses_default(self, tmp_path):
        """Test None path uses DEFAULT_CONFIG_PATH"""
        test_config = {"api": {"key": "test"}}

        with patch("venice_ai.cli.config.DEFAULT_CONFIG_PATH", tmp_path / "default.yaml"):
            save_config(test_config, None)
            assert (tmp_path / "default.yaml").exists()


class TestGetAPIKey:
    """Test API key retrieval"""

    def test_get_api_key_from_environment(self, monkeypatch):
        """Test getting API key from environment variable"""
        monkeypatch.setenv("VENICE_API_KEY", "env-api-key")

        api_key = get_api_key()
        assert api_key == "env-api-key"

    def test_get_api_key_from_config(self, tmp_path, monkeypatch):
        """Test getting API key from config file when env var not set"""
        # Ensure no env var
        monkeypatch.delenv("VENICE_API_KEY", raising=False)

        config_file = tmp_path / "config.yaml"
        config_file.write_text("api:\n  key: config-api-key\n")

        with patch("venice_ai.cli.config.DEFAULT_CONFIG_PATH", config_file):
            api_key = get_api_key()
            assert api_key == "config-api-key"

    def test_get_api_key_priority_env_over_config(self, tmp_path, monkeypatch):
        """Test environment variable takes priority over config file"""
        monkeypatch.setenv("VENICE_API_KEY", "env-key")

        config_file = tmp_path / "config.yaml"
        config_file.write_text("api:\n  key: config-key\n")

        with patch("venice_ai.cli.config.DEFAULT_CONFIG_PATH", config_file):
            api_key = get_api_key()
            assert api_key == "env-key"

    def test_get_api_key_returns_none_when_missing(self, monkeypatch, tmp_path):
        """Test returns None when no API key found"""
        monkeypatch.delenv("VENICE_API_KEY", raising=False)

        non_existent = tmp_path / "missing.yaml"
        with patch("venice_ai.cli.config.DEFAULT_CONFIG_PATH", non_existent):
            api_key = get_api_key()
            assert api_key is None


class TestEnsureAPIKey:
    """Test ensure_api_key function"""

    def test_ensure_api_key_returns_key_when_present(self, monkeypatch):
        """Test ensure_api_key returns key when available"""
        monkeypatch.setenv("VENICE_API_KEY", "test-key")

        api_key = ensure_api_key()
        assert api_key == "test-key"

    def test_ensure_api_key_raises_when_missing(self, monkeypatch, tmp_path):
        """Test ensure_api_key raises click.ClickException when no key found"""
        import click as _click

        monkeypatch.delenv("VENICE_API_KEY", raising=False)

        non_existent = tmp_path / "missing.yaml"
        with patch("venice_ai.cli.config.DEFAULT_CONFIG_PATH", non_existent):
            with pytest.raises(_click.ClickException) as exc_info:
                ensure_api_key()

            assert "No API key found" in exc_info.value.message
            assert "VENICE_API_KEY" in exc_info.value.message


class TestActiveConfigPath:
    """Test the active-config-path helpers."""

    def test_default_active_path_is_none(self):
        assert get_active_config_path() is None

    def test_set_and_get_active_path(self, tmp_path):
        p = tmp_path / "cfg.yaml"
        set_active_config_path(p)
        assert get_active_config_path() == p

    def test_get_api_key_reads_explicit_config_path(self, tmp_path, monkeypatch):
        """get_api_key(config_path=tmpfile) returns the file's key when env unset."""
        monkeypatch.delenv("VENICE_API_KEY", raising=False)
        config_file = tmp_path / "config.yaml"
        config_file.write_text("api:\n  key: file-api-key\n")

        assert get_api_key(config_path=config_file) == "file-api-key"

    def test_env_wins_over_explicit_config_path(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VENICE_API_KEY", "env-api-key")
        config_file = tmp_path / "config.yaml"
        config_file.write_text("api:\n  key: file-api-key\n")

        assert get_api_key(config_path=config_file) == "env-api-key"

    def test_get_api_key_uses_active_path_when_none(self, tmp_path, monkeypatch):
        monkeypatch.delenv("VENICE_API_KEY", raising=False)
        config_file = tmp_path / "config.yaml"
        config_file.write_text("api:\n  key: active-key\n")
        set_active_config_path(config_file)

        assert get_api_key() == "active-key"

    def test_get_base_url_from_config(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("api:\n  base_url: https://custom.example/api\n")

        assert get_base_url(config_path=config_file) == "https://custom.example/api"

    def test_get_base_url_falls_back_to_default(self, tmp_path):
        """Config without api.base_url falls back to DEFAULT_CONFIG's url."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("api:\n  key: only-key\n")

        assert get_base_url(config_path=config_file) == DEFAULT_CONFIG["api"]["base_url"]

    def test_get_client_kwargs_returns_both(self, tmp_path, monkeypatch):
        monkeypatch.delenv("VENICE_API_KEY", raising=False)
        config_file = tmp_path / "config.yaml"
        config_file.write_text("api:\n  key: kw-key\n  base_url: https://kw.example/api\n")

        kwargs = get_client_kwargs(config_path=config_file)
        assert kwargs == {
            "api_key": "kw-key",
            "base_url": "https://kw.example/api",
        }


class TestDefaultConfig:
    """Test default configuration structure"""

    def test_default_config_structure(self):
        """Test DEFAULT_CONFIG has expected structure"""
        assert "api" in DEFAULT_CONFIG
        assert "defaults" in DEFAULT_CONFIG
        assert "output" in DEFAULT_CONFIG
        assert "features" in DEFAULT_CONFIG

    def test_default_config_api_section(self):
        """Test default API configuration"""
        assert "base_url" in DEFAULT_CONFIG["api"]
        assert "venice.ai" in DEFAULT_CONFIG["api"]["base_url"].lower()

    def test_default_config_defaults_section(self):
        """Test default parameter settings (model IDs are resolved at runtime, not hardcoded)"""
        defaults = DEFAULT_CONFIG["defaults"]
        assert "max_completion_tokens" in defaults
        assert "temperature" in defaults
        # Model IDs must NOT be hardcoded in DEFAULT_CONFIG; they are resolved at runtime
        assert "chat_model" not in defaults
        assert "image_model" not in defaults

    def test_default_config_features_section(self):
        """Test default feature flags"""
        features = DEFAULT_CONFIG["features"]
        assert "streaming" in features
        assert "cost_tracking" in features
        assert features["streaming"] is True


class TestDefaultConfigNoHardcodedModels:
    def test_no_model_id_defaults(self):
        from venice_ai.cli.config import DEFAULT_CONFIG

        defaults = DEFAULT_CONFIG["defaults"]
        for key in (
            "chat_model",
            "image_model",
            "tts_model",
            "stt_model",
            "embedding_model",
            "video_t2v_model",
            "video_i2v_model",
        ):
            assert key not in defaults, f"{key} must not hardcode a model ID"

    def test_non_model_defaults_preserved(self):
        from venice_ai.cli.config import DEFAULT_CONFIG

        defaults = DEFAULT_CONFIG["defaults"]
        assert defaults["max_completion_tokens"] == 2048
        assert defaults["temperature"] == 0.7

    def test_fallback_model_lists_removed(self):
        import venice_ai.cli.config as cfg

        assert not hasattr(cfg, "FALLBACK_CHAT_MODELS")
        assert not hasattr(cfg, "FALLBACK_IMAGE_MODELS")
