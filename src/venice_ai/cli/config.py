"""
Configuration management for Venice AI CLI
"""

import os
from pathlib import Path
from typing import Any

import click
from dotenv import load_dotenv

# ``yaml`` is gated behind the ``[cli]`` extra. Lazy-import it inside the
# read/write helpers so subcommands that don't touch the config file (e.g.
# ``venice lint``) work on a bare ``pip install venice-ai`` install.

# Load environment variables
load_dotenv()

DEFAULT_CONFIG_PATH = Path.home() / ".venice" / "config.yaml"
DEFAULT_CONFIG: dict[str, Any] = {
    "api": {"base_url": "https://api.venice.ai/api/v1"},
    "defaults": {
        # Model IDs are intentionally NOT defaulted here. When unset, the CLI
        # resolves a model from the live /models API at runtime (see
        # cli/_model_defaults.py). A user's choice in `venice configure` is
        # written back here per-key.
        "max_completion_tokens": 2048,
        "temperature": 0.7,
    },
    "output": {
        "format": "markdown",
        "images_dir": str(Path.home() / "Pictures" / "venice"),
    },
    "features": {"streaming": True, "cost_tracking": True},
}

# Active config path selected by the root ``venice --config`` option. Set once
# in the CLI root callback so downstream commands resolve the key / base_url
# from the user-chosen file instead of always defaulting to
# ``DEFAULT_CONFIG_PATH``.
_ACTIVE_CONFIG_PATH: Path | None = None


def set_active_config_path(p: Path | None) -> None:
    """Record the active config path (from ``venice --config``)."""
    global _ACTIVE_CONFIG_PATH
    _ACTIVE_CONFIG_PATH = p


def get_active_config_path() -> Path | None:
    """Return the active config path, if one was set via ``--config``."""
    return _ACTIVE_CONFIG_PATH


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge override into base, returning new dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load configuration from file or create default"""
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    if config_path.exists():
        try:
            try:
                import yaml  # lazy: only imported when a config file actually exists
            except ModuleNotFoundError:
                click.echo(
                    f"Warning: PyYAML not installed; cannot read {config_path}. "
                    "Install with: pip install 'venice-ai[cli]'",
                    err=True,
                )
                return DEFAULT_CONFIG.copy()

            with open(config_path) as f:
                config = yaml.safe_load(f) or {}
                # Migrate legacy key
                defaults = config.get("defaults", {})
                if "max_tokens" in defaults:
                    if "max_completion_tokens" not in defaults:
                        defaults["max_completion_tokens"] = defaults.pop("max_tokens")
                    else:
                        # Both keys present — remove stale max_tokens
                        del defaults["max_tokens"]
                    # Persist migration to disk so it doesn't re-run
                    try:
                        save_config(config, config_path)
                        click.echo(
                            "Migrated 'max_tokens' to 'max_completion_tokens' in config file"
                        )
                    except Exception as e:
                        click.echo(
                            f"Warning: Config migration could not be saved to disk: {e}",
                            err=True,
                        )
                # Deep merge with defaults to preserve nested keys
                return _deep_merge(DEFAULT_CONFIG, config)
        except Exception as e:
            click.echo(
                f"Warning: Could not load config from {config_path}: {e}",
                err=True,
            )

    return DEFAULT_CONFIG.copy()


def save_config(config: dict[str, Any], config_path: Path | None = None) -> None:
    """Save configuration to file"""
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    import yaml  # lazy: only imported when actually persisting config

    # Create directory if it doesn't exist
    config_path.parent.mkdir(parents=True, exist_ok=True)
    # The config dir may hold a plaintext API key; restrict it to the owner
    # (0o700) so other local users can't list / traverse it.
    os.chmod(config_path.parent, 0o700)

    # The config file may contain a plaintext API key, so it is created with
    # owner-only permissions (0o600) rather than the umask default (often
    # world-readable 0o644). os.open with the mode applies it at creation
    # time; a plain open() followed by chmod would leave a window in which
    # the key is on disk world-readable.
    fd = os.open(config_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False)

    # os.open honours the umask, so an existing file (or a restrictive umask
    # having been widened) still needs the explicit narrowing.
    os.chmod(config_path, 0o600)


def get_api_key(config_path: Path | None = None) -> str | None:
    """Get API key from environment or config.

    When ``config_path`` is ``None`` it defaults to the active config path set
    via ``venice --config`` (see :func:`set_active_config_path`). Priority is
    always the ``VENICE_API_KEY`` env var first, then the resolved config
    file's ``api.key``.
    """
    if config_path is None:
        config_path = _ACTIVE_CONFIG_PATH

    # Priority: 1. Environment variable
    api_key = os.getenv("VENICE_API_KEY")

    if not api_key:
        # 2. Try config file (config_path may still be None → DEFAULT_CONFIG_PATH)
        config = load_config(config_path)
        api_key = config.get("api", {}).get("key")

    return api_key


def ensure_api_key(config_path: Path | None = None) -> str:
    """Get API key, raising error if not found."""
    api_key = get_api_key(config_path)
    if not api_key:
        raise click.ClickException(
            "No API key found. Set it using:\n"
            "  1. Run: venice configure\n"
            "  2. Or set environment variable: export VENICE_API_KEY=your-key"
        )
    return api_key


def get_base_url(config_path: Path | None = None) -> str:
    """Return the API base URL from the resolved config.

    Falls back to :data:`DEFAULT_CONFIG`'s ``api.base_url`` when the config
    file omits it.
    """
    if config_path is None:
        config_path = _ACTIVE_CONFIG_PATH

    config = load_config(config_path)
    base_url = config.get("api", {}).get("base_url")
    if not base_url:
        base_url = DEFAULT_CONFIG["api"]["base_url"]
    return str(base_url)


def get_client_kwargs(config_path: Path | None = None) -> dict[str, Any]:
    """Return ``VeniceClient`` constructor kwargs (api_key + base_url).

    Resolves the API key (raising if absent) and the base URL from the
    resolved config. This is the canonical helper for CLI command modules
    constructing a client.
    """
    return {
        "api_key": ensure_api_key(config_path),
        "base_url": get_base_url(config_path),
    }
