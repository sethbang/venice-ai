"""
Configuration command for Venice AI CLI
"""

import asyncio
import os
from pathlib import Path
from typing import Any

import questionary
from rich.panel import Panel

from venice_ai import VeniceClient
from venice_ai.resources.models import ModelListType

from ..config import (
    DEFAULT_CONFIG_PATH,
    get_active_config_path,
    get_base_url,
    load_config,
    save_config,
)
from ..utils import console, print_error, print_info, print_success


async def _fetch_models_by_type(api_key: str, model_type: ModelListType) -> list[str]:
    """Fetch available model IDs from the Venice API by type."""
    try:
        async with VeniceClient(
            api_key=api_key, base_url=get_base_url(get_active_config_path())
        ) as client:
            response = await client.models.list(type=model_type)
            if response.data:
                return [model.id for model in response.data]
            return []
    except Exception:
        return []


def _fetch_models_sync(api_key: str, model_type: ModelListType) -> list[str]:
    """Synchronous wrapper for fetching models."""
    return asyncio.run(_fetch_models_by_type(api_key, model_type))


def configure_cli(ctx: dict[str, Any]) -> None:
    """Interactive configuration setup for Venice AI CLI"""

    console.print(
        Panel(
            "[bold cyan]Venice AI CLI Configuration[/bold cyan]\n\n"
            "Let's set up your Venice AI CLI configuration.",
            border_style="cyan",
        )
    )

    # Honor ``venice --config <path>``: read from and write back to the
    # user-chosen file (falling back to the default location).
    active_path = get_active_config_path() or DEFAULT_CONFIG_PATH

    # Load existing config
    config = load_config(active_path)

    # API Key configuration
    console.print("\n[bold]API Configuration[/bold]")

    # Check for existing API key
    current_key = os.getenv("VENICE_API_KEY") or config.get("api", {}).get("key", "")
    if current_key:
        masked_key = current_key[:8] + "..." + current_key[-4:] if len(current_key) > 12 else "***"
        console.print(f"Current API key: {masked_key}")

        if questionary.confirm("Do you want to update the API key?").ask():
            api_key = questionary.password("Enter your Venice AI API key:").ask()
            if api_key:
                config.setdefault("api", {})["key"] = api_key
                current_key = api_key  # Update for model fetching
    else:
        console.print("[yellow]No API key found[/yellow]")
        api_key = questionary.password("Enter your Venice AI API key:").ask()
        if api_key:
            config.setdefault("api", {})["key"] = api_key
            current_key = api_key  # Update for model fetching

    # Model defaults
    console.print("\n[bold]Default Models[/bold]")

    if questionary.confirm("Configure default models?", default=True).ask():
        # Each pinnable default: (config key, /models API type, human label).
        # ``video`` is fetched once and reused for both t2v and i2v prompts.
        model_choices: list[tuple[str, ModelListType, str]] = [
            ("chat_model", "text", "chat"),
            ("image_model", "image", "image"),
            ("tts_model", "tts", "text-to-speech"),
            ("stt_model", "asr", "speech-to-text"),
            ("embedding_model", "embedding", "embedding"),
            ("video_t2v_model", "video", "text-to-video"),
            ("video_i2v_model", "video", "image-to-video"),
        ]

        if current_key:
            print_info("Fetching available models from Venice API...")

        # Cache fetched lists per /models type so video isn't fetched twice.
        fetched: dict[ModelListType, list[str]] = {}

        def _models_for(model_type: ModelListType) -> list[str]:
            if not current_key:
                return []
            if model_type not in fetched:
                fetched[model_type] = _fetch_models_sync(current_key, model_type)
            return fetched[model_type]

        for config_key, model_type, label in model_choices:
            models = _models_for(model_type)

            # Offline / API unreachable: skip selection and leave the key unset
            # so the CLI auto-resolves at runtime. Never substitute a literal.
            if not models:
                print_error(
                    f"Could not fetch {label} models from the API. Skipping "
                    f"{label}-model selection — the CLI will auto-resolve one at "
                    "runtime. Re-run 'venice-py configure' while online to pin one."
                )
                continue

            current = config.get("defaults", {}).get(config_key)
            if current:
                console.print(f"Current {label} model: {current}")

            # Ensure the current value is a valid choice, else default to first.
            default_choice = current if current in models else models[0]
            selected = questionary.select(
                f"Select default {label} model:", choices=models, default=default_choice
            ).ask()

            if selected:
                config.setdefault("defaults", {})[config_key] = selected

    # Generation parameters
    console.print("\n[bold]Generation Parameters[/bold]")

    if questionary.confirm("Configure generation parameters?", default=False).ask():
        # Temperature
        current_temp = config.get("defaults", {}).get("temperature", 0.7)
        temp_str = questionary.text(
            f"Temperature (0.0-2.0, current: {current_temp}):",
            default=str(current_temp),
        ).ask()

        try:
            temperature = float(temp_str)
            if 0.0 <= temperature <= 2.0:
                config.setdefault("defaults", {})["temperature"] = temperature
            else:
                print_error("Temperature must be between 0.0 and 2.0")
        except ValueError:
            print_error("Invalid temperature value")

        # Max completion tokens
        current_tokens = config.get("defaults", {}).get("max_completion_tokens", 2048)
        tokens_str = questionary.text(
            f"Max completion tokens (current: {current_tokens}):",
            default=str(current_tokens),
        ).ask()

        try:
            max_completion_tokens = int(tokens_str)
            if max_completion_tokens > 0:
                config.setdefault("defaults", {})["max_completion_tokens"] = max_completion_tokens
            else:
                print_error("Max completion tokens must be positive")
        except ValueError:
            print_error("Invalid max completion tokens value")

    # Output settings
    console.print("\n[bold]Output Settings[/bold]")

    if questionary.confirm("Configure output settings?", default=False).ask():
        # Images directory
        current_dir = config.get("output", {}).get(
            "images_dir", str(Path.home() / "Pictures" / "venice")
        )
        images_dir = questionary.path("Directory for saving images:", default=current_dir).ask()

        if images_dir:
            config.setdefault("output", {})["images_dir"] = str(Path(images_dir).expanduser())

    # Features
    console.print("\n[bold]Features[/bold]")

    # Streaming
    current_stream = config.get("features", {}).get("streaming", True)
    streaming = questionary.confirm("Enable response streaming?", default=current_stream).ask()

    config.setdefault("features", {})["streaming"] = streaming

    # Save configuration
    console.print("\n[bold]Save Configuration[/bold]")

    if questionary.confirm("Save configuration?", default=True).ask():
        save_path = active_path

        if questionary.confirm(f"Save to {save_path}?", default=True).ask():
            try:
                save_config(config, save_path)
                print_success(f"Configuration saved to {save_path}")
            except Exception as e:
                print_error(f"Failed to save configuration: {e}")
                raise SystemExit(1) from e
        else:
            custom_path = questionary.path(
                "Enter configuration file path:", default=str(DEFAULT_CONFIG_PATH)
            ).ask()

            if custom_path:
                try:
                    save_config(config, Path(custom_path))
                    print_success(f"Configuration saved to {custom_path}")
                except Exception as e:
                    print_error(f"Failed to save configuration: {e}")
                    raise SystemExit(1) from e

    # Final message
    console.print("\n[bold green]Configuration complete![/bold green]")
    print_info("You can now use 'venice-py chat' or 'venice-py image generate' commands")
