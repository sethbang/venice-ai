"""
Preset management for Venice AI CLI
Handles saving, loading, and managing image generation presets
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from .utils import print_error, print_success

# Default preset directory
DEFAULT_PRESETS_DIR = Path.home() / ".venice" / "presets"

_SAFE_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


def _safe_preset_name(name: str) -> str:
    """Validate a preset name, preventing path traversal.

    Args:
        name: The preset name to validate.

    Returns:
        The validated name.

    Raises:
        ValueError: If the name contains characters outside ``[a-zA-Z0-9_-]``.
    """
    if not _SAFE_NAME_RE.match(name):
        raise ValueError(
            f"Invalid preset name: {name!r}. "
            "Only alphanumeric characters, hyphens, and underscores are allowed."
        )
    return name


def get_presets_dir() -> Path:
    """Get the presets directory, creating it if needed"""
    presets_dir = DEFAULT_PRESETS_DIR
    presets_dir.mkdir(parents=True, exist_ok=True)
    return presets_dir


def save_preset(name: str, config: dict[str, Any]) -> bool:
    """
    Save an image generation preset

    Args:
        name: Name of the preset
        config: Preset configuration dictionary

    Returns:
        True if successful, False otherwise
    """
    try:
        name = _safe_preset_name(name)
        presets_dir = get_presets_dir()
        preset_file = presets_dir / f"{name}.json"

        # Add metadata
        preset_data = {
            "name": name,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "config": config,
        }

        with open(preset_file, "w") as f:
            json.dump(preset_data, f, indent=2)

        print_success(f"Preset '{name}' saved successfully")
        return True

    except Exception as e:
        print_error(f"Failed to save preset: {e}")
        return False


def load_preset(name: str) -> dict[str, Any] | None:
    """
    Load a preset by name

    Args:
        name: Name of the preset to load

    Returns:
        Preset configuration dictionary or None if not found
    """
    try:
        name = _safe_preset_name(name)
        presets_dir = get_presets_dir()
        preset_file = presets_dir / f"{name}.json"

        if not preset_file.exists():
            print_error(f"Preset '{name}' not found")
            return None

        with open(preset_file) as f:
            preset_data = json.load(f)

        # Update last used timestamp
        preset_data["updated_at"] = datetime.now().isoformat()
        with open(preset_file, "w") as f:
            json.dump(preset_data, f, indent=2)

        config: dict[str, Any] | None = preset_data.get("config")
        return config

    except Exception as e:
        print_error(f"Failed to load preset: {e}")
        return None


def list_presets() -> list[dict[str, Any]]:
    """
    List all available presets

    Returns:
        List of preset metadata dictionaries
    """
    try:
        presets_dir = get_presets_dir()
        presets = []

        for preset_file in presets_dir.glob("*.json"):
            try:
                with open(preset_file) as f:
                    preset_data = json.load(f)
                    presets.append(
                        {
                            "name": preset_data.get("name", preset_file.stem),
                            "created_at": preset_data.get("created_at", "Unknown"),
                            "updated_at": preset_data.get("updated_at", "Unknown"),
                            "config": preset_data.get("config", {}),
                        }
                    )
            except Exception as e:
                # Skip malformed preset files but record the failure.
                import logging

                logging.getLogger(__name__).debug("skipping preset %s: %s", preset_file, e)
                continue

        return sorted(presets, key=lambda p: p.get("updated_at", ""), reverse=True)

    except Exception as e:
        print_error(f"Failed to list presets: {e}")
        return []


def delete_preset(name: str) -> bool:
    """
    Delete a preset by name

    Args:
        name: Name of the preset to delete

    Returns:
        True if successful, False otherwise
    """
    try:
        name = _safe_preset_name(name)
        presets_dir = get_presets_dir()
        preset_file = presets_dir / f"{name}.json"

        if not preset_file.exists():
            print_error(f"Preset '{name}' not found")
            return False

        preset_file.unlink()
        print_success(f"Preset '{name}' deleted successfully")
        return True

    except Exception as e:
        print_error(f"Failed to delete preset: {e}")
        return False


def get_builtin_presets() -> dict[str, dict[str, Any]]:
    """
    Get built-in presets for common use cases

    Returns:
        Dictionary of preset name to configuration
    """
    return {
        "photorealistic": {
            "steps": 30,
            "cfg_scale": 7.5,
            "format": "png",
            "safe_mode": True,
            "embed_exif": True,
            "description": "High-quality photorealistic images",
        },
        "artistic": {
            "steps": 25,
            "cfg_scale": 9.0,
            "format": "webp",
            "safe_mode": True,
            "description": "Artistic and creative styles",
        },
        "quick": {
            "steps": 15,
            "cfg_scale": 7.0,
            "format": "webp",
            "safe_mode": True,
            "description": "Fast generation with good quality",
        },
        "high-quality": {
            "steps": 50,
            "cfg_scale": 8.0,
            "format": "png",
            "safe_mode": True,
            "embed_exif": True,
            "description": "Maximum quality (slower)",
        },
        "creative": {
            "steps": 20,
            "cfg_scale": 5.0,
            "format": "webp",
            "safe_mode": False,
            "description": "More creative freedom, less prompt adherence",
        },
    }


def apply_preset_to_config(config: dict[str, Any], preset_config: dict[str, Any]) -> dict[str, Any]:
    """
    Apply preset configuration to a base config

    Args:
        config: Base configuration
        preset_config: Preset configuration to apply

    Returns:
        Merged configuration
    """
    # Create a copy to avoid modifying original
    result = config.copy()

    # Apply preset values, but don't override explicitly set values
    for key, value in preset_config.items():
        if key not in result or result[key] is None:
            result[key] = value

    return result
