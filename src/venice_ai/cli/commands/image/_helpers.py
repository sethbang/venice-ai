"""
Shared helpers for the image command subpackage.
"""

import re

import click

from ...presets import get_builtin_presets, load_preset


def _load_preset_config(preset_name: str) -> dict | None:
    """Load preset configuration from built-in or custom presets"""
    # Check built-in presets first
    builtin_presets = get_builtin_presets()
    if preset_name in builtin_presets:
        return builtin_presets[preset_name]

    # Try to load custom preset
    return load_preset(preset_name)


def validate_size(ctx: click.Context, _param: click.Parameter, value: str | None) -> str | None:
    """Validate WxH size format."""
    if value is None:
        return None

    match = re.match(r"^(\d+)x(\d+)$", value)
    if not match:
        raise click.BadParameter(f"Size must be in WxH format (e.g., 1024x1024), got: {value}")
    w, h = int(match.group(1)), int(match.group(2))
    if w < 64 or h < 64:
        raise click.BadParameter(f"Minimum dimension is 64px, got: {w}x{h}")
    if w > 4096 or h > 4096:
        raise click.BadParameter(f"Maximum dimension is 4096px, got: {w}x{h}")
    return value
