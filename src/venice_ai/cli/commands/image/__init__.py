"""
Image command subpackage for Venice AI CLI.

Provides image generation, editing, upscaling, background removal,
style listing, preset management, and an interactive wizard.
"""

import click

from .edit import edit_image, remove_bg
from .generate import batch_generate, generate_image
from .multi_edit import multi_edit_image
from .presets_cmd import manage_presets
from .styles import list_styles
from .upscale import upscale_image


@click.group()
def image():
    """Generate images with AI models"""
    pass


# Register all subcommands on the image group
image.add_command(generate_image)
image.add_command(batch_generate)
image.add_command(edit_image)
image.add_command(multi_edit_image)
image.add_command(remove_bg)
image.add_command(upscale_image)
image.add_command(manage_presets)
image.add_command(list_styles)

__all__ = [
    "image",
    "generate_image",
    "batch_generate",
    "edit_image",
    "multi_edit_image",
    "remove_bg",
    "upscale_image",
    "manage_presets",
    "list_styles",
]
