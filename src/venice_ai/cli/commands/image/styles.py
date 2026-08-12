"""
Style listing command for image generation.
"""

import asyncio

import click
from rich.table import Table

from venice_ai import VeniceClient

from ...config import get_client_kwargs
from ...utils import (
    console,
    print_error,
    print_info,
    print_success,
)


@click.command(name="list-styles")
@click.pass_context
def list_styles(ctx: click.Context):
    """List available style presets for image generation"""
    asyncio.run(_list_styles_async(ctx))


async def _list_styles_async(ctx: click.Context):
    """Async implementation of style listing"""
    async with VeniceClient(**get_client_kwargs()) as client:
        try:
            print_info("Fetching available style presets...")
            response = await client.image.list_styles()

            if not response or not response.data:
                print_error("No styles available")
                return

            table = Table(
                title="Available Style Presets",
                show_header=True,
                header_style="bold magenta",
            )
            table.add_column("#", style="dim", width=4)
            table.add_column("Style Name", style="cyan")

            for idx, style_name in enumerate(response.data, 1):
                table.add_row(str(idx), style_name)

            console.print(table)
            print_success(f"\nFound {len(response.data)} available styles")
            print_info("Use with: venice image generate 'prompt' --style-preset 'Style Name'")

        except Exception as e:
            print_error(f"Failed to fetch styles: {e}")
