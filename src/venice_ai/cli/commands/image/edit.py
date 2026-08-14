"""
Image editing and background removal commands.
"""

import asyncio
from pathlib import Path
from typing import Any

import click
from rich.progress import Progress, SpinnerColumn, TextColumn

from venice_ai import VeniceClient
from venice_ai.exceptions import VeniceError

from ...config import get_client_kwargs
from ...utils import (
    console,
    is_plain_mode,
    open_file,
    print_error,
    print_info,
    print_success,
)


@click.command(name="edit")
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--prompt", "-p", required=True, help="Edit instruction (what to change)")
@click.option("--model", "-m", default=None, help="Edit model to use")
@click.option("--output", "-o", default=None, help="Output file path")
@click.option(
    "--save-dir",
    default=".",
    help="Directory to save the result (default: current dir)",
)
@click.option(
    "--format",
    "img_format",
    type=click.Choice(["png", "jpeg", "webp"]),
    default=None,
    help="Output format hint for filename extension",
)
@click.option("--aspect-ratio", default=None, help="Aspect ratio for the output (e.g. 1:1, 16:9)")
@click.option(
    "--resolution",
    type=click.Choice(["1K", "2K", "4K"]),
    default=None,
    help="Resolution tier for the output image",
)
@click.option(
    "--output-format",
    type=click.Choice(["jpeg", "png", "webp"]),
    default=None,
    help="Output format for the edited image (sent to the API)",
)
@click.option(
    "--safe-mode/--no-safe-mode",
    "safe_mode",
    default=None,
    help="Blur adult content (server default on; --no-safe-mode disables)",
)
@click.option("--open", "open_image", is_flag=True, default=False, help="Open image after saving")
@click.pass_context
def edit_image(
    ctx: click.Context,
    input_file: str,
    prompt: str,
    model: str | None,
    output: str | None,
    save_dir: str,
    img_format: str | None,
    aspect_ratio: str | None,
    resolution: str | None,
    output_format: str | None,
    safe_mode: bool | None,
    open_image: bool,
) -> None:
    """Edit an image using a text prompt

    Examples:

        venice-py image edit photo.jpg --prompt "Add a rainbow to the sky"

        venice-py image edit portrait.png --prompt "Change hair color to red"
    """
    asyncio.run(
        _edit_async(
            ctx,
            input_file,
            prompt,
            model,
            output,
            save_dir,
            img_format,
            open_image,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            output_format=output_format,
            safe_mode=safe_mode,
        )
    )


async def _edit_async(
    ctx: click.Context,
    input_file: str,
    prompt: str,
    model: str | None,
    output: str | None,
    save_dir: str,
    img_format: str | None,
    open_image: bool,
    *,
    aspect_ratio: str | None = None,
    resolution: str | None = None,
    output_format: str | None = None,
    safe_mode: bool | None = None,
) -> None:
    """Async implementation of image editing"""
    plain = ctx.obj.get("plain", False) or is_plain_mode()

    input_path = Path(input_file)
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    if plain:
        click.echo(f"Editing: {input_path}")
        click.echo(f"Prompt: {prompt}")
    else:
        print_info(f"Editing: {input_path}")
        console.print(f"[cyan]Prompt:[/cyan] {prompt}")
        if model:
            console.print(f"[cyan]Model:[/cyan] {model}")

    # Build kwargs, only including provided values
    edit_kwargs: dict[str, Any] = {
        "prompt": prompt,
        "image": str(input_path),
    }
    if model is not None:
        edit_kwargs["model"] = model
    if aspect_ratio:
        edit_kwargs["aspect_ratio"] = aspect_ratio
    if resolution:
        edit_kwargs["resolution"] = resolution
    if output_format:
        edit_kwargs["output_format"] = output_format
    if safe_mode is not None:
        edit_kwargs["safe_mode"] = safe_mode

    try:
        async with VeniceClient(**get_client_kwargs()) as client:
            if plain:
                click.echo("Sending to API...")
                try:
                    result_bytes = await client.image.edit(**edit_kwargs)
                    click.echo("Edit complete.")
                except Exception as e:
                    click.echo(f"Edit failed: {e}")
                    raise
            else:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    task = progress.add_task("Editing image...", total=None)
                    try:
                        result_bytes = await client.image.edit(**edit_kwargs)
                        progress.update(task, description="[green]Edit complete!")
                    except Exception as e:
                        progress.update(task, description=f"[red]Failed: {e}")
                        raise

            # Determine output extension. The API-side --output-format wins,
            # then the filename-hint --format, then the input file's suffix.
            if output_format:
                ext = f".{output_format}"
            elif img_format:
                ext = f".{img_format}"
            else:
                ext = input_path.suffix or ".png"

            if output:
                out_file = Path(output)
            else:
                from datetime import datetime

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_file = save_path / f"edited_{timestamp}{ext}"

            out_file.parent.mkdir(parents=True, exist_ok=True)
            with open(out_file, "wb") as f:
                f.write(result_bytes)

            if plain:
                click.echo(f"Saved: {out_file}")
                click.echo(f"  Size: {len(result_bytes):,} bytes")
            else:
                print_success(f"Saved: {out_file}")
                console.print(f"  Size: {len(result_bytes):,} bytes")

            if open_image:
                open_file(str(out_file))

    except VeniceError as e:
        print_error(f"Venice API error: {e}")
        raise SystemExit(1) from e
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        raise SystemExit(1) from e


@click.command(name="remove-bg")
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--output", "-o", default=None, help="Output file path")
@click.option(
    "--save-dir",
    default=".",
    help="Directory to save the result (default: current dir)",
)
@click.option(
    "--format",
    "img_format",
    type=click.Choice(["png", "jpeg", "webp"]),
    default="png",
    help="Output format (png recommended for transparency)",
)
@click.option("--open", "open_image", is_flag=True, default=False, help="Open image after saving")
@click.pass_context
def remove_bg(
    ctx: click.Context,
    input_file: str,
    output: str | None,
    save_dir: str,
    img_format: str,
    open_image: bool,
) -> None:
    """Remove the background from an image

    Returns a PNG with transparent background.

    Examples:

        venice-py image remove-bg photo.jpg

        venice-py image remove-bg product.png --save-dir ./cutouts --open
    """
    asyncio.run(_remove_bg_async(ctx, input_file, output, save_dir, img_format, open_image))


async def _remove_bg_async(
    ctx: click.Context,
    input_file: str,
    output: str | None,
    save_dir: str,
    img_format: str,
    open_image: bool,
) -> None:
    """Async implementation of background removal"""
    plain = ctx.obj.get("plain", False) or is_plain_mode()

    input_path = Path(input_file)
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    if plain:
        click.echo(f"Removing background: {input_path}")
    else:
        print_info(f"Removing background: {input_path}")

    try:
        async with VeniceClient(**get_client_kwargs()) as client:
            if plain:
                click.echo("Sending to API...")
                try:
                    result_bytes = await client.image.background_remove(image=str(input_path))
                    click.echo("Background removal complete.")
                except Exception as e:
                    click.echo(f"Background removal failed: {e}")
                    raise
            else:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    task = progress.add_task("Removing background...", total=None)
                    try:
                        result_bytes = await client.image.background_remove(image=str(input_path))
                        progress.update(task, description="[green]Background removal complete!")
                    except Exception as e:
                        progress.update(task, description=f"[red]Failed: {e}")
                        raise

            ext = f".{img_format}"

            if output:
                out_file = Path(output)
            else:
                from datetime import datetime

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_file = save_path / f"no_bg_{timestamp}{ext}"

            out_file.parent.mkdir(parents=True, exist_ok=True)
            with open(out_file, "wb") as f:
                f.write(result_bytes)

            if plain:
                click.echo(f"Saved: {out_file}")
                click.echo(f"  Size: {len(result_bytes):,} bytes")
            else:
                print_success(f"Saved: {out_file}")
                console.print(f"  Size: {len(result_bytes):,} bytes")
                console.print("[dim]Tip: PNG format preserves transparency[/dim]")

            if open_image:
                open_file(str(out_file))

    except VeniceError as e:
        print_error(f"Venice API error: {e}")
        raise SystemExit(1) from e
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        raise SystemExit(1) from e
