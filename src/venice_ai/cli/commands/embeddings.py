"""Embeddings command for Venice AI CLI — Generate text embeddings."""

import asyncio
import json
import sys

import click

from venice_ai.cli.utils.console import console


@click.command("embeddings")
@click.argument("text", required=False)
@click.option(
    "--model",
    "-m",
    default=None,
    help="Embedding model to use (defaults to the API-recommended model)",
)
@click.option(
    "--encoding-format",
    type=click.Choice(["float", "base64"]),
    default="float",
    help="Encoding format for embeddings (default: float)",
)
@click.option(
    "--dimensions",
    type=int,
    default=None,
    help="Number of dimensions for the output embeddings",
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output full JSON response with embedding vector",
)
@click.option(
    "--output",
    "-o",
    default=None,
    help="Save embeddings to file",
)
@click.pass_context
def embeddings(ctx, text, model, encoding_format, dimensions, output_json, output):
    """Generate text embeddings.

    Converts text into vector representations for semantic analysis,
    similarity search, clustering, and more.

    Examples:

      # Generate embeddings for text
      venice embeddings "The quick brown fox"

      # Use a specific model
      venice embeddings "Hello world" --model <embedding-model>

      # Output full JSON embedding vector
      venice embeddings "Some text" --json

      # Save embeddings to file
      venice embeddings "Some text" --output embeddings.json

      # Pipe text via stdin
      echo "Some text" | venice embeddings
    """
    asyncio.run(
        _embeddings_async(ctx, text, model, encoding_format, dimensions, output_json, output)
    )


async def _embeddings_async(ctx, text, model, encoding_format, dimensions, output_json, output):
    from venice_ai import VeniceClient
    from venice_ai.cli._model_defaults import resolve_default_model
    from venice_ai.cli.config import get_client_kwargs, load_config

    plain = ctx.obj.get("plain", False) if ctx.obj else False
    config = ctx.obj.get("config", load_config()) if ctx.obj else load_config()

    # Support piped input
    if not text:
        if not sys.stdin.isatty():
            text = sys.stdin.read().strip()
        if not text:
            raise click.ClickException("Text is required. Provide as argument or pipe via stdin.")

    async with VeniceClient(**get_client_kwargs()) as client:
        model = await resolve_default_model(client, config, "embedding", explicit=model)

        if not plain and not output_json:
            console.print("[bold blue]🔢 Generating embeddings...[/bold blue]")
            console.print(f"  Model: {model}")
        kwargs = {
            "model": model,
            "input": text,
            "encoding_format": encoding_format,
        }
        if dimensions is not None:
            kwargs["dimensions"] = dimensions

        response = await client.embeddings.create(**kwargs)

    embedding_data = response.data[0]
    embedding_vector = embedding_data.embedding
    dims = len(embedding_vector) if isinstance(embedding_vector, list) else None

    if output_json:
        # Build JSON output
        result = {
            "model": response.model,
            "object": response.object,
            "data": [
                {
                    "index": d.index,
                    "object": d.object,
                    "embedding": d.embedding,
                }
                for d in response.data
            ],
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        }
        json_str = json.dumps(result, indent=2)

        if output:
            with open(output, "w") as f:
                f.write(json_str)
            if not plain:
                console.print(f"[bold green]✅ Embeddings saved to:[/bold green] {output}")
            else:
                click.echo(f"Saved: {output}")
        else:
            click.echo(json_str)
        return

    # Default summary output
    if output:
        # Save embedding vector to file as JSON
        save_data = {
            "model": response.model,
            "dimensions": dims,
            "encoding_format": encoding_format,
            "embedding": embedding_vector,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        }
        with open(output, "w") as f:
            json.dump(save_data, f, indent=2)
        if not plain:
            console.print(f"\n[bold green]✅ Embeddings saved to:[/bold green] {output}")
            console.print(f"  Model: {response.model}")
            if dims is not None:
                console.print(f"  Dimensions: {dims}")
            console.print(f"  Tokens used: {response.usage.total_tokens}")
        else:
            click.echo(f"Saved: {output}")
            click.echo(f"Model: {response.model}")
            if dims is not None:
                click.echo(f"Dimensions: {dims}")
            click.echo(f"Tokens: {response.usage.total_tokens}")
    else:
        if not plain:
            console.print("\n[bold green]✅ Embeddings generated successfully[/bold green]")
            console.print(f"  Model: {response.model}")
            if dims is not None:
                console.print(f"  Dimensions: {dims}")
            if isinstance(embedding_vector, list) and len(embedding_vector) > 0:
                preview = embedding_vector[:5]
                console.print(f"  First 5 values: {preview}")
            console.print(f"  Tokens used: {response.usage.total_tokens}")
        else:
            click.echo(f"Model: {response.model}")
            if dims is not None:
                click.echo(f"Dimensions: {dims}")
            if isinstance(embedding_vector, list) and len(embedding_vector) > 0:
                preview = embedding_vector[:5]
                click.echo(f"First 5 values: {preview}")
            click.echo(f"Tokens: {response.usage.total_tokens}")
