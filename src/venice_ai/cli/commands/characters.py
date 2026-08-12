"""Characters commands for Venice AI CLI — Discover and view AI characters."""

import asyncio
import json

import click

from venice_ai.cli.utils.console import console


@click.group()
def characters():
    """Discover and view Venice AI characters.

    Browse pre-configured AI personalities and specialized assistants.
    """
    pass


_SORT_BY_CHOICES = (
    "featured",
    "highestRating",
    "highlyRated",
    "highlyRatedAndRecent",
    "imports",
    "mostRecent",
    "ratingCount",
)


@characters.command("list")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON for scripting")
@click.option(
    "--search",
    default=None,
    help="Free-text search across name, description, tags, and hashtags (server-side)",
)
@click.option(
    "--sort-by",
    "sort_by",
    type=click.Choice(_SORT_BY_CHOICES),
    default=None,
    help="Sort results using a supported character-discovery mode",
)
@click.option(
    "--sort-order",
    "sort_order",
    type=click.Choice(["asc", "desc"]),
    default=None,
    help="Sort order applied to the selected --sort-by mode",
)
@click.option("--tags", "tags", multiple=True, help="Filter by tag name (repeatable)")
@click.option(
    "--categories", "categories", multiple=True, help="Filter by category name (repeatable)"
)
@click.option("--limit", "limit", type=int, default=None, help="Number of characters to return")
@click.option("--offset", "offset", type=int, default=None, help="Number of characters to skip")
@click.option("--adult/--no-adult", "is_adult", default=None, help="Filter by adult content flag")
@click.option(
    "--pro/--no-pro", "is_pro", default=None, help="Filter to characters using pro models"
)
@click.option(
    "--web-enabled/--no-web-enabled",
    "is_web_enabled",
    default=None,
    help="Filter to web-enabled characters",
)
@click.option("--model-id", "model_id", default=None, help="Filter by model ID")
@click.pass_context
def list_characters(
    ctx,
    output_json,
    search,
    sort_by=None,
    sort_order=None,
    tags=(),
    categories=(),
    limit=None,
    offset=None,
    is_adult=None,
    is_pro=None,
    is_web_enabled=None,
    model_id=None,
):
    """List available AI characters.

    Examples:
        venice characters list
        venice characters list --search coding
        venice characters list --sort-by highlyRated --limit 20
        venice characters list --tags philosophy --web-enabled
        venice characters list --categories education
        venice characters list --json
    """
    asyncio.run(
        _list_characters_async(
            ctx,
            output_json,
            search,
            sort_by=sort_by,
            sort_order=sort_order,
            tags=tags,
            categories=categories,
            limit=limit,
            offset=offset,
            is_adult=is_adult,
            is_pro=is_pro,
            is_web_enabled=is_web_enabled,
            model_id=model_id,
        )
    )


async def _list_characters_async(
    ctx,
    output_json,
    search,
    sort_by=None,
    sort_order=None,
    tags=(),
    categories=(),
    limit=None,
    offset=None,
    is_adult=None,
    is_pro=None,
    is_web_enabled=None,
    model_id=None,
):
    from venice_ai import VeniceClient
    from venice_ai.cli.config import get_client_kwargs
    from venice_ai.cli.utils.console import enable_plain_mode, is_plain_mode

    plain = ctx.obj.get("plain", False) if ctx.obj else False
    if plain and not is_plain_mode():
        enable_plain_mode()

    # Build kwargs from only the filters the user actually set. The server
    # searches name/description/tags/hashtag across the full catalog, so we
    # delegate filtering rather than paging-and-filtering locally.
    list_kwargs = {}
    if search is not None:
        list_kwargs["search"] = search
    if sort_by is not None:
        list_kwargs["sort_by"] = sort_by
    if sort_order is not None:
        list_kwargs["sort_order"] = sort_order
    if tags:
        list_kwargs["tags"] = list(tags)
    if categories:
        list_kwargs["categories"] = list(categories)
    if limit is not None:
        list_kwargs["limit"] = limit
    if offset is not None:
        list_kwargs["offset"] = offset
    if is_adult is not None:
        list_kwargs["is_adult"] = is_adult
    if is_pro is not None:
        list_kwargs["is_pro"] = is_pro
    if is_web_enabled is not None:
        list_kwargs["is_web_enabled"] = is_web_enabled
    if model_id is not None:
        list_kwargs["model_id"] = [model_id]

    async with VeniceClient(**get_client_kwargs()) as client:
        response = await client.characters.list(**list_kwargs)

    chars = response.data

    if output_json:
        data = [
            {
                "slug": c.slug,
                "name": c.name,
                "description": c.description,
                "tags": c.tags,
                "modelId": c.modelId,
                "adult": c.adult,
            }
            for c in chars
        ]
        click.echo(json.dumps(data, indent=2))
        return

    from venice_ai.cli.utils.output import OutputManager

    if not chars:
        msg = f"No characters matching '{search}'." if search else "No characters found."
        OutputManager.warning(msg)
        return

    rows = [[c.slug, c.name, (c.description or "")[:50]] for c in chars]
    OutputManager.table(
        headers=["Slug", "Name", "Description"],
        rows=rows,
        title="Venice AI Characters",
        col_styles=["cyan", "bold white", "dim"],
    )


@characters.command("info")
@click.argument("slug")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON for scripting")
@click.pass_context
def info_character(ctx, slug, output_json):
    """Show detailed information about a character.

    Examples:
        venice characters info alan-watts
        venice characters info alan-watts --json
    """
    asyncio.run(_info_character_async(ctx, slug, output_json))


async def _info_character_async(ctx, slug, output_json):
    from venice_ai import VeniceClient
    from venice_ai.cli.config import get_client_kwargs

    plain = ctx.obj.get("plain", False) if ctx.obj else False

    async with VeniceClient(**get_client_kwargs()) as client:
        response = await client.characters.get(slug)

    char = response.data

    if output_json:
        stats = char.stats
        data = {
            "id": getattr(char, "id", None),
            "slug": char.slug,
            "name": char.name,
            "author": getattr(char, "author", None),
            "description": char.description,
            "tags": char.tags,
            "modelId": char.modelId,
            "adult": char.adult,
            "webEnabled": char.webEnabled,
            "featured": getattr(char, "featured", None),
            "isOwner": getattr(char, "isOwner", None),
            "shareUrl": char.shareUrl,
            "photoUrl": char.photoUrl,
            "createdAt": char.createdAt,
            "updatedAt": char.updatedAt,
            "stats": {
                "imports": stats.imports,
                "averageRating": getattr(stats, "averageRating", None),
                "ratingCount": getattr(stats, "ratingCount", None),
                "ratingSum": getattr(stats, "ratingSum", None),
                "userRating": getattr(stats, "userRating", None),
            },
        }
        click.echo(json.dumps(data, indent=2))
        return

    if plain:
        click.echo(f"Character: {char.name}")
        click.echo(f"Slug:      {char.slug}")
        click.echo(f"Model:     {char.modelId}")
        if char.description:
            click.echo(f"\nDescription:\n  {char.description}")
        click.echo(f"\nTags:      {', '.join(char.tags) if char.tags else 'None'}")
        click.echo(f"Adult:     {char.adult}")
        click.echo(f"Web:       {char.webEnabled}")
        click.echo(f"Imports:   {char.stats.imports}")
        click.echo(f"Created:   {char.createdAt}")
        click.echo(f"Updated:   {char.updatedAt}")
        if char.shareUrl:
            click.echo(f"Share URL: {char.shareUrl}")
    else:
        from rich.panel import Panel
        from rich.text import Text

        lines = Text()
        lines.append("Slug:    ", style="bold")
        lines.append(f"{char.slug}\n", style="cyan")
        lines.append("Model:   ", style="bold")
        lines.append(f"{char.modelId}\n", style="green")

        if char.description:
            lines.append("\n")
            lines.append(f"{char.description}\n", style="dim")

        lines.append("\n")
        lines.append("Tags:    ", style="bold")
        tags_str = ", ".join(char.tags) if char.tags else "None"
        lines.append(f"{tags_str}\n", style="yellow")

        lines.append("Adult:   ", style="bold")
        lines.append(f"{'Yes' if char.adult else 'No'}\n")
        lines.append("Web:     ", style="bold")
        lines.append(f"{'Enabled' if char.webEnabled else 'Disabled'}\n")
        lines.append("Imports: ", style="bold")
        lines.append(f"{char.stats.imports}\n")
        lines.append("Created: ", style="bold")
        lines.append(f"{char.createdAt}\n")
        lines.append("Updated: ", style="bold")
        lines.append(f"{char.updatedAt}\n")

        if char.shareUrl:
            lines.append("\n")
            lines.append("Share:   ", style="bold")
            lines.append(f"{char.shareUrl}\n", style="blue underline")

        panel = Panel(lines, title=f"🎭 {char.name}", border_style="blue")
        console.print(panel)


@characters.command("reviews")
@click.argument("slug")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON for scripting")
@click.option("--page", "page", type=int, default=None, help="1-indexed page number")
@click.option("--page-size", "page_size", type=int, default=None, help="Reviews per page")
@click.pass_context
def reviews_character(ctx, slug, output_json, page, page_size):
    """Show public reviews for a character.

    Examples:
        venice characters reviews alan-watts
        venice characters reviews alan-watts --page 2 --page-size 50
        venice characters reviews alan-watts --json
    """
    asyncio.run(_reviews_character_async(ctx, slug, output_json, page, page_size))


async def _reviews_character_async(ctx, slug, output_json, page, page_size):
    from venice_ai import VeniceClient
    from venice_ai.cli.config import get_client_kwargs
    from venice_ai.cli.utils.console import enable_plain_mode, is_plain_mode

    plain = ctx.obj.get("plain", False) if ctx.obj else False
    if plain and not is_plain_mode():
        enable_plain_mode()

    async with VeniceClient(**get_client_kwargs()) as client:
        response = await client.characters.reviews(slug, page=page, page_size=page_size)

    reviews = response.data
    summary = response.summary

    if output_json:
        data = {
            "slug": slug,
            "summary": {
                "averageRating": summary.averageRating,
                "totalReviews": summary.totalReviews,
            },
            "reviews": [
                {
                    "rating": r.rating,
                    "createdAt": r.createdAt,
                    "message": r.message,
                }
                for r in reviews
            ],
        }
        click.echo(json.dumps(data, indent=2))
        return

    from venice_ai.cli.utils.output import OutputManager

    OutputManager.info(
        f"{summary.totalReviews} review(s) — average rating {summary.averageRating}/5"
    )

    if not reviews:
        OutputManager.warning(f"No reviews found for '{slug}'.")
        return

    if plain:
        for r in reviews:
            click.echo(f"\n{r.rating}/5  ({r.createdAt})")
            click.echo(f"  {r.message or '(no message)'}")
    else:
        rows = [[f"{r.rating}/5", r.createdAt, (r.message or "")[:60]] for r in reviews]
        OutputManager.table(
            headers=["Rating", "Created", "Message"],
            rows=rows,
            title=f"Reviews for {slug}",
            col_styles=["cyan", "dim", "white"],
        )
