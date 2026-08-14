"""
Preset management command for image generation.

Named presets_cmd to avoid conflict with venice_ai.cli.presets module.
"""

import asyncio

import click
import questionary
from rich.table import Table

from ...presets import (
    delete_preset,
    get_builtin_presets,
    list_presets,
    save_preset,
)
from ...utils import (
    console,
    print_info,
)


@click.command(name="presets")
@click.pass_context
def manage_presets(ctx: click.Context):
    """Manage image generation presets"""
    asyncio.run(_manage_presets_async(ctx))


async def _manage_presets_async(ctx: click.Context):
    """Interactive preset management"""

    while True:
        console.print("\n[bold cyan]🎨 Preset Management[/bold cyan]")

        action = await asyncio.to_thread(
            lambda: questionary.select(
                "What would you like to do?",
                choices=[
                    "List all presets",
                    "View built-in presets",
                    "Save current config as preset",
                    "Delete a preset",
                    "Exit",
                ],
                qmark="🔧",
            ).ask()
        )

        if action == "List all presets":
            presets = list_presets()
            if not presets:
                print_info("No custom presets found")
            else:
                table = Table(title="Custom Presets", show_header=True)
                table.add_column("Name", style="cyan")
                table.add_column("Created", style="dim")
                table.add_column("Last Used", style="dim")

                for preset in presets:
                    table.add_row(
                        preset["name"],
                        preset["created_at"][:10]
                        if preset["created_at"] != "Unknown"
                        else "Unknown",
                        preset["updated_at"][:10]
                        if preset["updated_at"] != "Unknown"
                        else "Unknown",
                    )

                console.print(table)

        elif action == "View built-in presets":
            builtin = get_builtin_presets()
            table = Table(title="Built-in Presets", show_header=True)
            table.add_column("Name", style="cyan")
            table.add_column("Description", style="white")
            table.add_column("Steps", style="yellow")
            table.add_column("CFG", style="yellow")

            for name, config in builtin.items():
                table.add_row(
                    name,
                    config.get("description", ""),
                    str(config.get("steps", "-")),
                    str(config.get("cfg_scale", "-")),
                )

            console.print(table)
            print_info("\nUse with: venice-py image generate 'prompt' --preset <name>")

        elif action == "Save current config as preset":
            preset_name = await asyncio.to_thread(
                lambda: questionary.text("Enter preset name:", qmark="💾").ask()
            )

            if preset_name:
                # Build config interactively
                console.print(
                    "\n[yellow]Configure preset parameters (leave empty to skip)[/yellow]"
                )

                config = {}

                steps_str = await asyncio.to_thread(
                    lambda: questionary.text("Steps (1-50):", default="").ask()
                )
                if steps_str:
                    config["steps"] = int(steps_str)

                cfg_str = await asyncio.to_thread(
                    lambda: questionary.text("CFG Scale (0-20):", default="").ask()
                )
                if cfg_str:
                    config["cfg_scale"] = float(cfg_str)

                format_choice = await asyncio.to_thread(
                    lambda: questionary.select(
                        "Format:",
                        choices=["Skip", "webp", "png", "jpeg"],
                        default="Skip",
                    ).ask()
                )
                if format_choice != "Skip":
                    config["format"] = format_choice

                safe_mode = await asyncio.to_thread(
                    lambda: questionary.confirm("Safe mode?", default=True).ask()
                )
                config["safe_mode"] = safe_mode

                description = await asyncio.to_thread(
                    lambda: questionary.text("Description (optional):", default="").ask()
                )
                if description:
                    config["description"] = description

                save_preset(preset_name, config)

        elif action == "Delete a preset":
            presets = list_presets()
            if not presets:
                print_info("No custom presets to delete")
            else:
                preset_names = [p["name"] for p in presets] + ["Cancel"]

                def _ask_preset_to_delete(choices: list[str] = preset_names) -> str:
                    return str(
                        questionary.select(
                            "Select preset to delete:", choices=choices, qmark="🗑️"
                        ).ask()
                    )

                preset_to_delete = await asyncio.to_thread(_ask_preset_to_delete)

                if preset_to_delete and preset_to_delete != "Cancel":

                    def _ask_confirm_delete(preset: str = preset_to_delete) -> bool:
                        return bool(
                            questionary.confirm(f"Delete preset '{preset}'?", default=False).ask()
                        )

                    confirm = await asyncio.to_thread(_ask_confirm_delete)
                    if confirm:
                        delete_preset(preset_to_delete)

        else:  # Exit
            break
