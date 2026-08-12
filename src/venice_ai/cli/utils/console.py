"""
Console utilities for Venice AI CLI
"""

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

# Global console instance - can be reconfigured for plain mode
console = Console()

# Track plain mode state
_plain_mode = False


def enable_plain_mode() -> None:
    """Enable plain text output (no colors, no formatting, no panels)"""
    global console, _plain_mode
    _plain_mode = True
    console = Console(force_terminal=False, no_color=True, highlight=False)


def is_plain_mode() -> bool:
    """Check if plain mode is enabled"""
    return _plain_mode


def print_error(message: str) -> None:
    """Print an error message"""
    if _plain_mode:
        console.print(f"ERROR: {message}")
    else:
        console.print(f"[red]❌ Error:[/red] {message}")


def print_success(message: str) -> None:
    """Print a success message"""
    if _plain_mode:
        console.print(f"SUCCESS: {message}")
    else:
        console.print(f"[green]✅ Success:[/green] {message}")


def print_info(message: str) -> None:
    """Print an info message"""
    if _plain_mode:
        console.print(f"INFO: {message}")
    else:
        console.print(f"[blue]ℹ️  Info:[/blue] {message}")


def print_warning(message: str) -> None:
    """Print a warning message"""
    if _plain_mode:
        console.print(f"WARNING: {message}")
    else:
        console.print(f"[yellow]⚠️  Warning:[/yellow] {message}")


def print_version_info() -> None:
    """Print version information"""
    from .. import __version__

    if _plain_mode:
        console.print(f"Venice AI CLI v{__version__}")
    else:
        version_text = Text()
        version_text.append("Venice AI CLI", style="bold cyan")
        version_text.append(f" v{__version__}", style="dim")

        panel = Panel(version_text, title="[bold]Version[/bold]", border_style="cyan")
        console.print(panel)


def print_markdown(content: str) -> None:
    """Print markdown formatted content"""
    if _plain_mode:
        console.print(content)
    else:
        md = Markdown(content)
        console.print(md)


def print_code(code: str, language: str = "python") -> None:
    """Print syntax-highlighted code"""
    if _plain_mode:
        console.print(code)
    else:
        syntax = Syntax(code, language, theme="monokai", line_numbers=False)
        console.print(syntax)


def print_panel(content: str, title: str = "", style: str = "cyan") -> None:
    """Print content in a panel"""
    if _plain_mode:
        if title:
            console.print(f"=== {title} ===")
        console.print(content)
    else:
        panel = Panel(
            content,
            title=f"[bold]{title}[/bold]" if title else None,
            border_style=style,
        )
        console.print(panel)


def open_file(path: str) -> None:
    """Open a file using the system's default application.

    Uses ``os.startfile`` on Windows, ``open`` on macOS, and ``xdg-open``
    on Linux.  Failures are logged and otherwise ignored — this is a
    non-fatal convenience feature for the CLI.
    """
    import logging
    import os
    import platform

    # Cross-platform "open with default app" helper; see usage below.
    import subprocess  # nosec B404

    log = logging.getLogger(__name__)
    system = platform.system()
    try:
        if system == "Darwin":
            # Hardcoded "open" command on macOS; partial path is correct here.
            subprocess.run(["open", path], check=False)  # nosec B603 B607
        elif system == "Linux":
            # Hardcoded "xdg-open" command on Linux; partial path is correct here.
            subprocess.run(["xdg-open", path], check=False)  # nosec B603 B607
        elif system == "Windows":
            os.startfile(path)  # type: ignore[attr-defined]  # nosec B606  # Windows-only API
    except Exception as e:
        log.debug("open_file(%r) failed on %s: %s", path, system, e)
