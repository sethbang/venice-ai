#!/usr/bin/env python3
"""
Venice AI Test Runner

A modern, user-friendly test runner for the Venice AI project.
Replaces the Bash script (run_tests.sh) with a Python-based solution
that offers both command-line and interactive interfaces.
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple

import questionary
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table

# Initialize Rich console
console = Console()

# Test file categories and default log file
E2E_TESTS_DIR = Path("e2e_tests")
UNIT_TESTS_DIR = Path("tests")
DEFAULT_LOG_DIR = Path("logs/tests")
DEFAULT_LOG_FILE = "pytest_output"
DEFAULT_COVERAGE_SOURCE = "src/venice_ai"

# Colors and styling
class Style:
    """Defines style constants for rich text output."""
    SUCCESS = "bold green"
    ERROR = "bold red"
    WARNING = "yellow"
    INFO = "cyan"
    HIGHLIGHT = "bold magenta"
    TITLE = "bold blue"
    COMMAND = "italic bright_black"


def print_title():
    """Print the title banner for the test runner."""
    title = Text("Venice AI Test Runner", style=Style.TITLE)
    console.print(Panel(title, expand=False))
    console.print("")


def get_api_key() -> str:
    """
    Get the Venice API key from environment variables or prompt the user.
    
    Returns:
        str: The API key to use for tests
    """
    api_key = os.environ.get("VENICE_API_KEY")
    
    if not api_key:
        console.print("VENICE_API_KEY environment variable not found.", style=Style.WARNING)
        console.print("You can set this permanently in your environment or enter it now.")
        
        api_key = questionary.password(
            "Enter your Venice API key (input will be hidden):",
            validate=lambda text: len(text) > 0 or "API key cannot be empty"
        ).ask()
        
        if not api_key:
            console.print("No API key provided. Exiting.", style=Style.ERROR)
            sys.exit(1)
    
    return api_key


def get_test_files() -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Discover available test files in the project.
    
    Returns:
        tuple: Lists of (e2e_test_files, benchmark_files, unit_test_files)
    """
    # E2E test files
    e2e_files = sorted(list(E2E_TESTS_DIR.glob("test_*.py")))
    
    # Benchmark files
    benchmark_files = sorted(list(E2E_TESTS_DIR.glob("benchmark_*.py")))
    
    # Unit test files
    unit_files = sorted(list(UNIT_TESTS_DIR.glob("test_*.py")))
    unit_files += sorted(list((UNIT_TESTS_DIR / "resources").glob("test_*.py")))
    
    return e2e_files, benchmark_files, unit_files


def ensure_test_environment():
    """Ensure necessary directories and test data exist."""
    # Create necessary directories
    (E2E_TESTS_DIR / "output").mkdir(parents=True, exist_ok=True)
    (E2E_TESTS_DIR / "data").mkdir(parents=True, exist_ok=True)
    
    # Create sample image if it doesn't exist
    sample_image = E2E_TESTS_DIR / "data" / "sample_image.png"
    if not sample_image.exists():
        console.print("Creating sample image for tests...", style=Style.INFO)
        
        try:
            # Try to use ImageMagick if available
            result = subprocess.run(
                ["which", "convert"], 
                capture_output=True, 
                text=True, 
                check=False
            )
            
            if result.returncode == 0:
                subprocess.run([
                    "convert", "-size", "100x100", "xc:white", 
                    "-fill", "black", "-draw", "text 10,50 'Test'",
                    str(sample_image)
                ], check=True)
                console.print("Sample image created.", style=Style.SUCCESS)
            else:
                console.print(
                    "WARNING: ImageMagick not found. Please manually create a sample image at "
                    f"{sample_image}", 
                    style=Style.WARNING
                )
        except Exception as e:
            console.print(
                f"Error creating sample image: {e}. Please create it manually.", 
                style=Style.ERROR
            )


def print_models_info():
    """Print information about models being used for tests."""
    table = Table(title="Test Models")
    table.add_column("Type", style="cyan")
    table.add_column("Model", style="green")
    
    table.add_row("Text model", "qwen-2.5-qwq-32b")
    table.add_row("Image model", "venice-sd35")
    table.add_row("Embedding model", "llama-3.2-3b")
    table.add_row("TTS model", "tts-kokoro with am_michael voice")
    
    console.print(table)
    console.print("")


def interactive_test_selection() -> dict:
    """
    Show interactive menu for test configuration.
    
    Returns:
        dict: Selected test options and paths
    """
    console.print("Test Configuration", style=Style.TITLE)
    console.print("")
    
    # Get options via questionary
    options = {}
    
    # Toggle options with checkboxes
    selected_options = questionary.checkbox(
        "Select test options:",
        choices=[
            questionary.Choice("Verbose output", checked=False),
            questionary.Choice("Generate coverage report", checked=False),
            questionary.Choice("Save output to log file", checked=False),
            questionary.Choice("Generate HTML coverage report", checked=False),
            questionary.Choice("Stop on first failure", checked=False),
            questionary.Choice("Run tests in parallel", checked=False),
        ],
    ).ask()
    
    # Map selected options to variables
    options["verbose"] = "Verbose output" in selected_options
    options["coverage"] = "Generate coverage report" in selected_options
    options["log"] = "Save output to log file" in selected_options
    options["html_coverage"] = "Generate HTML coverage report" in selected_options
    options["fail_fast"] = "Stop on first failure" in selected_options
    options["parallel"] = "Run tests in parallel" in selected_options
    
    # If HTML coverage is selected, ensure coverage is also selected
    if options["html_coverage"] and not options["coverage"]:
        options["coverage"] = True
        console.print(
            "Note: Enabled coverage report because HTML coverage was selected.",
            style=Style.INFO
        )
    
    # Get log file name if logging is enabled
    if options["log"]:
        log_filename_base = questionary.text(
            "Log file name base (timestamp will be appended):",
            default=DEFAULT_LOG_FILE
        ).ask()
        options["log_filename_base"] = log_filename_base
    
    # Get coverage source if coverage is enabled
    if options["coverage"]:
        cov_source = questionary.text(
            "Coverage source directory:",
            default=DEFAULT_COVERAGE_SOURCE
        ).ask()
        options["cov_source"] = cov_source
    
    # Get test files
    e2e_files, benchmark_files, unit_files = get_test_files()
    
    # Group selection choices
    test_choices = [
        questionary.Choice("Run ALL tests and benchmarks", value="all"),
        questionary.Choice("Run ALL E2E test files", value="e2e"),
        questionary.Choice("Run ALL benchmark files", value="benchmark"),
        questionary.Choice("Run ALL unit test files", value="unit"),
        questionary.Choice("Select specific test files", value="specific"),
    ]
    
    group_selection = questionary.select(
        "Which tests would you like to run?",
        choices=test_choices,
    ).ask()
    
    options["group"] = group_selection
    
    # If specific test files were selected, show another selection menu
    if group_selection == "specific":
        # Create a list of all test files with prefixes to distinguish types
        all_files = []
        
        # Add e2e test files
        for file in e2e_files:
            name = file.name
            all_files.append(questionary.Choice(f"E2E: {name}", value=str(file)))
        
        # Add benchmark files
        for file in benchmark_files:
            name = file.name
            all_files.append(questionary.Choice(f"Benchmark: {name}", value=str(file)))
        
        # Add unit test files
        for file in unit_files:
            name = file.name
            if "resources" in str(file):
                all_files.append(questionary.Choice(f"Unit (resources): {name}", value=str(file)))
            else:
                all_files.append(questionary.Choice(f"Unit: {name}", value=str(file)))
        
        selected_files = questionary.checkbox(
            "Select test files to run:",
            choices=all_files,
        ).ask()
        
        options["specific_files"] = selected_files
    
    return options


def build_pytest_command(options: dict) -> List[str]:
    """
    Build the pytest command based on selected options.
    
    Args:
        options (dict): Test options selected by the user
    
    Returns:
        list: The pytest command as a list of arguments
    """
    # Start with the base command
    cmd = ["poetry", "run", "pytest"]
    
    # Add verbose flag
    if options.get("verbose", False):
        cmd.append("-v")
    
    # Add fail-fast flag
    if options.get("fail_fast", False):
        cmd.append("-x")
    
    # Add parallel execution flag if requested
    if options.get("parallel", False):
        # First check if pytest-xdist is installed
        try:
            import pytest_xdist  # type: ignore[import-not-found]
            cmd.extend(["-n", "auto"])
        except ImportError:
            console.print(
                "Warning: pytest-xdist not installed. Cannot run tests in parallel.",
                style=Style.WARNING
            )
            console.print("To enable parallel testing, install: poetry add --dev pytest-xdist")
    
    # Handle coverage: disable pytest-cov plugin when coverage.py will handle it
    if options.get("coverage", False):
        # Disable pytest-cov plugin to prevent interference with coverage.py
        cmd.extend(["-p", "no:cov"])
    
    # Add test paths based on group selection
    e2e_files, benchmark_files, unit_files = get_test_files()
    
    # Handle different group selections
    group = options.get("group")
    if group == "all":
        # Add all test files
        for file in e2e_files + benchmark_files + unit_files:
            cmd.append(str(file))
    elif group == "e2e":
        # Add only e2e test files
        for file in e2e_files:
            cmd.append(str(file))
    elif group == "benchmark":
        # Add only benchmark files
        for file in benchmark_files:
            cmd.append(str(file))
    elif group == "unit":
        # Add only unit test files
        for file in unit_files:
            cmd.append(str(file))
    elif group == "specific" and "specific_files" in options:
        # Add specifically selected files
        for file in options["specific_files"]:
            cmd.append(file)
    
    return cmd


def prepare_log_file(options: dict) -> Optional[Path]:
    """
    Prepare log file path if logging is enabled.
    
    Args:
        options (dict): Test options
        
    Returns:
        Path or None: The log file path if logging is enabled, None otherwise
    """
    if not options.get("log", False):
        return None
    
    # Create logs directory
    DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get base filename
    log_filename_base = options.get("log_filename_base", DEFAULT_LOG_FILE)
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = DEFAULT_LOG_DIR / f"{log_filename_base}_{timestamp}.log"
    
    return log_file


def run_tests(options: dict) -> int:
    """
    Run tests with the specified options.
    
    Args:
        options (dict): Test options
        
    Returns:
        int: The exit code from pytest
    """
    # Get API key
    api_key = get_api_key()
    
    # Set up environment
    ensure_test_environment()
    
    # Display models info
    print_models_info()
    
    # Build the pytest command
    pytest_cmd = build_pytest_command(options)
    
    # Prepare environment variables
    env = os.environ.copy()
    env["VENICE_API_KEY"] = api_key
    
    # Build the final command - wrap with coverage run if coverage is enabled
    if options.get("coverage", False):
        # Use coverage.py to run pytest (similar to old run_tests.sh)
        cov_source = options.get("cov_source", DEFAULT_COVERAGE_SOURCE)
        cmd = ["poetry", "run", "python", "-m", "coverage", "run", f"--source={cov_source}", "-m", "pytest"]
        # Add the pytest arguments after removing "poetry run pytest" prefix
        cmd.extend(pytest_cmd[3:])  # Skip "poetry run pytest" from pytest_cmd
    else:
        # Run pytest directly when coverage is not enabled
        cmd = pytest_cmd
    
    # Prepare log file if logging is enabled
    log_file = prepare_log_file(options)
    
    # Display the command being executed
    cmd_str = " ".join(cmd)
    console.print(f"Executing: {cmd_str}", style=Style.COMMAND)
    console.print("")
    
    # Run the main command
    if log_file:
        console.print(f"Logging output to: {log_file}", style=Style.INFO)
        
        try:
            with open(log_file, "wb") as f_log:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    env=env
                )
                
                # Stream output to both terminal and log file
                if process.stdout:
                    for line in iter(process.stdout.readline, b''):
                        f_log.write(line)
                        try:
                            print(line.decode().strip())
                        except UnicodeDecodeError:
                            print(line)
                    
                    process.stdout.close()
                return_code = process.wait()
        except Exception as e:
            console.print(f"Error: {e}", style=Style.ERROR)
            return 1
    else:
        # Run with output directly to terminal
        process = subprocess.run(cmd, env=env)
        return_code = process.returncode
    
    # Generate coverage reports if coverage was enabled
    if options.get("coverage", False):
        console.print("\nCombining coverage data and generating reports...", style=Style.INFO)
        
        # Combine coverage data (safe to run even if not parallel)
        try:
            subprocess.run(["poetry", "run", "python", "-m", "coverage", "combine"],
                         check=False, env=env)
        except Exception as e:
            console.print(f"Warning: Coverage combine failed: {e}", style=Style.WARNING)
        
        # Generate terminal coverage report
        console.print("Terminal coverage report:", style=Style.INFO)
        try:
            subprocess.run(["poetry", "run", "python", "-m", "coverage", "report"],
                         check=False, env=env)
        except Exception as e:
            console.print(f"Error generating coverage report: {e}", style=Style.ERROR)
        
        # Generate HTML coverage report if requested
        if options.get("html_coverage", False):
            console.print("Generating HTML coverage report...", style=Style.INFO)
            try:
                subprocess.run(["poetry", "run", "python", "-m", "coverage", "html"],
                             check=False, env=env)
            except Exception as e:
                console.print(f"Error generating HTML coverage report: {e}", style=Style.ERROR)
        
        # Generate XML coverage report for Codecov
        console.print("Generating XML coverage report...", style=Style.INFO)
        try:
            subprocess.run(["poetry", "run", "python", "-m", "coverage", "xml"],
                         check=False, env=env)
            # Check if coverage.xml was created
            if Path("coverage.xml").exists():
                console.print("coverage.xml generated successfully.", style=Style.SUCCESS)
            else:
                console.print("coverage.xml was NOT generated.", style=Style.WARNING)
        except Exception as e:
            console.print(f"Error generating XML coverage report: {e}", style=Style.ERROR)
            
    # Display results
    if return_code == 0:
        console.print("\n✓ All selected tests passed!", style=Style.SUCCESS)
    else:
        console.print(f"\n✗ Some tests failed (exit code: {return_code}).", style=Style.ERROR)
    
    # Note HTML report location if applicable
    if options.get("html_coverage", False):
        html_report = Path("htmlcov/index.html").resolve()
        console.print(f"HTML coverage report: {html_report}", style=Style.INFO)
    
    return return_code


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Venice AI Test Runner - A modern replacement for run_tests.sh",
        epilog="Run with no arguments for interactive mode."
    )
    
    # Test selection options
    parser.add_argument(
        "paths", nargs="*", 
        help="Specific test files or directories to run"
    )
    parser.add_argument(
        "-g", "--group", choices=["unit", "e2e", "benchmark", "all"],
        help="Run a predefined group of tests"
    )
    
    # Test execution options
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "-c", "--coverage", action="store_true",
        help="Generate coverage report"
    )
    parser.add_argument(
        "--html", action="store_true",
        help="Generate HTML coverage report (implies --coverage)"
    )
    parser.add_argument(
        "-l", "--log", action="store_true",
        help="Save output to log file"
    )
    parser.add_argument(
        "--log-file", default=DEFAULT_LOG_FILE,
        help="Base name for log file"
    )
    parser.add_argument(
        "-x", "--fail-fast", action="store_true",
        help="Stop on first test failure"
    )
    parser.add_argument(
        "-p", "--parallel", action="store_true",
        help="Run tests in parallel (requires pytest-xdist)"
    )
    parser.add_argument(
        "--cov-dir", default=DEFAULT_COVERAGE_SOURCE,
        help="Source directory for coverage"
    )
    
    # Mode selection
    parser.add_argument(
        "-i", "--interactive", action="store_true",
        help="Force interactive mode"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the test runner."""
    args = parse_args()
    print_title()
    
    # Determine if we should use interactive mode
    use_interactive_mode = args.interactive or (
        not args.paths and 
        not args.group and 
        not any([args.verbose, args.coverage, args.html, args.log, args.fail_fast, args.parallel])
    )
    
    if use_interactive_mode:
        # Interactive mode
        console.print("Starting interactive mode...\n", style=Style.INFO)
        options = interactive_test_selection()
    else:
        # CLI mode - use provided arguments
        options = {
            "verbose": args.verbose,
            "coverage": args.coverage or args.html,  # HTML coverage implies coverage
            "html_coverage": args.html,
            "log": args.log,
            "log_filename_base": args.log_file,
            "fail_fast": args.fail_fast,
            "parallel": args.parallel,
            "cov_source": args.cov_dir,
        }
        
        # Handle test selection
        if args.paths:
            options["group"] = "specific"
            options["specific_files"] = args.paths
        elif args.group:
            options["group"] = args.group
        else:
            # Default to all tests if no selection is specified
            options["group"] = "all"
    
    # Run tests with selected options
    return_code = run_tests(options)
    
    # Exit with the same code as pytest
    sys.exit(return_code)


if __name__ == "__main__":
    main()