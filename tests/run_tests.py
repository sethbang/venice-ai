#!/usr/bin/env python
"""
Venice AI Test Runner

A comprehensive test runner that orchestrates unit, integration, and e2e tests
with unified coverage reporting and various execution modes.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestType(Enum):
    """Test type enumeration."""

    UNIT = "unit"
    INTEGRATION = "integration"
    E2E = "e2e"
    ALL = "all"
    SMOKE = "smoke"
    STRESS = "stress"


class Color:
    """ANSI color codes for terminal output."""

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


@dataclass
class TestResult:
    """Container for test execution results."""

    test_type: str
    passed: int
    failed: int
    skipped: int
    errors: int
    duration: float
    coverage: float | None = None
    exit_code: int = 0


class TestRunner:
    """Main test runner class."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.results: list[TestResult] = []
        self.project_root = Path(__file__).parent.parent
        self.test_dir = self.project_root / "tests"
        self.reports_dir = self.test_dir / "reports"

        # Ensure reports directory exists
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        (self.reports_dir / "coverage").mkdir(exist_ok=True)

    def run(self) -> int:
        """Run the test suite based on arguments."""
        print(f"{Color.HEADER}{'=' * 60}{Color.ENDC}")
        print(f"{Color.HEADER}Venice AI Test Suite Runner{Color.ENDC}")
        print(f"{Color.HEADER}{'=' * 60}{Color.ENDC}\n")

        # Determine which tests to run
        test_types = self._get_test_types()

        # Run tests
        for test_type in test_types:
            self._run_test_suite(test_type)

        # Combine coverage if needed
        if self.args.coverage and len(test_types) > 1:
            self._combine_coverage()

        # Generate reports
        if self.args.html_report:
            self._generate_html_report()

        # Print summary
        self._print_summary()

        # Return overall exit code
        return 0 if all(r.exit_code == 0 for r in self.results) else 1

    def _get_test_types(self) -> list[str]:
        """Determine which test types to run."""
        if self.args.test_type == TestType.ALL.value:
            return [TestType.UNIT.value, TestType.INTEGRATION.value, TestType.E2E.value]
        elif self.args.test_type == TestType.SMOKE.value:
            return [TestType.SMOKE.value]
        elif self.args.test_type == TestType.STRESS.value:
            return [TestType.STRESS.value]
        else:
            return [self.args.test_type]

    def _run_test_suite(self, test_type: str) -> None:
        """Run a specific test suite."""
        print(f"\n{Color.OKBLUE}Running {test_type.upper()} tests...{Color.ENDC}")
        print("-" * 40)

        start_time = time.time()

        # Build pytest command
        cmd = self._build_pytest_command(test_type)

        # Run tests
        if self.args.verbose:
            print(f"Command: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=not self.args.verbose)

        duration = time.time() - start_time

        # Parse results
        test_result = self._parse_test_results(test_type, result, duration)
        self.results.append(test_result)

        # Print immediate feedback
        if test_result.exit_code == 0:
            print(f"{Color.OKGREEN}✓ {test_type.upper()} tests passed{Color.ENDC}")
        else:
            print(f"{Color.FAIL}✗ {test_type.upper()} tests failed{Color.ENDC}")

        print(f"  Duration: {duration:.2f}s")
        print(
            f"  Passed: {test_result.passed}, Failed: {test_result.failed}, "
            f"Skipped: {test_result.skipped}"
        )

        if test_result.coverage is not None:
            print(f"  Coverage: {test_result.coverage:.1f}%")

    def _build_pytest_command(self, test_type: str) -> list[str]:
        """Build the pytest command for a test type."""
        cmd = ["pytest"]

        # Add test directory or marker
        if test_type == TestType.UNIT.value:
            cmd.extend([f"{self.test_dir}/unit", "-m", "unit"])
        elif test_type == TestType.INTEGRATION.value:
            cmd.extend([f"{self.test_dir}/integration", "-m", "integration"])
        elif test_type == TestType.E2E.value:
            cmd.extend([f"{self.test_dir}/e2e", "-m", "e2e"])
        elif test_type == TestType.SMOKE.value:
            cmd.extend([str(self.test_dir), "-m", "smoke"])
        elif test_type == TestType.STRESS.value:
            cmd.extend([f"{self.test_dir}/e2e/stress", "-m", "stress"])
        else:
            cmd.append(str(self.test_dir))

        # Add common options
        if not self.args.verbose:
            cmd.append("-q")
        else:
            cmd.extend(["-vv", "--tb=short"])

        # Add parallel execution
        if self.args.parallel:
            cmd.extend(["-n", str(self.args.parallel)])

        # Add coverage
        if self.args.coverage:
            coverage_file = self.reports_dir / "coverage" / f".coverage.{test_type}"
            cmd.extend(
                [
                    "--cov=src/venice_ai",
                    "--cov-branch",
                    "--cov-report=",  # No terminal report for individual runs
                    "--cov-append",
                ]
            )
            os.environ["COVERAGE_FILE"] = str(coverage_file)

        # Add markers to exclude
        if not self.args.include_slow:
            cmd.extend(["-m", "not slow"])

        if not self.args.include_api:
            cmd.extend(["-m", "not requires_api"])

        # Add specific test file if provided
        if self.args.test_file:
            cmd.append(self.args.test_file)

        # Add specific test if provided
        if self.args.test_name:
            cmd.extend(["-k", self.args.test_name])

        # Add max failures
        if self.args.max_failures:
            cmd.extend(["--maxfail", str(self.args.max_failures)])

        # Add JSON report for parsing
        json_report = self.reports_dir / f"test_results_{test_type}.json"
        cmd.extend(["--json-report", f"--json-report-file={json_report}"])

        return cmd

    def _parse_test_results(
        self, test_type: str, result: subprocess.CompletedProcess, duration: float
    ) -> TestResult:
        """Parse test results from pytest output."""
        # Try to parse JSON report
        json_report = self.reports_dir / f"test_results_{test_type}.json"

        test_result = TestResult(
            test_type=test_type,
            passed=0,
            failed=0,
            skipped=0,
            errors=0,
            duration=duration,
            exit_code=result.returncode,
        )

        if json_report.exists():
            try:
                with open(json_report) as f:
                    data = json.load(f)
                    summary = data.get("summary", {})
                    test_result.passed = summary.get("passed", 0)
                    test_result.failed = summary.get("failed", 0)
                    test_result.skipped = summary.get("skipped", 0)
                    test_result.errors = summary.get("error", 0)
            except Exception as e:
                print(f"Warning: Could not parse JSON report: {e}")

        # Parse coverage if enabled
        if self.args.coverage:
            coverage_file = self.reports_dir / "coverage" / f".coverage.{test_type}"
            if coverage_file.exists():
                coverage_pct = self._get_coverage_percentage(coverage_file)
                test_result.coverage = coverage_pct

        return test_result

    def _get_coverage_percentage(self, coverage_file: Path) -> float | None:
        """Get coverage percentage from a coverage file."""
        try:
            # Use coverage tool to get percentage
            cmd = [
                "coverage",
                "report",
                "--data-file",
                str(coverage_file),
                "--format=total",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                try:
                    return float(result.stdout.strip())
                except ValueError:
                    pass
        except Exception:
            pass
        return None

    def _combine_coverage(self) -> None:
        """Combine coverage from multiple test runs."""
        print(f"\n{Color.OKCYAN}Combining coverage reports...{Color.ENDC}")

        coverage_files = list((self.reports_dir / "coverage").glob(".coverage.*"))
        if not coverage_files:
            print("No coverage files to combine")
            return

        # Combine coverage files
        combined_file = self.reports_dir / "coverage" / ".coverage"
        cmd = ["coverage", "combine", "--data-file", str(combined_file)]
        cmd.extend([str(f) for f in coverage_files])

        result = subprocess.run(cmd, capture_output=True)
        if result.returncode == 0:
            print(f"{Color.OKGREEN}✓ Coverage combined successfully{Color.ENDC}")

            # Generate reports
            self._generate_coverage_reports(combined_file)
        else:
            print(f"{Color.FAIL}✗ Failed to combine coverage{Color.ENDC}")

    def _generate_coverage_reports(self, coverage_file: Path) -> None:
        """Generate various coverage reports."""
        # HTML report
        html_dir = self.reports_dir / "coverage" / "html"
        subprocess.run(
            [
                "coverage",
                "html",
                "--data-file",
                str(coverage_file),
                "--directory",
                str(html_dir),
            ],
            capture_output=True,
        )

        # XML report (for CI)
        xml_file = self.reports_dir / "coverage" / "coverage.xml"
        subprocess.run(
            ["coverage", "xml", "--data-file", str(coverage_file), "-o", str(xml_file)],
            capture_output=True,
        )

        # Terminal report
        result = subprocess.run(
            ["coverage", "report", "--data-file", str(coverage_file)],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print("\n" + result.stdout)

    def _generate_html_report(self) -> None:
        """Generate an HTML test report."""
        print(f"\n{Color.OKCYAN}Generating HTML report...{Color.ENDC}")

        html_file = self.reports_dir / "test_report.html"

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Venice AI Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .passed {{ color: green; font-weight: bold; }}
        .failed {{ color: red; font-weight: bold; }}
        .summary {{ margin-top: 20px; padding: 15px; background-color: #f9f9f9; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>Venice AI Test Report</h1>
    <div class="summary">
        <h2>Summary</h2>
        <p>Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}</p>
        <p>Total Test Suites: {len(self.results)}</p>
        <p>Overall Status: {'<span class="passed">PASSED</span>' if all(r.exit_code == 0 for r in self.results) else '<span class="failed">FAILED</span>'}</p>
    </div>

    <table>
        <tr>
            <th>Test Type</th>
            <th>Status</th>
            <th>Passed</th>
            <th>Failed</th>
            <th>Skipped</th>
            <th>Errors</th>
            <th>Duration (s)</th>
            <th>Coverage (%)</th>
        </tr>
"""

        for result in self.results:
            status = (
                '<span class="passed">PASSED</span>'
                if result.exit_code == 0
                else '<span class="failed">FAILED</span>'
            )
            coverage = f"{result.coverage:.1f}" if result.coverage else "N/A"

            html_content += f"""
        <tr>
            <td>{result.test_type.upper()}</td>
            <td>{status}</td>
            <td>{result.passed}</td>
            <td>{result.failed}</td>
            <td>{result.skipped}</td>
            <td>{result.errors}</td>
            <td>{result.duration:.2f}</td>
            <td>{coverage}</td>
        </tr>
"""

        html_content += """
    </table>
</body>
</html>
"""

        with open(html_file, "w") as f:
            f.write(html_content)

        print(f"{Color.OKGREEN}✓ HTML report generated: {html_file}{Color.ENDC}")

    def _print_summary(self) -> None:
        """Print final test summary."""
        print(f"\n{Color.HEADER}{'=' * 60}{Color.ENDC}")
        print(f"{Color.HEADER}Test Summary{Color.ENDC}")
        print(f"{Color.HEADER}{'=' * 60}{Color.ENDC}\n")

        total_passed = sum(r.passed for r in self.results)
        total_failed = sum(r.failed for r in self.results)
        total_skipped = sum(r.skipped for r in self.results)
        total_duration = sum(r.duration for r in self.results)

        print(f"Total Tests Run: {total_passed + total_failed}")
        print(f"  {Color.OKGREEN}Passed: {total_passed}{Color.ENDC}")
        print(f"  {Color.FAIL}Failed: {total_failed}{Color.ENDC}")
        print(f"  {Color.WARNING}Skipped: {total_skipped}{Color.ENDC}")
        print(f"\nTotal Duration: {total_duration:.2f}s")

        if self.args.coverage and any(r.coverage for r in self.results):
            avg_coverage = sum(r.coverage or 0 for r in self.results) / len(self.results)
            print(f"Average Coverage: {avg_coverage:.1f}%")

        # Overall status
        if all(r.exit_code == 0 for r in self.results):
            print(f"\n{Color.OKGREEN}{'=' * 60}{Color.ENDC}")
            print(f"{Color.OKGREEN}ALL TESTS PASSED ✓{Color.ENDC}")
            print(f"{Color.OKGREEN}{'=' * 60}{Color.ENDC}")
        else:
            print(f"\n{Color.FAIL}{'=' * 60}{Color.ENDC}")
            print(f"{Color.FAIL}TESTS FAILED ✗{Color.ENDC}")
            print(f"{Color.FAIL}{'=' * 60}{Color.ENDC}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Venice AI Test Runner - Comprehensive test suite orchestrator"
    )

    # Test selection
    parser.add_argument(
        "test_type",
        nargs="?",
        default="all",
        choices=[t.value for t in TestType],
        help="Type of tests to run (default: all)",
    )

    parser.add_argument("-f", "--test-file", help="Specific test file to run")

    parser.add_argument("-k", "--test-name", help="Specific test name/pattern to run")

    # Execution options
    parser.add_argument(
        "-p",
        "--parallel",
        type=int,
        metavar="N",
        help="Run tests in parallel with N workers",
    )

    parser.add_argument("--max-failures", type=int, help="Stop after N failures")

    parser.add_argument("--include-slow", action="store_true", help="Include slow tests")

    parser.add_argument(
        "--include-api", action="store_true", help="Include tests requiring API access"
    )

    # Coverage options
    parser.add_argument("-c", "--coverage", action="store_true", help="Enable coverage reporting")

    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=80.0,
        help="Minimum coverage threshold (default: 80%%)",
    )

    # Reporting options
    parser.add_argument("--html-report", action="store_true", help="Generate HTML test report")

    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Run tests
    runner = TestRunner(args)
    exit_code = runner.run()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
