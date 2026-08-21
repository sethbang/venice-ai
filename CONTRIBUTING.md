# Contributing to the Venice AI Python Client

Thank you for your interest in contributing to the Venice AI Python Client!

## Reporting Issues

If you encounter a bug or have an issue with the library:

1. **Search Existing Issues:** Before submitting a new issue, please check if a similar one has already been reported by searching on GitHub under [Issues](https://github.com/sethbang/venice-py/issues).
2. **Open a New Issue:** If you can't find an existing issue that addresses your problem, please [open a new one](https://github.com/sethbang/venice-py/issues/new).
   - Provide a **clear and descriptive title**.
   - Include a **detailed description** of the issue.
   - If applicable, provide a **code sample or an executable test case** that demonstrates the problem.
   - Mention your **Python version, library version, and operating system**.

## Development Setup

This project uses [Poetry](https://python-poetry.org/) for dependency management and requires **Python ≥3.13**.

```bash
# Install dependencies (including dev extras)
poetry install --all-extras

# Activate the virtual environment (Poetry 2.0+)
eval "$(poetry env activate)"
```

## Running Tests

```bash
# Run all unit tests
poetry run pytest tests/unit/

# Run with coverage
poetry run pytest tests/unit/ --cov=src/venice_ai --cov-report=term-missing

# Run integration tests (may require API credentials)
poetry run pytest tests/integration/

# Use the custom test runner for full suite orchestration
poetry run python tests/run_tests.py --help
```

## Code Style

This project enforces code style with **ruff** and type correctness with **mypy**.

```bash
# Lint and auto-fix
poetry run ruff check src/ tests/ --fix

# Format
poetry run ruff format src/ tests/

# Type check
poetry run mypy src/
```

All checks must pass before a PR can be merged. You can run the full check suite via:

```bash
make lint        # runs ruff lint
make type-check  # runs mypy
make test        # runs the test suite
```

## Pull Request Guidelines

- Branch naming: use descriptive names such as `fix/issue-description` or `feat/short-feature-name`.
- Keep PRs focused — one logical change per PR makes review easier.
- Add or update tests to cover your changes.
- Ensure `make lint` and `make test` pass locally before opening the PR.
- Update `CHANGELOG.md` under the `[Unreleased]` section if your change is user-visible.

## Questions?

If you have general questions about using the library that aren't bug reports, you can open an issue for discussion.

---

Your contributions are valuable!
