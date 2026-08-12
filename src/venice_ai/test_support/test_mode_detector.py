"""Test mode detection utilities for Venice AI.

This module provides the TestModeDetector class which helps detect when code
is running in a test environment, including support for parallel test execution.
"""

import os
import sys


class TestModeDetector:
    """Detects when code is running in a test environment.

    This class provides static methods to determine if the code is running
    in a test environment, whether tests are running in parallel mode,
    and to get the worker ID for parallel test execution.
    """

    @staticmethod
    def is_test_mode() -> bool:
        """Check if the code is running in a test environment.

        Returns True if any of the following conditions are met:
        - PYTEST_CURRENT_TEST environment variable is set
        - 'pytest' is in sys.modules
        - VENICE_TEST_MODE environment variable is set to '1' or 'true'

        Returns:
            bool: True if running in test mode, False otherwise.
        """
        # Check if PYTEST_CURRENT_TEST is set (pytest sets this during test runs)
        if os.environ.get("PYTEST_CURRENT_TEST"):
            return True

        # Check if pytest is imported
        if "pytest" in sys.modules:
            return True

        # Check custom test mode environment variable
        venice_test_mode = os.environ.get("VENICE_TEST_MODE", "").lower()
        return venice_test_mode in ("1", "true")

    @staticmethod
    def is_parallel_mode() -> bool:
        """Check if tests are running in parallel mode using pytest-xdist.

        Returns True if the PYTEST_XDIST_WORKER environment variable is set,
        which indicates that pytest-xdist is running tests in parallel.

        Returns:
            bool: True if running in parallel mode, False otherwise.
        """
        return bool(os.environ.get("PYTEST_XDIST_WORKER"))

    @staticmethod
    def get_worker_id() -> str:
        """Get the worker ID for parallel test execution.

        Returns the value of the PYTEST_XDIST_WORKER environment variable,
        or 'master' if it's not set (indicating single-process execution).

        Returns:
            str: The worker ID (e.g., 'gw0', 'gw1') or 'master' if not in parallel mode.
        """
        return os.environ.get("PYTEST_XDIST_WORKER", "master")
