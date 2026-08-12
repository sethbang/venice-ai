"""Base abstract class for shared state backends.

This module provides the abstract base class that defines the interface
for shared state backends used in parallel test execution.
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseSharedStateBackend(ABC):
    """Abstract base class for shared state backends.

    This class defines the interface that all shared state backends must implement
    to support parallel test execution with shared rate limit state.
    """

    @abstractmethod
    def get_state(self, key: str) -> dict[str, Any] | None:
        """Get the state for a given key.

        Args:
            key: The key to retrieve state for.

        Returns:
            The state dictionary if it exists, None otherwise.
        """
        pass

    @abstractmethod
    def set_state(self, key: str, state: dict[str, Any]) -> None:
        """Set the state for a given key.

        Args:
            key: The key to set state for.
            state: The state dictionary to store.
        """
        pass

    @abstractmethod
    def get_all_states(self) -> dict[str, dict[str, Any]]:
        """Get all stored states.

        Returns:
            A dictionary mapping keys to their state dictionaries.
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all stored states."""
        pass

    @abstractmethod
    def check_and_reserve_capacity(
        self, key: str, requests: int, tokens: int
    ) -> tuple[bool, str | None]:
        """Atomically check and reserve capacity."""
        _ = (key, requests, tokens)
        pass

    @abstractmethod
    def release_reservation(self, key: str, reservation_id: str) -> None:
        """Release a reservation."""
        pass

    def update_state(self, key: str, updates: dict[str, Any]) -> None:
        """Update the state for a given key with partial data.

        This method is optional and provides a default implementation that
        can be overridden by backends for performance.

        Args:
            key: The key to update state for.
            updates: A dictionary of key-value pairs to update.
        """
        current_state = self.get_state(key) or {}
        current_state.update(updates)
        self.set_state(key, current_state)
