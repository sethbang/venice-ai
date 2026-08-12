"""Mock utilities for testing API failure scenarios.

This module provides utilities for creating controlled API failures during testing,
allowing tests to verify rate limit gate behavior without polluting the global
failed request counter with real API calls.
"""

from typing import Any

from venice_ai.exceptions import APIError
from venice_ai.types import (
    ChatChoice,
    ChatCompletionResponse,
    ChatMessage,
    ChatUsage,
)


def create_mock_chat_response(
    response_id: str = "mock-chatcmpl-123",
    model: str = "llama-3.2-3b",
    content: str = "This is a mock response.",
) -> ChatCompletionResponse:
    """Create a realistic mock chat completion response.

    Args:
        response_id: The response ID to use
        model: The model name to use
        content: The response content

    Returns:
        A ChatCompletionResponse object with mock data
    """
    # Create mock message
    message = ChatMessage(role="assistant", content=content)

    # Create mock choice
    choice = ChatChoice(
        index=0, message=message, finish_reason="stop", logprobs=None, stop_reason=None
    )

    # Create mock usage
    usage = ChatUsage(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        prompt_tokens_details=None,
    )

    # Create the response
    response = ChatCompletionResponse(
        id=response_id,
        object="chat.completion",
        created=1234567890,
        model=model,
        choices=[choice],
        usage=usage,
        cost=None,
        prompt_logprobs=None,
        venice_parameters=None,
        service_tier=None,
        system_fingerprint=None,
        kv_transfer_params=None,
    )

    return response


def create_mock_api_error(
    status_code: int = 400, message: str = "Invalid model specified"
) -> APIError:
    """Create a mock API error for testing failure scenarios.

    Args:
        status_code: HTTP status code
        message: Error message

    Returns:
        An APIError instance
    """

    # Create a mock response object using a simple class
    class MockResponse:
        def __init__(self, status: int, headers: dict[str, str] | None = None):
            self.status = status
            self.headers = headers or {}

        async def json(self) -> dict[str, Any]:
            return {"error": {"message": message, "type": "invalid_request_error"}}

        async def text(self) -> str:
            return f'{{"error": {{"message": "{message}", "type": "invalid_request_error"}}}}'

    mock_response = MockResponse(status_code)

    return APIError(message=message, response=mock_response)


class MockFailureScenario:
    """Context manager for controlled API failure testing."""

    def __init__(self, client: Any, failure_count: int = 1, success_after: bool = True):
        """Initialize the mock failure scenario.

        Args:
            client: The Venice client to mock
            failure_count: Number of failures to simulate
            success_after: Whether to allow success after failures
        """
        self.client = client
        self.failure_count = failure_count
        self.success_after = success_after
        self.original_create = None
        self.call_count = 0

    async def __aenter__(self) -> "MockFailureScenario":
        """Enter the context and set up mocking."""
        # Store original method
        self.original_create = self.client.chat.completions.create

        # Create mock that fails then succeeds
        async def mock_create(*_: Any, **kwargs: Any) -> Any:
            self.call_count += 1

            if self.call_count <= self.failure_count:
                # Manually record the failure in state manager
                scheduler = getattr(self.client, "rate_limiter", None)
                if scheduler and hasattr(scheduler, "rate_limit_gate"):
                    state_manager = scheduler.rate_limit_gate.state_manager
                    model = kwargs.get("model", "test-model")
                    await state_manager.record_failed_request(model)

                # Raise the API error
                raise create_mock_api_error(message=f"Mock failure {self.call_count}")
            elif self.success_after:
                # Return success response
                return create_mock_chat_response(
                    response_id=f"mock-success-{self.call_count}",
                    content=f"Mock success response {self.call_count}",
                )
            else:
                # Continue failing
                raise create_mock_api_error(message=f"Continued mock failure {self.call_count}")

        # Replace the method
        self.client.chat.completions.create = mock_create
        return self

    async def __aexit__(self, exc_type: Any, _exc_val: Any, _exc_tb: Any) -> None:
        """Exit the context and restore original method."""
        if self.original_create:
            self.client.chat.completions.create = self.original_create


async def simulate_api_failures(
    client: Any, failure_count: int, model: str = "test-model"
) -> list[Exception]:
    """Simulate API failures without making real API calls.

    Args:
        client: The Venice client
        failure_count: Number of failures to simulate
        model: Model name to use for failure recording

    Returns:
        List of exceptions that would have been raised
    """
    exceptions = []
    scheduler = getattr(client, "rate_limiter", None)

    if scheduler and hasattr(scheduler, "rate_limit_gate"):
        state_manager = scheduler.rate_limit_gate.state_manager

        for i in range(failure_count):
            # Record the failure in state manager
            await state_manager.record_failed_request(model)

            # Create the exception that would have been raised
            error = create_mock_api_error(message=f"Simulated API failure {i + 1}", status_code=400)
            exceptions.append(error)

    return exceptions


async def reset_failed_request_counter(client: Any, model: str = "default") -> None:
    """Reset the failed request counter for a specific model.

    Args:
        client: The Venice client
        model: Model name to reset counter for
    """
    scheduler = getattr(client, "rate_limiter", None)
    if scheduler and hasattr(scheduler, "rate_limit_gate"):
        state_manager = scheduler.rate_limit_gate.state_manager
        if model in state_manager.failed_request_counters:
            await state_manager.failed_request_counters[model].reset()


async def manually_record_failures(
    client: Any, failure_count: int, model: str = "test-model"
) -> None:
    """Manually record failures in the state manager without making API calls.

    This is useful for testing gate behavior with a controlled number of failures.

    Args:
        client: The Venice client
        failure_count: Number of failures to record
        model: Model name to record failures for
    """
    scheduler = getattr(client, "rate_limiter", None)
    if scheduler and hasattr(scheduler, "rate_limit_gate"):
        state_manager = scheduler.rate_limit_gate.state_manager

        for _i in range(failure_count):
            await state_manager.record_failed_request(model)


async def get_failed_request_count(client: Any, model: str = "default") -> int:
    """Get the current failed request count for a model.

    Args:
        client: The Venice client
        model: Model name to check

    Returns:
        Current failed request count
    """
    scheduler = getattr(client, "rate_limiter", None)
    if scheduler and hasattr(scheduler, "rate_limit_gate"):
        state_manager = scheduler.rate_limit_gate.state_manager
        count = await state_manager.get_failed_count(model)
        return int(count) if count is not None else 0
    return 0
