# Expose the main client and core exceptions
from ._client import VeniceClient
from ._async_client import AsyncVeniceClient
from .exceptions import (
    VeniceError,
    APIError,
    AuthenticationError,
    PermissionDeniedError,
    InvalidRequestError,
    NotFoundError,
    ConflictError,
    UnprocessableEntityError,
    RateLimitError,
    InternalServerError,
    StreamConsumedError,
    StreamClosedError,
)

# Import utility functions
from .utils import get_filtered_models
from .costs import calculate_completion_cost, calculate_embedding_cost, estimate_completion_cost

# Optional: Configure logging
import logging
# Note: Removed NullHandler to allow retry log messages to be displayed
# logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "VeniceClient",
    "AsyncVeniceClient",
    "VeniceError",
    "APIError",
    "AuthenticationError",
    "PermissionDeniedError",
    "InvalidRequestError",
    "NotFoundError",
    "ConflictError",
    "UnprocessableEntityError",
    "RateLimitError",
    "InternalServerError",
    "StreamConsumedError",
    "StreamClosedError",
    "get_filtered_models", # Add the new utility function here
    "calculate_completion_cost",
    "calculate_embedding_cost",
    "estimate_completion_cost",
]

__version__ = "1.3.0" # Keep in sync with pyproject.toml