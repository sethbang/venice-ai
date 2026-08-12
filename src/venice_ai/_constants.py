"""
Venice AI SDK Constants
======================

This module defines global constants used throughout the Venice AI SDK.
These constants configure default behaviors for HTTP requests, timeouts,
and other core functionality.

The constants are designed to provide sensible defaults while allowing
customization through client configuration.
"""

import aiohttp

#: Default base URL for the Venice AI API
#: All API requests will be made to endpoints under this base URL
DEFAULT_BASE_URL = "https://api.venice.ai/api/v1"

#: Default timeout configuration for HTTP requests
#: Uses aiohttp.ClientTimeout with:
#: - total: 120 seconds maximum total request time
#: - connect: 5 seconds maximum connection establishment time
DEFAULT_TIMEOUT = aiohttp.ClientTimeout(total=120.0, connect=5.0)

# Environment variable names
#: Environment variable to enable rate limiter features
#: Set to "true" to enable rate limiting functionality
ENV_RATE_LIMITER_ENABLED = "VENICE_RATE_LIMITER_FEATURES_ENABLED"
