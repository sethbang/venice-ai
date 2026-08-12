"""
Shared error-wrapping utilities for aiohttp requests.

Provides :func:`wrap_aiohttp_errors`, an async context manager that
translates low-level ``aiohttp`` / ``asyncio`` exceptions into the SDK's
:class:`~venice_ai.exceptions.APITimeoutError` and
:class:`~venice_ai.exceptions.APIConnectionError`.

This is the single, canonical aiohttp/asyncio error-translation context
manager, used by both the client request path and the multipart resource path.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import aiohttp


@asynccontextmanager
async def wrap_aiohttp_errors() -> AsyncIterator[None]:
    """Translate aiohttp / asyncio errors into SDK exceptions.

    Usage::

        from venice_ai.utils.errors import wrap_aiohttp_errors

        async with wrap_aiohttp_errors():
            response = await session.request(**kwargs)

    The exception ordering is intentional — more specific types are caught
    before their base classes:

    1. ``aiohttp.ServerTimeoutError`` → :class:`APITimeoutError`
    2. ``TimeoutError`` (includes ``asyncio.TimeoutError``) → :class:`APITimeoutError`
    3. ``aiohttp.ClientConnectorError`` → :class:`APIConnectionError`
    4. ``aiohttp.ClientError`` (catch-all) → :class:`APIConnectionError`
    """
    # Lazy imports to avoid circular dependencies at module load time.
    from venice_ai.exceptions import APIConnectionError, APITimeoutError

    try:
        yield
    except aiohttp.ServerTimeoutError as e:
        # Server-side timeout — catch more specific exception first
        raise APITimeoutError("Server timeout during request", original_error=e) from e
    except TimeoutError as e:
        # General timeout — includes client-side / asyncio timeouts
        raise APITimeoutError("Request timed out", original_error=e) from e
    except aiohttp.ClientConnectorError as e:
        # Connection errors (DNS, network unreachable, etc.)
        raise APIConnectionError("Connection failed", original_error=e) from e
    except aiohttp.ClientError as e:
        # Catch-all for other aiohttp client errors
        raise APIConnectionError("A connection error occurred", original_error=e) from e


__all__ = [
    "wrap_aiohttp_errors",
]
