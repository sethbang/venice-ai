from typing import AsyncIterator, Iterator, Any, Generic, TypeVar, TYPE_CHECKING
import httpx

if TYPE_CHECKING:
    from ._client import VeniceClient
    from ._async_client import AsyncVeniceClient

from .exceptions import StreamConsumedError, StreamClosedError

ChunkType = TypeVar("ChunkType") # Represents the type of items yielded by the raw iterator (typically Dict[str, Any])

class Stream(Generic[ChunkType]):
    """
    A synchronous wrapper for handling streaming responses from the Venice.ai API.
    
    This class provides a convenient interface for iterating over streaming API responses,
    typically yielding dictionaries containing response chunks. It manages the underlying
    iterator and ensures proper resource cleanup when the stream is exhausted or closed.
    
    :param iterator: The underlying iterator that yields response chunks
    :type iterator: Iterator[ChunkType]
    :param client: The Venice client instance for context and resource management
    :type client: VeniceClient
    
    Example:
        >>> stream = Stream(response_iterator, client=client)
        >>> for chunk in stream:
        ...     print(chunk)
        >>> stream.close()  # Optional, automatically called when iteration completes
    """
    def __init__(self, iterator: Iterator[ChunkType], *, client: "VeniceClient"): # Removed item_model_cls
        self._iterator = iterator
        self._client = client # Retain client for potential context or resource management
        self._consumed = False # Track if the stream has been consumed

    def __next__(self) -> ChunkType:
        """
        Get the next chunk from the stream.
        
        :return: The next chunk from the streaming response
        :rtype: ChunkType
        :raises StopIteration: When the stream is exhausted
        :raises StreamConsumedError: When the stream has already been consumed
        :raises StreamClosedError: When the stream has been closed
        """
        try:
            # Simply pass through the item from the raw iterator
            return next(self._iterator)
        except StopIteration:
            self._consumed = True  # Mark stream as consumed when exhausted
            self.close() # Ensure resources are cleaned up when iteration stops
            raise
        except httpx.StreamClosed as e:
            # Handle httpx.StreamClosed - translate to our custom error
            self._consumed = True  # A closed stream is effectively consumed
            raise StreamClosedError("Stream has been closed.", request=getattr(e, 'request', None)) from e
        except httpx.StreamConsumed as e:
            # Handle httpx.StreamConsumed - translate to our custom error
            self._consumed = True
            raise StreamConsumedError("Stream has already been consumed.", request=getattr(e, 'request', None)) from e
        except httpx.RequestError as e:
            # Handle httpx RequestError (which includes ReadError, ConnectError, etc.) during stream iteration
            self._consumed = True # Stream is consumed on this error
            self.close() # Ensure resources are cleaned up
            api_error = self._client._translate_httpx_error_to_api_error(e, e.request, is_stream=True) # Pass e.request directly
            raise api_error from e
        except httpx.HTTPStatusError as e:
            # Handle httpx HTTPStatusError during stream iteration
            self._consumed = True # Stream is consumed on this error
            self.close() # Ensure resources are cleaned up
            api_error = self._client._translate_httpx_error_to_api_error(e, e.request, is_stream=True) # Pass e.request directly
            raise api_error from e
    
    def __iter__(self) -> Iterator[ChunkType]:
        """
        Return the iterator object itself.
        
        :return: The stream iterator
        :rtype: Iterator[ChunkType]
        :raises StreamConsumedError: If the stream has already been consumed
        """
        if self._consumed:
            raise StreamConsumedError("Cannot iterate over a consumed stream.")
        return self

    def close(self) -> None:
        """
        Close the stream and release any underlying resources.
        
        This method ensures proper cleanup of the underlying iterator and any
        associated resources. It can be called multiple times safely and is
        automatically called when the stream iteration completes.
        
        :raises: No exceptions are raised; any cleanup errors are silently handled
        """
        # If the iterator has a close method, call it to clean up resources
        if hasattr(self._iterator, "close"):
            try:
                self._iterator.close()  # type: ignore[misc]
            except Exception:
                # Log or handle the error if necessary, but allow execution to continue
                pass

class AsyncStream(Generic[ChunkType]):
    """
    An asynchronous wrapper for handling streaming responses from the Venice.ai API.
    
    This class provides a convenient interface for asynchronously iterating over streaming
    API responses, typically yielding dictionaries containing response chunks. It manages
    the underlying async iterator and ensures proper resource cleanup when the stream is
    exhausted or closed.
    
    :param iterator: The underlying async iterator that yields response chunks
    :type iterator: AsyncIterator[ChunkType]
    :param client: The async Venice client instance for context and resource management
    :type client: AsyncVeniceClient
    
    Example:
        >>> stream = AsyncStream(response_iterator, client=async_client)
        >>> async for chunk in stream:
        ...     print(chunk)
        >>> await stream.close()  # Optional, automatically called when iteration completes
    """
    def __init__(self, iterator: AsyncIterator[ChunkType], *, client: "AsyncVeniceClient"): # Removed item_model_cls
        self._iterator = iterator
        self._client = client # Retain client for potential context or resource management
        self._consumed = False # Track if the stream has been consumed

    async def __anext__(self) -> ChunkType:
        """
        Asynchronously get the next chunk from the stream.
        
        :return: The next chunk from the streaming response
        :rtype: ChunkType
        :raises StopAsyncIteration: When the stream is exhausted
        :raises StreamConsumedError: When the stream has already been consumed
        :raises StreamClosedError: When the stream has been closed
        """
        try:
            # Simply pass through the item from the raw async iterator
            return await self._iterator.__anext__()
        except StopAsyncIteration:
            self._consumed = True  # Mark stream as consumed when exhausted
            await self.close() # Ensure resources are cleaned up
            raise
        except httpx.StreamClosed as e:
            # Handle httpx.StreamClosed - translate to our custom error
            self._consumed = True  # A closed stream is effectively consumed
            raise StreamClosedError("Stream has been closed.", request=getattr(e, 'request', None)) from e
        except httpx.StreamConsumed as e:
            # Handle httpx.StreamConsumed - translate to our custom error
            self._consumed = True
            raise StreamConsumedError("Stream has already been consumed.", request=getattr(e, 'request', None)) from e
        except httpx.RequestError as e:
            # Handle httpx RequestError (which includes ReadError, ConnectError, etc.) during stream iteration
            self._consumed = True # Stream is consumed on this error
            await self.close() # Ensure resources are cleaned up
            api_error = await self._client._translate_httpx_error_to_api_error(e, e.request, is_stream=True) # Pass e.request directly
            raise api_error from e
        except httpx.HTTPStatusError as e:
            # Handle httpx HTTPStatusError during stream iteration
            self._consumed = True # Stream is consumed on this error
            await self.close() # Ensure resources are cleaned up
            api_error = await self._client._translate_httpx_error_to_api_error(e, e.request, is_stream=True) # Pass e.request directly
            raise api_error from e

    def __aiter__(self) -> AsyncIterator[ChunkType]:
        """
        Return the async iterator object itself.
        
        :return: The async stream iterator
        :rtype: AsyncIterator[ChunkType]
        :raises StreamConsumedError: If the stream has already been consumed
        """
        if self._consumed:
            raise StreamConsumedError("Cannot iterate over a consumed stream.")
        return self

    async def close(self) -> None:
        """
        Asynchronously close the stream and release any underlying resources.
        
        This method ensures proper cleanup of the underlying async iterator and any
        associated resources. It can be called multiple times safely and is
        automatically called when the stream iteration completes.
        
        :raises: No exceptions are raised; any cleanup errors are silently handled
        """
        # If the iterator has an aclose method, call it to clean up resources
        if hasattr(self._iterator, "aclose"):
            try:
                # If aclose exists, assume it should be awaited.
                # The previous inspect.iscoroutinefunction check was too strict for AsyncMock.
                await self._iterator.aclose()  # type: ignore[misc]
            except Exception:
                # Log or handle the error if necessary, but allow execution to continue
                pass