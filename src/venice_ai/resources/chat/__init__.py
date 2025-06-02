# Venice AI chat resources package
from typing import TYPE_CHECKING

from ..._resource import APIResource, AsyncAPIResource
from .completions import AsyncChatCompletions, ChatCompletions

if TYPE_CHECKING:
    from ..._client import VeniceClient
    from ..._async_client import AsyncVeniceClient


class ChatResource(APIResource):
    """
    Provides access to chat-related API operations.

    This class acts as a namespace for chat functionalities and is accessed
    via ``client.chat``. It serves as a container for chat-related operations,
    primarily providing access to chat completion functionality through the
    ``completions`` property.

    :param client: The synchronous VeniceClient instance.
    :type client: venice_ai._client.VeniceClient
    """
    completions: ChatCompletions
    """Access to chat completion creation operations."""

    def __init__(self, client: "VeniceClient") -> None:
        """
        Initialize the ChatResource.

        :param client: The synchronous VeniceClient instance.
        :type client: venice_ai._client.VeniceClient
        """
        super().__init__(client)
        self.completions = ChatCompletions(client)


class AsyncChatResource(AsyncAPIResource):
    """
    Provides asynchronous access to chat-related API operations.

    This class acts as a namespace for asynchronous chat functionalities and is accessed
    via ``async_client.chat``. It serves as a container for chat-related operations,
    primarily providing access to asynchronous chat completion functionality through the
    ``completions`` property.

    :param client: The asynchronous AsyncVeniceClient instance.
    :type client: venice_ai._async_client.AsyncVeniceClient
    """
    completions: AsyncChatCompletions
    """Access to asynchronous chat completion creation operations."""

    def __init__(self, client: "AsyncVeniceClient") -> None:
        """
        Initialize the AsyncChatResource.

        :param client: The asynchronous AsyncVeniceClient instance.
        :type client: venice_ai._async_client.AsyncVeniceClient
        """
        super().__init__(client)
        self.completions = AsyncChatCompletions(client)

__all__ = ["ChatResource", "AsyncChatResource", "ChatCompletions", "AsyncChatCompletions"] # Added Completions to __all__