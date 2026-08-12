# Venice AI chat resources package
from typing import TYPE_CHECKING

from ..._resource import APIResource
from .completions import ChatCompletions

if TYPE_CHECKING:
    from ..._client import VeniceClient


class ChatResource(APIResource["VeniceClient"]):
    """
    Provides asynchronous access to chat-related API operations.

    This class acts as a namespace for asynchronous chat functionalities and is accessed
    via ``client.chat``. It serves as a container for chat-related operations,
    primarily providing access to asynchronous chat completion functionality through the
    ``completions`` property.

    :param client: The VeniceClient instance.
    :type client: venice_ai._client.VeniceClient
    """

    completions: ChatCompletions
    """Access to asynchronous chat completion creation operations."""

    def __init__(self, client: "VeniceClient") -> None:
        """
        Initialize the ChatResource.

        :param client: The VeniceClient instance.
        :type client: venice_ai._client.VeniceClient
        """
        super().__init__(client)
        self.completions = ChatCompletions(client)


__all__ = ["ChatResource", "ChatCompletions"]
