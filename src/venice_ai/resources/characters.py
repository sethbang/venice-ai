from typing import List

from .._resource import APIResource, AsyncAPIResource
from ..types.characters import CharacterList

class Characters(APIResource):
    """
    Provides methods for managing AI character definitions.
    
    Characters represent pre-defined personalities or specialized AI assistants
    that can be referenced in chat completions requests. This resource provides
    methods to list and interact with available characters.
    
    :param client: The Venice AI client instance used for API requests.
    :type client: venice_ai._client.VeniceClient
    
    .. warning::

        The Characters API is currently in Preview and may change in future releases.
    """
    
    def list(self) -> CharacterList:
        """
        Lists available characters.

        Retrieves a list of all characters usable with the Venice AI API.
        Each character includes details such as ID, name, and description.

        :return: A paginated list of available characters.
        :rtype: :class:`~venice_ai.types.characters.CharacterList`

        :raises venice_ai.exceptions.APIError: If the API request fails.
        
        Example:

            .. code-block:: python

                from venice_ai import VeniceClient
                from venice_ai.types.characters import CharacterList
                
                client = VeniceClient(api_key="your-api-key")
                characters_response = client.characters.list()
                for character in characters_response.data:
                    print(f"Character ID: {character.id}, Name: {character.name}")
        """
        response = self._client.get(
            "/characters",
        )
        return CharacterList(**response)

class AsyncCharacters(AsyncAPIResource):
    """
    Provides methods for managing AI character definitions asynchronously.
    
    Provides asynchronous methods to list and interact with available characters.
    This class mirrors the functionality of the synchronous :class:`Characters` resource
    but operates in an asynchronous context.
    
    :param client: The async Venice AI client instance used for API requests.
    :type client: venice_ai._async_client.AsyncVeniceClient
    
    .. warning::

        The Characters API is currently in Preview and may change in future releases.
    """
    
    async def list(self) -> CharacterList:
        """
        Lists available characters asynchronously.

        Retrieves a list of all characters usable with the Venice AI API asynchronously.
        Each character includes details such as ID, name, and description.

        :return: A paginated list of available characters.
        :rtype: :class:`~venice_ai.types.characters.CharacterList`

        :raises venice_ai.exceptions.APIError: If the API request fails.
            
        Example:

            .. code-block:: python

                import asyncio
                from venice_ai import AsyncVeniceClient
                from venice_ai.types.characters import CharacterList
                
                async def main():
                    client = AsyncVeniceClient(api_key="your-api-key")
                    characters_response = await client.characters.list()
                    for character in characters_response.data:
                        print(f"Character ID: {character.id}, Name: {character.name}")
                    await client.close()

                asyncio.run(main())
        """
        response = await self._client.get(
            "/characters",
        )
        return CharacterList(**response)