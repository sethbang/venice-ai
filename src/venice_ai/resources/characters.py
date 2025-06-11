from typing import Any, Dict, Optional
import httpx

from .._resource import APIResource, AsyncAPIResource
from ..types.characters import CharacterList


class Characters(APIResource):
    """
    Provides methods for managing AI character definitions.
    
    Characters represent pre-defined personalities or specialized AI assistants
    that can be referenced in chat completions requests. This resource provides
    methods to list available characters.
    
    :param client: The Venice AI client instance used for API requests.
    :type client: venice_ai._client.VeniceClient
    
    .. warning::

        The Characters API is currently in Preview and may change in future releases.
    """
    
    def list(
        self,
        *,
        extra_headers: Optional[httpx.Headers] = None,
        extra_query: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> CharacterList:
        """
        List all characters.

        Retrieves a list of all characters usable with the Venice AI API.
        Each character includes details such as ID, name, and description.

        :param extra_headers: Additional HTTP headers to include in the request.
        :type extra_headers: Optional[httpx.Headers]
        :param extra_query: Additional query parameters to include in the request.
        :type extra_query: Optional[Dict[str, Any]]
        :param extra_body: Additional body parameters to include in the request.
        :type extra_body: Optional[Dict[str, Any]]
        :param timeout: Request timeout in seconds.
        :type timeout: Optional[float]
        :return: A list of available characters.
        :rtype: :class:`~venice_ai.types.characters.CharacterList`

        :raises venice_ai.exceptions.APIError: If the API request fails.
        
        Example:

            .. code-block:: python

                from venice_ai import VeniceClient
                
                client = VeniceClient(api_key="your-api-key")
                characters_response = client.characters.list()
                for character in characters_response.data:
                    print(f"Character ID: {character.slug}, Name: {character.name}")
        """
        headers = {}
        if extra_headers:
            headers.update(extra_headers)
        
        params = {}
        if extra_query:
            params.update(extra_query)
            
        return CharacterList.model_validate(self._client.get(
            "characters",
            headers=headers if headers else None,
            params=params if params else None,
            timeout=timeout,
        ))


class AsyncCharacters(AsyncAPIResource):
    """
    Provides methods for managing AI character definitions asynchronously.
    
    Provides asynchronous methods to list available characters.
    This class mirrors the functionality of the synchronous :class:`Characters` resource
    but operates in an asynchronous context.
    
    :param client: The async Venice AI client instance used for API requests.
    :type client: venice_ai._async_client.AsyncVeniceClient
    
    .. warning::

        The Characters API is currently in Preview and may change in future releases.
    """
    
    async def list(
        self,
        *,
        extra_headers: Optional[httpx.Headers] = None,
        extra_query: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> CharacterList:
        """
        List all characters asynchronously.

        Retrieves a list of all characters usable with the Venice AI API asynchronously.
        Each character includes details such as ID, name, and description.

        :param extra_headers: Additional HTTP headers to include in the request.
        :type extra_headers: Optional[httpx.Headers]
        :param extra_query: Additional query parameters to include in the request.
        :type extra_query: Optional[Dict[str, Any]]
        :param extra_body: Additional body parameters to include in the request.
        :type extra_body: Optional[Dict[str, Any]]
        :param timeout: Request timeout in seconds.
        :type timeout: Optional[float]
        :return: A list of available characters.
        :rtype: :class:`~venice_ai.types.characters.CharacterList`

        :raises venice_ai.exceptions.APIError: If the API request fails.
            
        Example:

            .. code-block:: python

                import asyncio
                from venice_ai import AsyncVeniceClient
                
                async def main():
                    client = AsyncVeniceClient(api_key="your-api-key")
                    characters_response = await client.characters.list()
                    for character in characters_response.data:
                        print(f"Character ID: {character.slug}, Name: {character.name}")
                    await client.close()

                asyncio.run(main())
        """
        headers = {}
        if extra_headers:
            headers.update(extra_headers)
        
        params = {}
        if extra_query:
            params.update(extra_query)
            
        return CharacterList.model_validate(await self._client.get(
            "characters",
            headers=headers if headers else None,
            params=params if params else None,
            timeout=timeout,
        ))