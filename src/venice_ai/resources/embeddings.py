"""
Venice AI Embeddings API resources.

This module provides classes for interacting with the Venice AI Embeddings API,
allowing clients to generate embeddings from text or token inputs. These embeddings
are vector representations of text that capture semantic meaning and can be used
for tasks such as semantic search, clustering, and classification.
"""

from typing import List, Literal, Optional, Sequence, TypedDict, Union, Any, Dict, TYPE_CHECKING
import httpx # For creating dummy response objects

from .._resource import APIResource, AsyncAPIResource
from ..exceptions import InvalidRequestError
from venice_ai.types.embeddings import CreateEmbeddingRequest, EmbeddingList

if TYPE_CHECKING:
    from .._client import VeniceClient
    from .._async_client import AsyncVeniceClient


class Embeddings(APIResource):
    """
    Provides access to text embedding generation operations.
    
    This class manages synchronous embedding operations through the Venice AI API.
    Embeddings are vector representations of text that capture semantic meaning
    and can be used for various natural language processing tasks such as semantic
    search, clustering, classification, and similarity analysis.
    
    :param client: The Venice AI client instance used to make API requests.
    :type client: venice_ai._client.VeniceClient
    """
    
    def create(
        self,
        *,
        model: str, # type: ignore
        input: Union[str, List[str], List[int], List[List[int]]], # type: ignore
        dimensions: Optional[int] = None,
        encoding_format: Optional[Literal["float", "base64"]] = None,
        user: Optional[str] = None,
    ) -> EmbeddingList:
        """
        Generates embeddings for input text(s).
        
        This method sends a request to the Venice AI API to generate vector embeddings
        for the provided text or token inputs using the specified model. The embeddings
        can be used for semantic search, clustering, classification, and other NLP tasks.
        
        :param model: The ID of the embedding model to use. Available models can be
            retrieved using the models API. Example: ``'text-embedding-bge-m3'``.
        :type model: str
        :param input: The input text(s) to generate embeddings for. Can be a single
            string, a list of strings for batch processing, a list of token integers,
            or a list of token lists. For batch processing, all inputs will be
            processed together in a single API call.
        :type input: Union[str, List[str], List[int], List[List[int]]]
        :param dimensions: The number of dimensions for the output embeddings.
            If not specified, uses the model's default dimensionality. Some models
            support reducing dimensions for efficiency.
        :type dimensions: Optional[int]
        :param encoding_format: The format for the returned embeddings. Defaults
            to ``'float'`` for numerical arrays. Use ``'base64'`` for base64-encoded
            string representation.
        :type encoding_format: Optional[Literal["float", "base64"]]
        :param user: A unique identifier representing your end-user, which can help
            Venice AI monitor and detect abuse.
        :type user: Optional[str]

        :return: A response object containing the generated embeddings and usage data.
            The response includes an array of embedding objects, each containing the
            vector representation and associated metadata.
        :rtype: :class:`~venice_ai.types.embeddings.EmbeddingList`

        :raises venice_ai.exceptions.InvalidRequestError: If parameter values are invalid
            (e.g., empty model or input, unsupported encoding format).
        :raises venice_ai.exceptions.AuthenticationError: If the API key is invalid
            or missing.
        :raises venice_ai.exceptions.PermissionDeniedError: If access to the specified
            model is denied.
        :raises venice_ai.exceptions.NotFoundError: If the specified model is not found.
        :raises venice_ai.exceptions.RateLimitError: If rate limits are exceeded.
        :raises venice_ai.exceptions.APIError: For other API-related errors.
            
        **Examples:**

        Generate an embedding for a single string:

        .. code-block:: python

            from venice_ai import VeniceClient
            
            client = VeniceClient(api_key="your-api-key")
            response = client.embeddings.create(
                model="text-embedding-bge-m3",
                input="The quick brown fox jumps over the lazy dog."
            )
            embedding = response.data[0].embedding
            print(f"Embedding dimensions: {len(embedding)}")
            print(f"First 5 dimensions: {embedding[:5]}")

        Generate embeddings for multiple strings (batch processing):

        .. code-block:: python

            inputs = [
                "First sentence for embedding.",
                "Second sentence for embedding.",
                "Third sentence for embedding."
            ]
            batch_response = client.embeddings.create(
                model="text-embedding-bge-m3",
                input=inputs
            )
            for i, data_item in enumerate(batch_response.data):
                print(f"Embedding for '{inputs[i]}' (first 3 dims): {data_item.embedding[:3]}")
            print(f"Total tokens used: {batch_response.usage.total_tokens}")

        Using optional parameters:

        .. code-block:: python

            response = client.embeddings.create(
                model="text-embedding-bge-m3",
                input="Sample text for embedding",
                dimensions=512,  # Reduce dimensions if supported
                encoding_format="base64",  # Get base64-encoded embeddings
                user="user-123"  # Track usage by user
            )

        """
        if not model:
            # Create a dummy request and response for client-side validation error
            dummy_request = httpx.Request("POST", str(self._client._base_url.join("embeddings")))
            dummy_response = httpx.Response(400, request=dummy_request, json={"error": {"message": "model parameter is required and cannot be empty."}})
            raise InvalidRequestError("model parameter is required and cannot be empty.", request=dummy_request, response=dummy_response, body=dummy_response.json())
        if not input: # Handles empty string and empty list
            dummy_request = httpx.Request("POST", str(self._client._base_url.join("embeddings")))
            dummy_response = httpx.Response(400, request=dummy_request, json={"error": {"message": "input cannot be empty."}})
            raise InvalidRequestError("input cannot be empty.", request=dummy_request, response=dummy_response, body=dummy_response.json())

        # Build the request body
        body: Dict[str, Any] = {
            "model": model,
            "input": input,
        }
        
        # Add optional parameters if they're not None
        if dimensions is not None:
            body["dimensions"] = dimensions
            
        if encoding_format is not None:
            body["encoding_format"] = encoding_format
            
        if user is not None:
            body["user"] = user
            
        # Remove None values to avoid sending unnecessary fields
        body = {k: v for k, v in body.items() if v is not None}
            
        # Make the API request and return the response
        return self._client.post("embeddings", json_data=body)


class AsyncEmbeddings(AsyncAPIResource):
    """
    Provides access to text embedding generation operations (asynchronous).
    
    This class manages asynchronous embedding operations through the Venice AI API.
    It provides the same functionality as the synchronous :class:`Embeddings` class
    but uses async/await patterns for non-blocking operations. Embeddings are vector
    representations of text that capture semantic meaning and can be used for various
    natural language processing tasks.
    
    :param client: The async Venice AI client instance used to make API requests.
    :type client: venice_ai._async_client.AsyncVeniceClient
    """
    
    async def create(
        self,
        *,
        model: str, # type: ignore
        input: Union[str, List[str], List[int], List[List[int]]], # type: ignore
        dimensions: Optional[int] = None,
        encoding_format: Optional[Literal["float", "base64"]] = None,
        user: Optional[str] = None,
    ) -> EmbeddingList:
        """
        Generates embeddings for input text(s) asynchronously.
        
        This method sends an asynchronous request to the Venice AI API to generate
        vector embeddings for the provided text or token inputs using the specified
        model. The embeddings can be used for semantic search, clustering,
        classification, and other NLP tasks.
        
        :param model: The ID of the embedding model to use. Available models can be
            retrieved using the models API. Example: ``'text-embedding-bge-m3'``.
        :type model: str
        :param input: The input text(s) to generate embeddings for. Can be a single
            string, a list of strings for batch processing, a list of token integers,
            or a list of token lists. For batch processing, all inputs will be
            processed together in a single API call.
        :type input: Union[str, List[str], List[int], List[List[int]]]
        :param dimensions: The number of dimensions for the output embeddings.
            If not specified, uses the model's default dimensionality. Some models
            support reducing dimensions for efficiency.
        :type dimensions: Optional[int]
        :param encoding_format: The format for the returned embeddings. Defaults
            to ``'float'`` for numerical arrays. Use ``'base64'`` for base64-encoded
            string representation.
        :type encoding_format: Optional[Literal["float", "base64"]]
        :param user: A unique identifier representing your end-user, which can help
            Venice AI monitor and detect abuse.
        :type user: Optional[str]

        :return: A response object containing the generated embeddings and usage data.
            The response includes an array of embedding objects, each containing the
            vector representation and associated metadata.
        :rtype: :class:`~venice_ai.types.embeddings.EmbeddingList`

        :raises venice_ai.exceptions.InvalidRequestError: If parameter values are invalid
            (e.g., empty model or input, unsupported encoding format).
        :raises venice_ai.exceptions.AuthenticationError: If the API key is invalid
            or missing.
        :raises venice_ai.exceptions.PermissionDeniedError: If access to the specified
            model is denied.
        :raises venice_ai.exceptions.NotFoundError: If the specified model is not found.
        :raises venice_ai.exceptions.RateLimitError: If rate limits are exceeded.
        :raises venice_ai.exceptions.APIError: For other API-related errors.
            
        **Examples:**

        Generate an embedding for a single string:

        .. code-block:: python

            import asyncio
            from venice_ai import AsyncVeniceClient
            
            async def create_embedding():
                async with AsyncVeniceClient(api_key="your-api-key") as client:
                    response = await client.embeddings.create(
                        model="text-embedding-bge-m3",
                        input="The quick brown fox jumps over the lazy dog."
                    )
                    embedding = response.data[0].embedding
                    print(f"Embedding dimensions: {len(embedding)}")
                    print(f"First 5 dimensions: {embedding[:5]}")
            
            asyncio.run(create_embedding())

        Generate embeddings for multiple strings (batch processing):

        .. code-block:: python

            async def create_batch_embeddings():
                inputs = [
                    "First sentence for embedding.",
                    "Second sentence for embedding.",
                    "Third sentence for embedding."
                ]
                async with AsyncVeniceClient(api_key="your-api-key") as client:
                    batch_response = await client.embeddings.create(
                        model="text-embedding-bge-m3",
                        input=inputs
                    )
                    for i, data_item in enumerate(batch_response.data):
                        print(f"Embedding for '{inputs[i]}' (first 3 dims): {data_item.embedding[:3]}")
                    print(f"Total tokens used: {batch_response.usage.total_tokens}")
            
            asyncio.run(create_batch_embeddings())

        Using optional parameters:

        .. code-block:: python

            async def create_custom_embedding():
                async with AsyncVeniceClient(api_key="your-api-key") as client:
                    response = await client.embeddings.create(
                        model="text-embedding-bge-m3",
                        input="Sample text for embedding",
                        dimensions=512,  # Reduce dimensions if supported
                        encoding_format="base64",  # Get base64-encoded embeddings
                        user="user-123"  # Track usage by user
                    )
            
            asyncio.run(create_custom_embedding())

        """
        if not model:
            # Create a dummy request and response for client-side validation error
            dummy_request = httpx.Request("POST", str(self._client._base_url.join("embeddings")))
            dummy_response = httpx.Response(400, request=dummy_request, json={"error": {"message": "model parameter is required and cannot be empty."}})
            raise InvalidRequestError("model parameter is required and cannot be empty.", request=dummy_request, response=dummy_response, body=dummy_response.json())
        if not input: # Handles empty string and empty list
            dummy_request = httpx.Request("POST", str(self._client._base_url.join("embeddings")))
            dummy_response = httpx.Response(400, request=dummy_request, json={"error": {"message": "input cannot be empty."}})
            raise InvalidRequestError("input cannot be empty.", request=dummy_request, response=dummy_response, body=dummy_response.json())

        # Build the request body
        body: Dict[str, Any] = {
            "model": model,
            "input": input,
        }
        
        # Add optional parameters if they're not None
        if dimensions is not None:
            body["dimensions"] = dimensions
            
        if encoding_format is not None:
            body["encoding_format"] = encoding_format
            
        if user is not None:
            body["user"] = user
            
        # Remove None values to avoid sending unnecessary fields
        body = {k: v for k, v in body.items() if v is not None}
            
        # Make the API request and return the response
        return await self._client.post("embeddings", json_data=body)