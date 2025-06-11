.. _api_reference:

API Reference
=============

Client Classes
--------------

.. automodule:: venice_ai
   :members: VeniceClient, AsyncVeniceClient
   :noindex:

.. autoclass:: venice_ai._client.VeniceClient
   :members: __init__, close, get, post, delete, chat, models, image, api_keys, audio, billing, embeddings, characters
   :undoc-members:
   :show-inheritance:
   :noindex:

.. autoclass:: venice_ai._async_client.AsyncVeniceClient
   :members: __init__, close, get, post, delete, chat, models, image, api_keys, audio, billing, embeddings, characters
   :undoc-members:
   :show-inheritance:
   :noindex:

Advanced HTTP Client Configuration
----------------------------------

The ``VeniceClient`` and ``AsyncVeniceClient`` offer flexible ways to configure the underlying HTTP client (``httpx.Client`` and ``httpx.AsyncClient`` respectively). This allows for advanced scenarios such as custom mTLS, specific proxy setups, detailed transport logging, or fine-tuning HTTP/2 behavior.

There are two primary methods for this:

1. Passing a Pre-configured ``httpx.Client`` / ``httpx.AsyncClient``
2. Passing Common ``httpx`` Settings Directly to the SDK Client Constructor

Passing a Pre-configured ``httpx.Client`` / ``httpx.AsyncClient``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can instantiate the SDK client with an ``http_client`` parameter, providing your own fully configured ``httpx.Client`` or ``httpx.AsyncClient`` instance. The SDK will use this instance directly for making API calls.

**Key Behaviors:**

*   **SDK Management:** The SDK will still manage the ``base_url`` and authentication headers for the requests. It will also apply its default ``timeout`` if a more specific timeout (e.g., per-request timeout) is not already configured on your provided client.
*   **Lifecycle Management:** You are responsible for the lifecycle (e.g., closing) of the ``httpx.Client`` or ``httpx.AsyncClient`` instance you provide. The SDK **will not** close a user-provided client, even when the SDK client is closed or used as a context manager.

**Use Cases:**

*   Implementing custom mutual TLS (mTLS) authentication.
*   Using a very specific proxy configuration not easily achieved with simple string/dict proxies.
*   Integrating advanced logging or monitoring at the HTTP transport layer.
*   Reusing an existing ``httpx.Client`` instance that is shared across different parts of your application.

**Example: ``VeniceClient`` with a custom ``httpx.Client``**

.. code-block:: python

   import httpx
   from venice_ai import VeniceClient

   # User creates and configures their own httpx.Client
   custom_transport = httpx.HTTPTransport(retries=5)
   my_httpx_client = httpx.Client(
       transport=custom_transport,
       proxies={"all://": "http://localhost:8080"}
   )

   # Pass it to VeniceClient
   # The user is responsible for closing my_httpx_client when done.
   client = VeniceClient(api_key="YOUR_API_KEY", http_client=my_httpx_client)

   # Use the client as usual
   try:
       models = client.models.list()
       print(models)
   finally:
       # User must close their client.
       # VeniceClient's close() or context manager __exit__ will NOT close my_httpx_client.
       if not my_httpx_client.is_closed:
           my_httpx_client.close()

**Example: ``AsyncVeniceClient`` with a custom ``httpx.AsyncClient``**

.. code-block:: python

   import httpx
   import asyncio
   from venice_ai import AsyncVeniceClient

   async def main():
       # User creates and configures their own httpx.AsyncClient
       custom_transport = httpx.AsyncHTTPTransport(retries=5)
       my_async_httpx_client = httpx.AsyncClient(
           transport=custom_transport,
           proxies={"all://": "http://localhost:8080"}
       )

       # Pass it to AsyncVeniceClient
       # The user is responsible for closing my_async_httpx_client when done.
       async_client = AsyncVeniceClient(api_key="YOUR_API_KEY", http_client=my_async_httpx_client)

       # Use the client as usual
       try:
           models = await async_client.models.list()
           print(models)
       finally:
           # User must close their client.
           # AsyncVeniceClient's aclose() or context manager __aexit__ will NOT close my_async_httpx_client.
           if not my_async_httpx_client.is_closed:
               await my_async_httpx_client.aclose()

   if __name__ == "__main__":
       asyncio.run(main())


Passing ``httpx`` Settings Directly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you do not provide your own ``http_client`` instance, you can pass common ``httpx.Client`` or ``httpx.AsyncClient`` constructor arguments directly to the ``VeniceClient`` or ``AsyncVeniceClient`` constructor. The SDK will use these arguments to create and manage its internal ``httpx`` client.

**Key Behaviors:**

*   **SDK Management:** The SDK creates, configures, and manages the lifecycle of the internal ``httpx.Client`` or ``httpx.AsyncClient``. It will be closed when the SDK client's ``close()`` (or ``aclose()``) method is called, or when the SDK client exits its context manager.
*   **Supported Parameters:** You can pass the following ``httpx`` constructor arguments:
    *   ``proxy``: A proxy URL or a dictionary mapping URL schemes to proxy URLs.
    *   ``proxies``: (Alternative to ``proxy``) A dictionary mapping URL schemes or specific domain/host patterns to proxy URLs.
    *   ``transport``: An ``httpx.HTTPTransport`` or ``httpx.AsyncHTTPTransport`` instance for advanced transport layer customization (e.g., connection pooling, retries, UNIX domain sockets).
    *   ``limits``: An ``httpx.Limits`` instance to configure connection limits (e.g., ``max_connections``, ``max_keepalive_connections``).
    *   ``cert``: An SSL certificate, either a path to a PEM file or a 2-tuple of (cert, key) file paths.
    *   ``verify``: SSL verification. Can be a boolean (``True``/``False``) or a path to a CA bundle file. Defaults to ``True``. Set to ``False`` with caution.
    *   ``trust_env``: A boolean indicating whether to trust environment variables for proxy configuration, SSL certificates, etc. Defaults to ``True``.
    *   ``http1``: A boolean indicating whether to allow HTTP/1.1 requests. Defaults to ``True``.
    *   ``http2``: A boolean indicating whether to enable HTTP/2 support. Defaults to ``False`` (httpx default).
    *   ``follow_redirects``: A boolean indicating whether to automatically follow redirects. Defaults to ``False`` for the SDK client.
    *   ``max_redirects``: The maximum number of redirects to follow if ``follow_redirects`` is ``True``.
    *   ``default_encoding``: A callable or string to determine the default encoding for response text.
    *   ``event_hooks``: A dictionary of event hooks (e.g., for request, response).

**Use Cases:**

*   Easily configuring a standard HTTP/S proxy.
*   Setting up custom SSL/TLS verification (e.g., using a corporate CA bundle).
*   Adjusting connection pool limits.
*   Enabling HTTP/2.
*   Customizing retry behavior via a custom ``transport``.

**Example: ``VeniceClient`` with direct ``httpx`` settings**

.. code-block:: python

   from venice_ai import VeniceClient
   import httpx # For httpx.Limits and httpx.HTTPTransport

   # Pass httpx settings directly to VeniceClient constructor
   # The SDK will create and manage its internal httpx.Client with these settings
   client = VeniceClient(
       api_key="YOUR_API_KEY",
       proxies={"all://": "http://localhost:8080"}, # Example proxy
       transport=httpx.HTTPTransport(retries=3),    # Example custom transport
       limits=httpx.Limits(max_connections=100, max_keepalive_connections=20), # Example limits
       verify=False                                 # Example: disable SSL verification (use with caution)
   )

   # Use the client as usual (SDK manages httpx.Client lifecycle)
   with client: # Or client.close() when done
       models = client.models.list()
       print(models)

**Example: ``AsyncVeniceClient`` with direct ``httpx`` settings**

.. code-block:: python

   from venice_ai import AsyncVeniceClient
   import httpx # For httpx.Limits and httpx.AsyncHTTPTransport
   import asyncio

   async def main():
       # Pass httpx settings directly to AsyncVeniceClient constructor
       # The SDK will create and manage its internal httpx.AsyncClient with these settings
       async_client = AsyncVeniceClient(
           api_key="YOUR_API_KEY",
           proxies={"all://": "http://localhost:8080"}, # Example proxy
           transport=httpx.AsyncHTTPTransport(retries=3), # Example custom transport
           limits=httpx.Limits(max_connections=100, max_keepalive_connections=20), # Example limits
           verify=False                                 # Example: disable SSL verification (use with caution)
       )

       # Use the client as usual (SDK manages httpx.AsyncClient lifecycle)
       async with async_client: # Or await async_client.aclose() when done
           models = await async_client.models.list()
           print(models)

   if __name__ == "__main__":
       asyncio.run(main())

Chat Resources
----------------

.. automodule:: venice_ai.resources.chat
   :members: ChatResource, AsyncChatResource
   :noindex:

.. automodule:: venice_ai.resources.chat.completions
   :members: ChatCompletions, AsyncChatCompletions
   :noindex:

.. autoclass:: venice_ai.resources.chat.ChatResource
   :members: completions
   :undoc-members:
   :show-inheritance:
   :noindex:

.. autoclass:: venice_ai.resources.chat.AsyncChatResource
   :members: completions
   :undoc-members:
   :show-inheritance:
   :noindex:

.. autoclass:: venice_ai.resources.chat.completions.ChatCompletions
   :members: create
   :undoc-members:
   :show-inheritance:
   :noindex:

.. autoclass:: venice_ai.resources.chat.completions.AsyncChatCompletions
   :members: create
   :undoc-members:
   :show-inheritance:
   :noindex:

Models Resources
----------------

.. autoclass:: venice_ai.resources.models.Models
   :members: list, list_traits, list_compatibility
   :undoc-members:
   :show-inheritance:
   :noindex:

.. autoclass:: venice_ai.resources.models.AsyncModels
   :members: list, list_traits, list_compatibility
   :undoc-members:
   :show-inheritance:
   :noindex:

Image Resources
---------------

.. automodule:: venice_ai.resources.image
   :members: Image, AsyncImage
   :noindex:

.. autoclass:: venice_ai.resources.image.Image
   :members: generate, simple_generate, upscale, list_styles
   :undoc-members:
   :show-inheritance:
   :noindex:

.. autoclass:: venice_ai.resources.image.AsyncImage
   :members: generate, simple_generate, upscale, list_styles
   :undoc-members:
   :show-inheritance:
   :noindex:

API Keys Resources
------------------

.. automodule:: venice_ai.resources.api_keys
   :members: ApiKeys, AsyncApiKeys
   :noindex:

.. autoclass:: venice_ai.resources.api_keys.ApiKeys
   :members: list, create, delete, get_web3_token, create_web3_key, get_rate_limits, get_rate_limit_logs
   :undoc-members:
   :show-inheritance:
   :noindex:

.. autoclass:: venice_ai.resources.api_keys.AsyncApiKeys
   :members: list, create, delete, get_web3_token, create_web3_key, get_rate_limits, get_rate_limit_logs
   :undoc-members:
   :show-inheritance:
   :noindex:

Audio Resources
---------------

.. automodule:: venice_ai.resources.audio
   :members: Audio, AsyncAudio
   :noindex:

.. autoclass:: venice_ai.resources.audio.Audio
   :members: create_speech
   :undoc-members:
   :show-inheritance:
   :noindex:

.. autoclass:: venice_ai.resources.audio.AsyncAudio
   :members: create_speech
   :undoc-members:
   :show-inheritance:
   :noindex:

Embeddings Resources
--------------------

.. automodule:: venice_ai.resources.embeddings
   :members: Embeddings, AsyncEmbeddings
   :noindex:

.. autoclass:: venice_ai.resources.embeddings.Embeddings
   :members: create
   :undoc-members:
   :show-inheritance:
   :noindex:

.. autoclass:: venice_ai.resources.embeddings.AsyncEmbeddings
   :members: create
   :undoc-members:
   :show-inheritance:
   :noindex:

Billing Resources
-----------------

.. automodule:: venice_ai.resources.billing
   :members: Billing, AsyncBilling
   :noindex:

.. autoclass:: venice_ai.resources.billing.Billing
   :members: get_usage
   :undoc-members:
   :show-inheritance:
   :noindex:

.. autoclass:: venice_ai.resources.billing.AsyncBilling
   :members: get_usage
   :undoc-members:
   :show-inheritance:
   :noindex:

Characters Resources
--------------------

.. automodule:: venice_ai.resources.characters
   :members: Characters, AsyncCharacters
   :noindex:

.. autoclass:: venice_ai.resources.characters.Characters
   :members: list
   :undoc-members:
   :show-inheritance:
   :noindex:

.. autoclass:: venice_ai.resources.characters.AsyncCharacters
   :members: list
   :undoc-members:
   :show-inheritance:
   :noindex:

Type Definitions
----------------

.. automodule:: venice_ai.types.chat
   :members:
   :noindex:

.. automodule:: venice_ai.types.models
   :members:
   :noindex:

.. automodule:: venice_ai.types.image
   :members:
   :noindex:

.. automodule:: venice_ai.types.api_keys
   :members:
   :noindex:

.. automodule:: venice_ai.types.audio
   :members:
   :noindex:

.. automodule:: venice_ai.types.embeddings
   :members:
   :noindex:

.. automodule:: venice_ai.types.billing
   :members:
   :noindex:

.. automodule:: venice_ai.types.characters
   :members:
   :noindex:

Exceptions
----------

.. automodule:: venice_ai.exceptions
   :members:
   :noindex:

Utilities
---------

For utility functions provided by the library, please see the :doc:`client_utilities` page.