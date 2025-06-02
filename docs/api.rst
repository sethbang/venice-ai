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
   :members: get_usage, export
   :undoc-members:
   :show-inheritance:
   :noindex:

.. autoclass:: venice_ai.resources.billing.AsyncBilling
   :members: get_usage, export
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

.. automodule:: venice_ai.utils
  :members: get_filtered_models, estimate_token_count, validate_chat_messages, find_model_by_id, get_model_capabilities, format_tool_response
  :noindex: