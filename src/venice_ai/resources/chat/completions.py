from typing import Optional, Dict, Any, Union, Literal, Mapping, overload, Sequence, Iterator, AsyncIterator, TYPE_CHECKING, Type, TypeVar, cast
import inspect
import warnings
from typing_extensions import TypedDict # Use typing.TypedDict in Python >= 3.8

from ...streaming import Stream, AsyncStream
from ..._resource import APIResource, AsyncAPIResource
if TYPE_CHECKING:
    from ..._client import VeniceClient
from ...exceptions import InvalidRequestError, MissingStreamClassError, APIResponseProcessingError
from ...types.chat import (
    MessageParam, VeniceParameters, ResponseFormat, UsageData,
    ChatCompletionChoice, ChatCompletion, ChatCompletionChunk,
    Tool, ToolChoice, ToolChoiceObject, StreamOptions, ChunkModelFactory
)

# Re-export types for backwards compatibility
__all__ = [
    "ChatCompletions",
    "AsyncChatCompletions"
]

# --- Resource Class ---

class ChatCompletions(APIResource):
    """
    Provides access to chat completion operations.

    This class manages synchronous chat completion operations with Venice AI models,
    supporting both standard (non-streaming) and streaming response formats. It serves
    as the primary interface for chat-based interactions with Venice AI language models.

    The class handles parameter validation, request formation, and response parsing
    for chat completion requests.

    :param _client: The client instance used to make API requests.
    :type _client: venice_ai._client.VeniceClient

    Example:

        .. code-block:: python

           from venice_ai import VeniceClient
           
           # Initialize the client
           client = VeniceClient(api_key="your-api-key")
           
           # Create a chat completion
           response = client.chat.completions.create(
               model="venice-1",
               messages=[
                   {"role": "system", "content": "You are a helpful assistant."},
                   {"role": "user", "content": "Tell me about Venice AI."}
               ]
           )
           
           # Access the response content
           print(response["choices"][0]["message"]["content"])
    """

    @overload
    def create(
        self,
        *,
        model: str,
        messages: Sequence[MessageParam],
        stream: Literal[False] = False, # Explicit non-streaming case
        # --- Common Optional Parameters ---
        frequency_penalty: Optional[float] = None,
        max_tokens: Optional[int] = None, # Deprecated. Please use max_completion_tokens instead.
        max_completion_tokens: Optional[int] = None,
        n: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        response_format: Optional[ResponseFormat] = None,
        seed: Optional[int] = None,
        stop: Optional[Union[str, Sequence[str]]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        tools: Optional[Sequence[Tool]] = None,
        tool_choice: Optional[Union[Literal["none", "auto"], ToolChoiceObject]] = None,
        user: Optional[str] = None, # Discarded but supported for OpenAI compat
        venice_parameters: Optional[VeniceParameters] = None,
        # --- Less Common / Newer Params from Docs ---
        logprobs: Optional[bool] = None, # If requesting logprobs (check API if bool or object)
        top_logprobs: Optional[int] = None,
        parallel_tool_calls: Optional[bool] = None,
        repetition_penalty: Optional[float] = None,
        stop_token_ids: Optional[Sequence[int]] = None,
        top_k: Optional[int] = None,
        stream_options: Optional[StreamOptions] = None,
        stream_cls: Optional[Type[ChunkModelFactory[ChatCompletionChunk]]] = None,
        **kwargs: Any
    ) -> ChatCompletion: # Return type for non-streaming
        ...
        
    @overload
    def create(
        self,
        *,
        model: str,
        messages: Sequence[MessageParam],
        stream: Literal[True],
        stream_cls: Optional[Type[ChunkModelFactory[ChatCompletionChunk]]] = None,
        # --- Common Optional Parameters ---
        frequency_penalty: Optional[float] = None,
        max_tokens: Optional[int] = None, # Deprecated. Please use max_completion_tokens instead.
        max_completion_tokens: Optional[int] = None,
        n: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        response_format: Optional[ResponseFormat] = None,
        seed: Optional[int] = None,
        stop: Optional[Union[str, Sequence[str]]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        tools: Optional[Sequence[Tool]] = None,
        tool_choice: Optional[Union[Literal["none", "auto"], ToolChoiceObject]] = None,
        user: Optional[str] = None,
        venice_parameters: Optional[VeniceParameters] = None,
        # --- Less Common / Newer Params ---
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        parallel_tool_calls: Optional[bool] = None,
        repetition_penalty: Optional[float] = None,
        stop_token_ids: Optional[Sequence[int]] = None,
        top_k: Optional[int] = None,
        stream_options: Optional[StreamOptions] = None,
        **kwargs: Any
    ) -> Iterator[ChatCompletionChunk]: # Return type for streaming (iterator of dicts)
        ...
 
    def create(
        self,
        *,
        model: str,
        messages: Sequence[MessageParam],
        stream: bool = False,
        stream_cls: Optional[Type[ChunkModelFactory[ChatCompletionChunk]]] = None,
        **kwargs: Any # Catch all other keyword args
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        """
        Create a model response for the given chat conversation.

        This method handles the core functionality of the chat completions API, allowing
        for both synchronous and streaming responses. It sends the provided messages
        and parameters to the Venice AI API and returns either a complete response or
        a stream of partial responses.

        The method automatically formats the request body, applies appropriate defaults,
        and routes the request to either the standard or streaming endpoint based on
        the ``stream`` parameter.

        :param model: ID of the model to use (e.g., ``"venice-1"``, ``"llama-3.3-70b"``).
        :type model: str
        :param messages: Sequence of messages forming the conversation.
        :type messages: Sequence[venice_ai.types.chat.MessageParam]
        :param stream: If ``True``, stream back partial progress. Defaults to ``False``.
            Returns an ``Iterator[ChatCompletionChunk]`` if ``True``, otherwise ``ChatCompletion``.
        :type stream: bool
        :param stream_cls: Optional stream wrapper class for streaming responses. Must conform to the ChunkModelFactory protocol.
        :type stream_cls: Optional[Type[venice_ai.types.chat.ChunkModelFactory[venice_ai.types.chat.ChatCompletionChunk]]]
        :param frequency_penalty: Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far.
        :type frequency_penalty: Optional[float]
        :param max_tokens: Deprecated. Please use ``max_completion_tokens`` instead.
                           The maximum number of tokens that can be generated in the chat completion.
                           The total length of input tokens and generated tokens is limited by the model's context length.
        :type max_tokens: Optional[int]
        :param max_completion_tokens: Maximum number of tokens that can be generated in the chat completion.
        :type max_completion_tokens: Optional[int]
        :param n: Number of chat completion choices to generate for each input message.
        :type n: Optional[int]
        :param presence_penalty: Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far.
        :type presence_penalty: Optional[float]
        :param response_format: Specifies the format that the model must output (e.g., for JSON mode).
        :type response_format: Optional[venice_ai.types.chat.ResponseFormat]
        :param seed: Random seed for reproducible outputs.
        :type seed: Optional[int]
        :param stop: Up to 4 sequences where the API will stop generating further tokens.
        :type stop: Optional[Union[str, Sequence[str]]]
        :param temperature: Sampling temperature between 0.0 and 2.0. Higher values make output more random, lower values more focused and deterministic. Defaults to 0.7.
        :type temperature: Optional[float]
        :param top_p: Nucleus sampling parameter between 0.0 and 1.0. Defaults to 1.0.
        :type top_p: Optional[float]
        :param tools: List of tools the model may call.
        :type tools: Optional[Sequence[venice_ai.types.chat.Tool]]
        :param tool_choice: Controls which (if any) tool is called by the model. Can be ``"none"``, ``"auto"``, or a specific tool.
        :type tool_choice: Optional[Union[Literal["none", "auto"], venice_ai.types.chat.ToolChoiceObject]]
        :param user: Unique identifier representing your end-user (discarded by API but supported for OpenAI compatibility).
        :type user: Optional[str]
        :param venice_parameters: Venice-specific parameters for fine-tuning model behavior.
        :type venice_parameters: Optional[venice_ai.types.chat.VeniceParameters]
        :param logprobs: Whether to return log probabilities of the output tokens.
        :type logprobs: Optional[bool]
        :param top_logprobs: Number of most likely tokens to return at each token position if ``logprobs`` is ``True``.
        :type top_logprobs: Optional[int]
        :param parallel_tool_calls: Whether to enable parallel function calling during tool use.
        :type parallel_tool_calls: Optional[bool]
        :param repetition_penalty: Penalty for token repetition.
        :type repetition_penalty: Optional[float]
        :param stop_token_ids: List of token IDs at which to stop generation.
        :type stop_token_ids: Optional[Sequence[int]]
        :param top_k: Number of highest probability vocabulary tokens to keep for top-k-filtering.
        :type top_k: Optional[int]
        :param stream_options: Additional options for controlling streaming behavior.
        :type stream_options: Optional[venice_ai.types.chat.StreamOptions]
        :param logit_bias: Modify the likelihood of specified tokens appearing in the completion. Accepts a JSON object that maps tokens (specified by their token ID in the tokenizer) to an associated bias value from -100 to 100.
        :type logit_bias: Optional[Dict[str, int]]
        :param kwargs: Additional keyword arguments.

        :return: A :class:`~venice_ai.types.chat.ChatCompletion` if ``stream`` is ``False``,
            otherwise an ``Iterator`` of :class:`~venice_ai.types.chat.ChatCompletionChunk`.
        :rtype: Union[venice_ai.types.chat.ChatCompletion, Iterator[venice_ai.types.chat.ChatCompletionChunk]]

        :raises venice_ai.exceptions.InvalidRequestError: If parameters are invalid or malformed.
        :raises venice_ai.exceptions.AuthenticationError: If the API key is invalid or missing.
        :raises venice_ai.exceptions.PermissionDeniedError: If access is denied to the requested model or feature.
        :raises venice_ai.exceptions.NotFoundError: If the model or resource is not found.
        :raises venice_ai.exceptions.RateLimitError: If rate limits are exceeded for the account.
        :raises venice_ai.exceptions.APIError: For other API-related errors not covered by specific exceptions.
            
        Example:

            .. code-block:: python

               # Non-streaming usage with system and user messages
               from venice_ai import VeniceClient
               client = VeniceClient(api_key="your-api-key")
               response = client.chat.completions.create(
                   model="llama-3.3-70b",
                   messages=[
                       {"role": "system", "content": "You are a helpful assistant specializing in Python."},
                       {"role": "user", "content": "Write a function to calculate the Fibonacci sequence."}
                   ],
                   temperature=0.3  # More deterministic/focused response
               )
               print(response["choices"][0]["message"]["content"])
               
               # Streaming usage with progress display
               for chunk in client.chat.completions.create(
                   model="venice-1",
                   messages=[{"role": "user", "content": "Explain quantum computing briefly."}],
                   stream=True,
                   max_completion_tokens=250  # Limit response length
               ):
                   content = chunk["choices"][0]["delta"].get("content", "")
                   if content:
                       print(content, end="", flush=True)
               
               # Using tools/function calling
               response = client.chat.completions.create(
                   model="llama-3.3-70b",
                   messages=[{"role": "user", "content": "What's the weather in New York?"}],
                   tools=[{
                       "type": "function",
                       "function": {
                           "name": "get_weather",
                           "description": "Get current weather for a location",
                           "parameters": {
                               "type": "object",
                               "properties": {
                                   "location": {"type": "string", "description": "City name"},
                                   "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                               },
                               "required": ["location"]
                           }
                       }
                   }]
               )
        """
        # Logic to handle max_tokens deprecation and precedence
        actual_max_completion_tokens: Optional[int] = kwargs.pop("max_completion_tokens", None)
        deprecated_max_tokens: Optional[int] = kwargs.pop("max_tokens", None)

        if actual_max_completion_tokens is not None:
            kwargs["max_completion_tokens"] = actual_max_completion_tokens
            if deprecated_max_tokens is not None:
                warnings.warn(
                    "Both `max_tokens` and `max_completion_tokens` were provided. "
                    "`max_tokens` is deprecated and will be ignored in favor of `max_completion_tokens`.",
                    DeprecationWarning,
                    stacklevel=2,
                )
        elif deprecated_max_tokens is not None:
            warnings.warn(
                "The `max_tokens` parameter is deprecated. Please use `max_completion_tokens` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            kwargs["max_completion_tokens"] = deprecated_max_tokens

        # Construct request body, filtering out None values from kwargs
        body: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,  # Set based on the stream parameter
        }

        # Add optional parameters from kwargs if they are not None
        for key, value in kwargs.items():
            if value is not None:
                body[key] = value

        # Handle specific naming or structuring if needed
        # e.g. if venice_parameters needs special handling

        if stream:
            user_provided_stream_cls = stream_cls
            effective_stream_cls: Any = Stream # Default to our Stream wrapper

            if user_provided_stream_cls is not None:
                # Check if the user_provided_stream_cls is a class and callable (basic check)
                if inspect.isclass(user_provided_stream_cls):
                    try:
                        # First check if it's a subclass of our known stream types
                        if issubclass(user_provided_stream_cls, (Stream, AsyncStream)):
                            effective_stream_cls = cast(Any, user_provided_stream_cls)
                        else:
                            # For custom classes, check if they have the proper interface
                            # They should have __init__ with iterator and client params, and __iter__ method
                            sig = inspect.signature(user_provided_stream_cls.__init__)
                            params = list(sig.parameters.keys())
                            has_proper_signature = len(params) >= 3 or 'client' in params
                            has_iter_method = hasattr(user_provided_stream_cls, '__iter__')
                            
                            if has_proper_signature and has_iter_method:
                                effective_stream_cls = cast(Any, user_provided_stream_cls)
                            # else: incompatible, use default
                    except (TypeError, ValueError):
                        # If we can't inspect the signature, fall back to default
                        pass # effective_stream_cls remains Stream
                # else: it's not a class, so use default.

            raw_iterator: Iterator[ChatCompletionChunk] = self._client._stream_request(
                method="POST",
                path="chat/completions",
                json_data=body,
                cast_to=ChatCompletionChunk
            )
            return effective_stream_cls(raw_iterator, client=self._client)
        else:
            # Use regular post method for non-streaming responses
            response = self._client.post("chat/completions", json_data=body, cast_to=ChatCompletion)
            # The response is now cast by the client to ChatCompletion
            return response


# --- Async Resource Class ---

class AsyncChatCompletions(AsyncAPIResource):
    """
    Provides access to asynchronous chat completion operations.

    This class manages asynchronous chat completion operations with Venice AI models,
    supporting both standard (non-streaming) and streaming response formats. It serves
    as the primary interface for chat-based interactions with Venice AI language models
    in asynchronous contexts.

    The class handles parameter validation, request formation, and response parsing
    for asynchronous chat completion requests.

    :param _client: The client instance used to make API requests.
    :type _client: venice_ai._async_client.AsyncVeniceClient

    Example:

        .. code-block:: python

           from venice_ai import AsyncVeniceClient
           import asyncio
           
           async def main():
               # Initialize the async client
               client = AsyncVeniceClient(api_key="your-api-key")
               
               # Create a chat completion asynchronously
               response = await client.chat.completions.create(
                   model="venice-1",
                   messages=[
                       {"role": "system", "content": "You are a helpful assistant."},
                       {"role": "user", "content": "Tell me about Venice AI."}
                   ]
               )
               
               # Access the response content
               print(response["choices"][0]["message"]["content"])
               
           # Run the async function
           asyncio.run(main())
    """

    @overload
    async def create(
        self,
        *,
        model: str,
        messages: Sequence[MessageParam],
        stream: Literal[False] = False, # Explicit non-streaming case
        # --- Common Optional Parameters ---
        frequency_penalty: Optional[float] = None,
        max_tokens: Optional[int] = None, # Deprecated. Please use max_completion_tokens instead.
        max_completion_tokens: Optional[int] = None,
        n: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        response_format: Optional[ResponseFormat] = None,
        seed: Optional[int] = None,
        stop: Optional[Union[str, Sequence[str]]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        tools: Optional[Sequence[Tool]] = None,
        tool_choice: Optional[Union[Literal["none", "auto"], ToolChoiceObject]] = None,
        user: Optional[str] = None, # Discarded but supported for OpenAI compat
        venice_parameters: Optional[VeniceParameters] = None,
        # --- Less Common / Newer Params from Docs ---
        logprobs: Optional[bool] = None, # If requesting logprobs (check API if bool or object)
        top_logprobs: Optional[int] = None,
        parallel_tool_calls: Optional[bool] = None,
        repetition_penalty: Optional[float] = None,
        stop_token_ids: Optional[Sequence[int]] = None,
        top_k: Optional[int] = None,
        stream_options: Optional[StreamOptions] = None,
        stream_cls: Optional[Type[ChunkModelFactory[ChatCompletionChunk]]] = None,
        # min_temp, max_temp - Check if these are standard or venice specific
        # stream_options - Handled by stream=True overload

        # Extra arguments are ignored for now, could add **kwargs
    ) -> ChatCompletion: # Return type for non-streaming
        ...
        
    @overload
    async def create(
        self,
        *,
        model: str,
        messages: Sequence[MessageParam],
        stream: Literal[True],
        stream_cls: Optional[Type[ChunkModelFactory[ChatCompletionChunk]]] = None,
        # --- Common Optional Parameters ---
        frequency_penalty: Optional[float] = None,
        max_tokens: Optional[int] = None, # Deprecated. Please use max_completion_tokens instead.
        max_completion_tokens: Optional[int] = None,
        n: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        response_format: Optional[ResponseFormat] = None,
        seed: Optional[int] = None,
        stop: Optional[Union[str, Sequence[str]]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        tools: Optional[Sequence[Tool]] = None,
        tool_choice: Optional[Union[Literal["none", "auto"], ToolChoiceObject]] = None,
        user: Optional[str] = None,
        venice_parameters: Optional[VeniceParameters] = None,
        # --- Less Common / Newer Params ---
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        parallel_tool_calls: Optional[bool] = None,
        repetition_penalty: Optional[float] = None,
        stop_token_ids: Optional[Sequence[int]] = None,
        top_k: Optional[int] = None,
        stream_options: Optional[StreamOptions] = None,
    ) -> AsyncIterator[ChatCompletionChunk]: # Return type for streaming (async iterator of dicts)
        ...
 
    async def create(
        self,
        *,
        model: str,
        messages: Sequence[MessageParam],
        stream: bool = False,
        stream_cls: Optional[Type[ChunkModelFactory[ChatCompletionChunk]]] = None,
        **kwargs: Any # Catch all other keyword args
    ) -> Union[ChatCompletion, AsyncIterator[ChatCompletionChunk]]:
        """
        Create a model response for the given chat conversation asynchronously.

        This method handles the core functionality of the chat completions API, allowing
        for both synchronous and streaming responses in async contexts. It sends the provided
        messages and parameters to the Venice AI API and returns either a complete response or
        a stream of partial responses.

        The method automatically formats the request body, applies appropriate defaults,
        and routes the request to either the standard or streaming endpoint based on
        the ``stream`` parameter.

        :param model: ID of the model to use (e.g., ``"venice-1"``, ``"llama-3.3-70b"``).
        :type model: str
        :param messages: Sequence of messages forming the conversation.
        :type messages: Sequence[venice_ai.types.chat.MessageParam]
        :param stream: If ``True``, stream back partial progress. Defaults to ``False``.
            Returns an ``AsyncIterator[ChatCompletionChunk]`` if ``True``, otherwise ``ChatCompletion``.
        :type stream: bool
        :param stream_cls: Optional stream wrapper class for streaming responses. Must conform to the ChunkModelFactory protocol.
        :type stream_cls: Optional[Type[venice_ai.types.chat.ChunkModelFactory[venice_ai.types.chat.ChatCompletionChunk]]]
        :param frequency_penalty: Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far.
        :type frequency_penalty: Optional[float]
        :param max_tokens: Deprecated. Please use ``max_completion_tokens`` instead.
                           The maximum number of tokens that can be generated in the chat completion.
                           The total length of input tokens and generated tokens is limited by the model's context length.
        :type max_tokens: Optional[int]
        :param max_completion_tokens: Maximum number of tokens that can be generated in the chat completion.
        :type max_completion_tokens: Optional[int]
        :param n: Number of chat completion choices to generate for each input message.
        :type n: Optional[int]
        :param presence_penalty: Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far.
        :type presence_penalty: Optional[float]
        :param response_format: Specifies the format that the model must output (e.g., for JSON mode).
        :type response_format: Optional[venice_ai.types.chat.ResponseFormat]
        :param seed: Random seed for reproducible outputs.
        :type seed: Optional[int]
        :param stop: Up to 4 sequences where the API will stop generating further tokens.
        :type stop: Optional[Union[str, Sequence[str]]]
        :param temperature: Sampling temperature between 0.0 and 2.0. Higher values make output more random, lower values more focused and deterministic. Defaults to 0.7.
        :type temperature: Optional[float]
        :param top_p: Nucleus sampling parameter between 0.0 and 1.0. Defaults to 1.0.
        :type top_p: Optional[float]
        :param tools: List of tools the model may call.
        :type tools: Optional[Sequence[venice_ai.types.chat.Tool]]
        :param tool_choice: Controls which (if any) tool is called by the model. Can be ``"none"``, ``"auto"``, or a specific tool.
        :type tool_choice: Optional[Union[Literal["none", "auto"], venice_ai.types.chat.ToolChoiceObject]]
        :param user: Unique identifier representing your end-user (discarded by API but supported for OpenAI compatibility).
        :type user: Optional[str]
        :param venice_parameters: Venice-specific parameters for fine-tuning model behavior.
        :type venice_parameters: Optional[venice_ai.types.chat.VeniceParameters]
        :param logprobs: Whether to return log probabilities of the output tokens.
        :type logprobs: Optional[bool]
        :param top_logprobs: Number of most likely tokens to return at each token position if ``logprobs`` is ``True``.
        :type top_logprobs: Optional[int]
        :param parallel_tool_calls: Whether to enable parallel function calling during tool use.
        :type parallel_tool_calls: Optional[bool]
        :param repetition_penalty: Penalty for token repetition.
        :type repetition_penalty: Optional[float]
        :param stop_token_ids: List of token IDs at which to stop generation.
        :type stop_token_ids: Optional[Sequence[int]]
        :param top_k: Number of highest probability vocabulary tokens to keep for top-k-filtering.
        :type top_k: Optional[int]
        :param stream_options: Additional options for controlling streaming behavior.
        :type stream_options: Optional[venice_ai.types.chat.StreamOptions]
        :param logit_bias: Modify the likelihood of specified tokens appearing in the completion. Accepts a JSON object that maps tokens (specified by their token ID in the tokenizer) to an associated bias value from -100 to 100.
        :type logit_bias: Optional[Dict[str, int]]
        :param kwargs: Additional keyword arguments.

        :return: A :class:`~venice_ai.types.chat.ChatCompletion` if ``stream`` is ``False``,
            otherwise an ``AsyncIterator`` of :class:`~venice_ai.types.chat.ChatCompletionChunk`.
        :rtype: Union[venice_ai.types.chat.ChatCompletion, AsyncIterator[venice_ai.types.chat.ChatCompletionChunk]]

        :raises venice_ai.exceptions.InvalidRequestError: If parameters are invalid or malformed.
        :raises venice_ai.exceptions.AuthenticationError: If the API key is invalid or missing.
        :raises venice_ai.exceptions.PermissionDeniedError: If access is denied to the requested model or feature.
        :raises venice_ai.exceptions.NotFoundError: If the model or resource is not found.
        :raises venice_ai.exceptions.RateLimitError: If rate limits are exceeded for the account.
        :raises venice_ai.exceptions.APIError: For other API-related errors not covered by specific exceptions.

        Example:

            .. code-block:: python

               # Non-streaming async usage
               import asyncio
               from venice_ai import AsyncVeniceClient
               
               async def main():
                   client = AsyncVeniceClient(api_key="your-api-key")
                   response = await client.chat.completions.create(
                       model="llama-3.3-70b",
                       messages=[
                           {"role": "system", "content": "You are a helpful assistant."},
                           {"role": "user", "content": "Explain async programming in Python."}
                       ],
                       temperature=0.3
                   )
                   print(response["choices"][0]["message"]["content"])
               
               asyncio.run(main())
               
               # Async streaming usage
               async def stream_example():
                   client = AsyncVeniceClient(api_key="your-api-key")
                   async for chunk in await client.chat.completions.create(
                       model="venice-1",
                       messages=[{"role": "user", "content": "Tell me a story."}],
                       stream=True,
                       max_completion_tokens=200
                   ):
                       content = chunk["choices"][0]["delta"].get("content", "")
                       if content:
                           print(content, end="", flush=True)
               
               asyncio.run(stream_example())
        """
        # Logic to handle max_tokens deprecation and precedence
        actual_max_completion_tokens: Optional[int] = kwargs.pop("max_completion_tokens", None)
        deprecated_max_tokens: Optional[int] = kwargs.pop("max_tokens", None)

        if actual_max_completion_tokens is not None:
            kwargs["max_completion_tokens"] = actual_max_completion_tokens
            if deprecated_max_tokens is not None:
                warnings.warn(
                    "Both `max_tokens` and `max_completion_tokens` were provided. "
                    "`max_tokens` is deprecated and will be ignored in favor of `max_completion_tokens`.",
                    DeprecationWarning,
                    stacklevel=2,
                )
        elif deprecated_max_tokens is not None:
            warnings.warn(
                "The `max_tokens` parameter is deprecated. Please use `max_completion_tokens` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            kwargs["max_completion_tokens"] = deprecated_max_tokens

        # Construct request body, filtering out None values from kwargs
        body: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,  # Set based on the stream parameter
        }

        # Add optional parameters from kwargs if they are not None
        # Exclude 'stream_cls' from being added to the body if it's in kwargs
        # as it's a type and not part of the API request body.
        processed_kwargs = {k: v for k, v in kwargs.items() if k != 'stream_cls'}
        for key, value in processed_kwargs.items():
            if value is not None:
                body[key] = value

        # Handle specific naming or structuring if needed
        # e.g. if venice_parameters needs special handling

        if stream:
            user_provided_stream_cls_async = stream_cls
            effective_stream_cls_async: Any = AsyncStream # Default

            if user_provided_stream_cls_async is not None:
                if inspect.isclass(user_provided_stream_cls_async):
                    try:
                        # First check if it's a subclass of our known stream types
                        if issubclass(user_provided_stream_cls_async, (Stream, AsyncStream)):
                            effective_stream_cls_async = cast(Any, user_provided_stream_cls_async)
                        else:
                            # For custom classes, check if they have the proper interface
                            # They should have __init__ with iterator and client params, and __aiter__ method
                            sig = inspect.signature(user_provided_stream_cls_async.__init__)
                            params = list(sig.parameters.keys())
                            has_proper_signature = len(params) >= 3 or 'client' in params
                            has_aiter_method = hasattr(user_provided_stream_cls_async, '__aiter__')
                            
                            if has_proper_signature and has_aiter_method:
                                effective_stream_cls_async = cast(Any, user_provided_stream_cls_async)
                            # else: incompatible, use default
                    except (TypeError, ValueError):
                        # If we can't inspect the signature, fall back to default
                        pass # effective_stream_cls_async remains AsyncStream
                # else: not a class, use default
            # else: stream_cls is None, use default

            # _stream_request is an async generator function, calling it returns the async generator object.
            raw_iterator: AsyncIterator[ChatCompletionChunk] = self._client._stream_request(
                method="POST",
                path="chat/completions",
                json_data=body,
                cast_to=ChatCompletionChunk
            )
            return effective_stream_cls_async(raw_iterator, client=self._client)
        else:
            # Use regular post method for non-streaming responses
            response = await self._client.post("chat/completions", json_data=body, cast_to=ChatCompletion)
            # The response is now cast by the client to ChatCompletion
            return response