"""
Type definitions for Venice AI Chat Completions API.

This module contains Pydantic models for response objects and TypedDict definitions
for request objects in the Venice AI Chat Completions API, including support for
tools, tool calls, log probabilities, and streaming.
"""

from typing import Optional, List, Dict, Any, Union, Literal, Sequence, Protocol, TypeVar
from typing_extensions import TypedDict, NotRequired
from pydantic import BaseModel, Field

__all__ = [
    "FunctionDefinition", "Tool", "ToolChoiceFunction", "ToolChoiceObject", "ToolChoice",
    "ToolCallFunction", "ToolCall", "MessageParam", "ChatCompletionMessage",
    "ChatCompletionTopLogprob", "ChatCompletionTokenLogprob", "ChatCompletionChoiceLogprobs",
    "UsageData", "ChatCompletionChoice", "ChatCompletion",
    "ChatCompletionChunkToolCallFunction", "ChatCompletionChunkToolCall",
    "ChatCompletionChunkChoiceDelta", "ChatCompletionChunkChoice", "ChatCompletionChunk",
    "StreamOptions", "ResponseFormat", "VeniceParameters", "CreateChatCompletionRequest",
    "ChunkModelFactory", "WebSearchCitation", "VeniceParametersResponse"
]

# --- Protocol Definitions ---

_ChunkModelT = TypeVar("_ChunkModelT", covariant=True)

class ChunkModelFactory(Protocol[_ChunkModelT]):
    """
    A protocol for classes that can be instantiated from keyword arguments.
    Used to define the expected interface for `stream_cls` in chat completions,
    where the class's __init__ method should accept ``**data``.
    """
    def __init__(self, **data: Any) -> None:
        ...


# --- Tool and Function Types (Request-related: TypedDict) ---

class FunctionDefinition(TypedDict):
    """
    Defines the structure and parameters of a function that can be called by the model.
    """
    name: str
    description: NotRequired[str]
    parameters: NotRequired[Dict[str, Any]]


class Tool(TypedDict):
    """
    Represents a tool that the model can invoke during chat completion.
    """
    type: Literal["function"]
    function: FunctionDefinition
    id: NotRequired[str]


class ToolChoiceFunction(TypedDict):
    """
    Specifies a particular function to be called when using structured tool choice.
    """
    name: str


class ToolChoiceObject(TypedDict):
    """
    Defines the object form of tool choice specification for forcing specific tool usage.
    """
    type: Literal["function"]
    function: ToolChoiceFunction


ToolChoice = Union[Literal["none", "auto"], ToolChoiceObject]


# --- Message and Tool Call Types ---

class ToolCallFunction(BaseModel):
    """
    Contains the details of a function call made by the model. (Response DTO)
    """
    name: str
    arguments: str

class ToolCall(BaseModel):
    """
    Represents a complete tool call made by the model during chat completion. (Response DTO)
    """
    id: str
    type: Literal["function"]
    function: ToolCallFunction


class MessageParam(TypedDict): # Request DTO
    """
    Defines the structure of a message in a chat conversation for requests.
    """
    role: Literal["system", "user", "assistant"]
    content: Union[str, Sequence[Dict[str, Any]], None]


class ChatCompletionMessage(BaseModel): # Response DTO
    """
    Represents a message returned by the model in a chat completion response.
    """
    role: Literal["system", "user", "assistant"]
    content: Union[str, Sequence[Dict[str, Any]], None] = None # Content can be None if tool_calls is present
    tool_calls: Optional[List[ToolCall]] = Field(default=None)
    name: Optional[str] = Field(default=None)
    reasoning_content: Optional[str] = Field(default=None)


# --- Logprobs Types (Response DTOs) ---

class ChatCompletionTopLogprob(BaseModel):
    """
    Represents log probability information for alternative tokens at a specific position.
    """
    token: str
    logprob: float
    bytes: Optional[List[int]] = Field(default=None)


class ChatCompletionTokenLogprob(BaseModel):
    """
    Contains comprehensive log probability information for a single token.
    """
    token: str
    logprob: float
    bytes: Optional[List[int]] = Field(default=None)
    top_logprobs: Optional[List[ChatCompletionTopLogprob]] = Field(default=None)


class ChatCompletionChoiceLogprobs(BaseModel):
    """
    Aggregates log probability information for all tokens in a completion choice.
    """
    content: Optional[List[ChatCompletionTokenLogprob]] = Field(default=None)


# --- Response Types (Response DTOs) ---

class WebSearchCitation(BaseModel):
    """
    Represents a web search citation in the Venice parameters response.
    
    Contains information about web sources cited by the model when web search
    is enabled, including the source URL, title, content snippet, and date.
    """
    title: str
    """The title of the web page or source."""
    url: str
    """The URL of the web source."""
    content: Optional[str] = Field(default=None)
    """A snippet of content from the web source."""
    date: Optional[str] = Field(default=None)
    """The date of the web source in ISO format."""


class VeniceParametersResponse(BaseModel):
    """
    Venice-specific parameters included in the chat completion response.
    
    Contains information about Venice-specific features that were used or
    configured for the request, including web search settings, character
    information, and thinking/reasoning controls.
    """
    enable_web_search: Literal["auto", "off", "on"]
    """The web search setting that was used for this request."""
    enable_web_citations: bool
    """Whether web citations were enabled for this request."""
    include_venice_system_prompt: bool
    """Whether the Venice system prompt was included."""
    include_search_results_in_stream: bool
    """Whether search results were included in the stream."""
    strip_thinking_response: bool
    """Whether thinking responses were stripped from the output."""
    disable_thinking: bool
    """Whether thinking was disabled for this request."""
    character_slug: Optional[str] = Field(default=None)
    """The character slug used for this request, if any."""
    web_search_citations: List[WebSearchCitation] = Field(default_factory=list)
    """List of web search citations if web search was performed."""


class UsageData(BaseModel):
    """
    Provides token usage statistics for a chat completion request.
    """
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: Optional[Any] = Field(default=None)


class ChatCompletionChoice(BaseModel):
    """
    Represents a single completion choice generated by the model.
    """
    index: int
    message: ChatCompletionMessage
    finish_reason: Optional[Literal["stop", "length", "tool_calls"]] = None
    logprobs: Optional[ChatCompletionChoiceLogprobs] = Field(default=None)
    stop_reason: Optional[str] = Field(default=None)


class ChatCompletion(BaseModel):
    """
    Represents the complete response from a chat completion request.
    """
    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Optional[UsageData] = Field(default=None)
    prompt_logprobs: Optional[Any] = Field(default=None)
    venice_parameters: Optional[VeniceParametersResponse] = Field(default=None)


# --- Streaming Types (Response DTOs) ---

class ChatCompletionChunkToolCallFunction(BaseModel):
    """
    Represents function call details within a streaming chat completion chunk.
    Fields are optional as they arrive incrementally.
    """
    name: Optional[str] = Field(default=None)
    arguments: Optional[str] = Field(default=None)


class ChatCompletionChunkToolCall(BaseModel):
    """
    Represents an incremental tool call within a streaming chat completion chunk.
    Fields are optional as they arrive incrementally.
    """
    id: Optional[str] = Field(default=None) # ID should be present once the tool call starts
    type: Optional[Literal["function"]] = Field(default=None)
    function: Optional[ChatCompletionChunkToolCallFunction] = Field(default=None)
    index: Optional[int] = Field(default=None) # OpenAI includes index for parallel tool calls in chunks


class ChatCompletionChunkChoiceDelta(BaseModel):
    """
    Contains the incremental changes for a choice in a streaming chat completion.
    """
    role: Optional[Literal["system", "user", "assistant", "tool"]] = Field(default=None) # Added tool role
    content: Optional[str] = Field(default=None)
    tool_calls: Optional[List[ChatCompletionChunkToolCall]] = Field(default=None)


class ChatCompletionChunkChoice(BaseModel):
    """
    Represents a single choice within a streaming chat completion chunk.
    """
    index: int
    delta: ChatCompletionChunkChoiceDelta
    finish_reason: Optional[Literal["stop", "length", "tool_calls"]] = None
    logprobs: Optional[ChatCompletionChoiceLogprobs] = Field(default=None) # Typically not in chunks, but for completeness


class ChatCompletionChunk(BaseModel):
    """
    Represents a single chunk in a streaming chat completion response.
    """
    id: str
    object: Literal["chat.completion.chunk"]
    created: int
    model: str
    choices: List[ChatCompletionChunkChoice]
    usage: Optional[UsageData] = Field(default=None) # Only if stream_options.include_usage is true
    system_fingerprint: Optional[str] = Field(default=None)


# --- Request Parameter Types ---

class StreamOptions(TypedDict, total=False):
    """
    Configures the behavior and features of streaming chat completion responses.
    
    This class provides options for controlling how streaming responses are
    delivered, including whether to include usage statistics in the final
    chunk. Used in chat completion requests when streaming is enabled to
    customize the streaming behavior according to client needs.
    
    Enables fine-grained control over streaming features, allowing clients
    to optimize for their specific use cases and processing requirements.
    """
    
    include_usage: bool
    """If set, an additional chunk will be streamed before the ``data: [DONE]`` message.
    This chunk will contain a ``usage`` field, providing token usage information for the entire request."""


class ResponseFormat(TypedDict, total=False):
    """
    Specifies the desired output format for the model's response.
    
    This class enables structured output generation by constraining the model
    to produce responses in specific formats, particularly JSON. Supports both
    general JSON mode and schema-constrained JSON generation for applications
    requiring structured data output.
    
    Used in chat completion requests to ensure the model's response conforms
    to expected formats, enabling reliable parsing and processing of model
    output in structured applications.
    """
    
    type: Literal["json_object", "json_schema"]
    """Must be one of ``json_object`` or ``json_schema``. Setting to ``json_object`` enables JSON mode,
    directing the model to generate a valid JSON object. Setting to ``json_schema`` also enables JSON mode
    and additionally requires the model to generate a JSON object that conforms to the provided JSON schema."""
    json_schema: NotRequired[Dict[str, Any]]
    """Optional. A JSON schema object that the model's output must adhere to.
    Only used if ``type`` is ``json_schema``."""


class VeniceParameters(TypedDict, total=False):
    """
    Contains Venice-specific parameters for customizing chat completion behavior.
    
    This class provides access to Venice AI's unique features and capabilities,
    including character personas, web search integration, and system prompt
    customization. These parameters extend the standard chat completion API
    with Venice-specific functionality.
    
    Used in chat completion requests to leverage Venice AI's distinctive
    features, enabling enhanced conversational experiences and specialized
    capabilities not available in standard chat completion APIs.
    """
    
    include_venice_system_prompt: bool
    """Optional. If ``true`` (default), the default Venice system prompt will be included.
    Set to ``false`` to exclude it and use only the provided messages."""
    character_slug: str
    """Optional. The slug of a specific character to use for the completion.
    This will influence the model's persona, response style, and behavior patterns."""
    enable_web_search: Literal["on", "off", "auto"]
    """Optional. Controls whether the model can perform web searches to enhance responses.
    ``on`` always enables search, ``off`` disables it completely, ``auto`` (default) lets the model decide based on context."""
    strip_thinking_response: bool
    """Optional. Strip ``<think></think>`` blocks from the response. Applicable only to reasoning/thinking models."""
    disable_thinking: bool
    """Optional. On supported reasoning models, will disable thinking and strip the ``<think></think>`` blocks from the response."""
    enable_web_citations: bool
    """Optional. When web search is enabled, this will request that the LLM cite its sources using a ``[REF]0[/REF]`` format."""
    include_search_results_in_stream: bool
    """Optional. Experimental feature. When set to true, the LLM will include search results in the first emitted chunk."""


class CreateChatCompletionRequest(TypedDict):
    """
    Defines the complete request structure for creating a chat completion.
    
    This class encapsulates all parameters and options available for chat
    completion requests, including conversation messages, model selection,
    generation parameters, tool specifications, and Venice-specific features.
    
    Used as the primary input type for chat completion endpoints, providing
    comprehensive control over model behavior, output format, tool usage,
    and specialized features. Supports both streaming and non-streaming
    completions with extensive customization options.
    """
    
    messages: Sequence[MessageParam]
    """A list of messages comprising the conversation so far."""
    model: str
    """ID of the model to use. See the model endpoint compatibility table for details on which models support this endpoint."""
    frequency_penalty: NotRequired[float]
    """Optional. Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim."""
    logit_bias: NotRequired[Dict[str, int]]
    """Optional. Modify the likelihood of specified tokens appearing in the completion. Accepts a JSON object that maps tokens (specified by their token ID in the tokenizer) to an associated bias value from -100 to 100."""
    logprobs: NotRequired[bool]
    """Optional. Whether to return log probabilities of the output tokens, which appear in the ``logprobs`` property of the ``choice`` object. Defaults to ``false``."""
    top_logprobs: NotRequired[int]
    """Optional. An integer between 0 and 5 specifying the number of most likely tokens to return at each token position, each with an associated log probability. ``logprobs`` must be set to ``true`` if this parameter is used."""
    max_tokens: NotRequired[int]
    """Optional. Deprecated. Please use max_completion_tokens instead. The maximum number of tokens to generate in the chat completion. The total length of input tokens and generated tokens is limited by the model's context length."""
    max_completion_tokens: NotRequired[int]
    """Optional. The maximum number of tokens that can be generated in the chat completion. The total length of input tokens and generated tokens is limited by the model's context length."""
    n: NotRequired[int]
    """Optional. How many chat completion choices to generate for each input message. Note that you will be charged for the number of generated tokens across all of the choices. Defaults to 1."""
    presence_penalty: NotRequired[float]
    """Optional. Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics."""
    response_format: NotRequired[ResponseFormat]
    """Optional. An object specifying the format that the model must output. Setting to ``{ "type": "json_object" }`` enables JSON mode, which guarantees the message the model generates is valid JSON."""
    seed: NotRequired[int]
    """Optional. This feature is in Beta. If specified, our system will make a best effort to sample deterministically, such that repeated requests with the same ``seed`` and parameters should return the same result."""
    stop: NotRequired[Union[str, List[str]]]
    """Optional. Up to 4 sequences where the API will stop generating further tokens."""
    stream: NotRequired[bool]
    """Optional. If set, partial message deltas will be sent, like in ChatGPT. Tokens will be sent as data-only server-sent events as they become available, with the stream terminated by a ``data: [DONE]`` message. Defaults to ``false``."""
    stream_options: NotRequired[StreamOptions]
    """Optional. Options for streaming response. Only used if ``stream`` is ``true``."""
    temperature: NotRequired[float]
    """Optional. What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic. Defaults to 0.7."""
    top_p: NotRequired[float]
    """Optional. An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered. Defaults to 1."""
    tools: NotRequired[List[Tool]]
    """Optional. A list of tools the model may call. Currently, only functions are supported as a tool. Use this to provide a list of functions the model may generate JSON inputs for."""
    tool_choice: NotRequired[ToolChoice]
    """Optional. Controls which (if any) function is called by the model. ``none`` means the model will not call a function and instead generates a message. ``auto`` means the model can pick between generating a message or calling a function. Specifying a particular function via ``{"type": "function", "function": {"name": "my_function"}}`` forces the model to call that function."""
    user: NotRequired[str]
    """Optional. A unique identifier representing your end-user, which can help Venice monitor and detect abuse."""
    venice_parameters: NotRequired[VeniceParameters]
    """Optional. Venice-specific parameters to extend or modify API behavior."""
    parallel_tool_calls: NotRequired[bool]
    """Optional. Whether to enable parallel function calling during tool use."""
    repetition_penalty: NotRequired[float]
    """Optional. Penalty for token repetition."""
    stop_token_ids: NotRequired[List[int]]
    """Optional. List of token IDs at which to stop generation."""
    top_k: NotRequired[int]
    """Optional. Number of highest probability vocabulary tokens to keep for top-k-filtering."""
    max_temp: NotRequired[float]
    """Optional. Maximum temperature value for dynamic temperature scaling. Range: 0 <= x <= 2."""
    min_p: NotRequired[float]
    """Optional. Sets a minimum probability threshold for token selection. Tokens with probabilities below this value are filtered out. Range: 0 <= x <= 1."""
    min_temp: NotRequired[float]
    """Optional. Minimum temperature value for dynamic temperature scaling. Range: 0 <= x <= 2."""