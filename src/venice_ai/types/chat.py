"""
Type definitions for Venice AI Chat Completions API.

This module contains TypedDict definitions for request and response objects
in the Venice AI Chat Completions API, including support for tools, tool calls,
log probabilities, and streaming.
"""

from typing import Optional, List, Dict, Any, Union, Literal, Sequence, Protocol, TypeVar
from typing_extensions import TypedDict, NotRequired

__all__ = [
    "FunctionDefinition", "Tool", "ToolChoiceFunction", "ToolChoiceObject", "ToolChoice",
    "ToolCallFunction", "ToolCall", "MessageParam", "ChatCompletionMessage",
    "ChatCompletionTopLogprob", "ChatCompletionTokenLogprob", "ChatCompletionChoiceLogprobs",
    "UsageData", "ChatCompletionChoice", "ChatCompletion",
    "ChatCompletionChunkToolCallFunction", "ChatCompletionChunkToolCall",
    "ChatCompletionChunkChoiceDelta", "ChatCompletionChunkChoice", "ChatCompletionChunk",
    "StreamOptions", "ResponseFormat", "VeniceParameters", "CreateChatCompletionRequest",
    "ChunkModelFactory"
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


# --- Tool and Function Types ---

class FunctionDefinition(TypedDict):
    """
    Defines the structure and parameters of a function that can be called by the model.
    
    This class represents the schema for function definitions used in tool calling,
    providing the model with information about available functions, their parameters,
    and descriptions. Used as part of the :class:`Tool` definition in chat completion
    requests to enable function calling capabilities.
    
    The function definition follows JSON Schema conventions for parameter specification,
    allowing the model to understand the expected input format and generate appropriate
    function calls during conversation.
    """
    
    name: str
    """The name of the function to be called."""
    description: NotRequired[str]
    """Optional. A description of what the function does."""
    parameters: NotRequired[Dict[str, Any]]
    """Optional. The parameters the function accepts, described as a JSON Schema object."""


class Tool(TypedDict):
    """
    Represents a tool that the model can invoke during chat completion.
    
    This class defines the structure for tools available to the model, currently
    supporting function-type tools. Used in chat completion requests to specify
    which functions the model can call to extend its capabilities beyond text
    generation, such as retrieving information, performing calculations, or
    interacting with external systems.
    
    Tools are provided to the model via the ``tools`` parameter in chat completion
    requests and can be controlled using the ``tool_choice`` parameter.
    """
    
    type: Literal["function"]
    """The type of the tool. Currently, only "function" is supported."""
    function: FunctionDefinition
    """The definition of the function."""
    id: NotRequired[str]
    """Optional. A unique identifier for the tool."""


class ToolChoiceFunction(TypedDict):
    """
    Specifies a particular function to be called when using structured tool choice.
    
    This class is used within :class:`ToolChoiceObject` to force the model to call
    a specific function rather than allowing it to choose between available tools
    or generating a regular text response. Provides precise control over model
    behavior when function calling is required.
    """
    
    name: str
    """The name of the function."""


class ToolChoiceObject(TypedDict):
    """
    Defines the object form of tool choice specification for forcing specific tool usage.
    
    This class represents the structured way to specify that the model must call
    a particular tool, rather than using string literals like "auto" or "none".
    Used in chat completion requests when you need to ensure the model calls a
    specific function rather than generating a text response or choosing from
    multiple available tools.
    """
    
    type: Literal["function"]
    """The type of the tool choice. Must be "function"."""
    function: ToolChoiceFunction
    """The specific function to force the model to call."""


# ToolChoice can be a string literal or an object
ToolChoice = Union[Literal["none", "auto"], ToolChoiceObject]


# --- Message and Tool Call Types ---

class ToolCallFunction(TypedDict):
    """
    Contains the details of a function call made by the model.
    
    This class represents the specific function that was called by the model,
    including the function name and the arguments provided. The arguments are
    serialized as a JSON string and need to be parsed by the client application
    to extract the actual parameter values for function execution.
    
    Used within :class:`ToolCall` to provide complete information about model-
    generated function calls in chat completion responses.
    """
    
    name: str
    """The name of the function that was called."""
    arguments: str
    """The arguments to call the function with, as a JSON string."""


class ToolCall(TypedDict):
    """
    Represents a complete tool call made by the model during chat completion.
    
    This class encapsulates all information about a tool invocation, including
    a unique identifier, the tool type, and the specific function details.
    Appears in chat completion responses when the model decides to call a tool
    rather than generate text content.
    
    Tool calls can be used by client applications to execute the requested
    functions and provide results back to the model in subsequent messages.
    """
    
    id: str
    """The ID of the tool call."""
    type: Literal["function"]
    """The type of the tool called. Currently, only "function" is supported."""
    function: ToolCallFunction
    """The details of the function call."""


class MessageParam(TypedDict):
    """
    Defines the structure of a message in a chat conversation.
    
    This class represents a single message within a chat completion request,
    supporting different roles (system, user, assistant) and various content
    types. Used to build conversation history and provide context to the model
    for generating appropriate responses.
    
    The content field supports multiple formats: plain text strings for simple
    messages, structured content blocks for multimodal inputs (such as images),
    and None for special cases like tool response messages.
    """
    
    role: Literal["system", "user", "assistant"]
    """The role of the author of this message. 'system' sets the assistant's behavior, 'user' represents the human input, 'assistant' represents the AI's response."""
    content: Union[str, Sequence[Dict[str, Any]], None]
    """The contents of the message. Can be a string for text, a list of content blocks for multimodal inputs (such as images), or None for special cases like tool response messages."""


class ChatCompletionMessage(MessageParam):
    """
    Represents a message returned by the model in a chat completion response.
    
    This class extends :class:`MessageParam` to include additional fields that
    may be present in model-generated messages, particularly tool calls. Used
    in chat completion responses to represent the model's output, which may
    include both text content and function calls.
    
    When the model generates tool calls, the content may be None and the tool
    calls will be specified in the ``tool_calls`` field.
    """
    
    tool_calls: NotRequired[List[ToolCall]]
    """Optional. A list of tool calls generated by the model, if any."""


# --- Logprobs Types ---

class ChatCompletionTopLogprob(TypedDict):
    """
    Represents log probability information for alternative tokens at a specific position.
    
    This class provides detailed information about token alternatives that the model
    considered at a particular position in the generated text. Used within log
    probability analysis to understand the model's confidence and decision-making
    process during text generation.
    
    Appears in the ``top_logprobs`` field of :class:`ChatCompletionTokenLogprob`
    when detailed probability information is requested via the ``top_logprobs``
    parameter in chat completion requests.
    """
    
    token: str
    """The token."""
    logprob: float
    """The log probability of this token."""
    bytes: NotRequired[List[int]]
    """Optional. A list of integers representing the UTF-8 bytes representation of the token. Useful in instances where characters are represented by multiple tokens."""


class ChatCompletionTokenLogprob(TypedDict):
    """
    Contains comprehensive log probability information for a single token.
    
    This class provides detailed probability information for each token in the
    model's output, including the token itself, its log probability, and
    optionally the most likely alternative tokens at that position. Used for
    analyzing model confidence and understanding the generation process.
    
    Available in chat completion responses when log probabilities are requested
    via the ``logprobs`` parameter, enabling detailed analysis of model behavior
    and uncertainty quantification.
    """
    
    token: str
    """The token."""
    logprob: float
    """The log probability of this token."""
    bytes: NotRequired[List[int]]
    """Optional. A list of integers representing the UTF-8 bytes representation of the token."""
    top_logprobs: NotRequired[List[ChatCompletionTopLogprob]]
    """Optional. A list of the most likely tokens and their log probabilities at this token position."""


class ChatCompletionChoiceLogprobs(TypedDict):
    """
    Aggregates log probability information for all tokens in a completion choice.
    
    This class contains the complete log probability data for a chat completion
    choice, providing token-level probability information for the entire generated
    response. Used for detailed analysis of model confidence and generation
    patterns across the full output.
    
    Appears in :class:`ChatCompletionChoice` when log probabilities are requested,
    enabling comprehensive analysis of the model's decision-making process
    throughout the generation.
    """
    
    content: NotRequired[List[ChatCompletionTokenLogprob]]
    """Optional. A list of log probability information for each token in the generated content."""


# --- Response Types ---

class UsageData(TypedDict):
    """
    Provides token usage statistics for a chat completion request.
    
    This class tracks the computational cost of a chat completion by counting
    tokens used in the prompt, generated in the completion, and the total
    consumption. Essential for monitoring API usage, cost calculation, and
    understanding the efficiency of different prompting strategies.
    
    Appears in chat completion responses and optionally in streaming responses
    when usage tracking is enabled, providing transparency into resource
    consumption for each API call.
    """
    
    prompt_tokens: int
    """Number of tokens in the prompt."""
    completion_tokens: int
    """Number of tokens in the generated completion."""
    total_tokens: int
    """Total number of tokens used in the request (prompt + completion)."""


class ChatCompletionChoice(TypedDict):
    """
    Represents a single completion choice generated by the model.
    
    This class encapsulates one possible response from the model, including the
    generated message, completion metadata, and optional probability information.
    Multiple choices can be generated when the ``n`` parameter is greater than 1,
    allowing comparison of different model outputs for the same input.
    
    Each choice includes information about why the model stopped generating
    (finish reason) and optionally detailed log probability data for analysis
    of model confidence and decision-making.
    """
    
    index: int
    """The index of the choice in the list of choices."""
    message: ChatCompletionMessage
    """A chat completion message generated by the model."""
    finish_reason: Optional[Literal["stop", "length", "tool_calls"]]
    """The reason the model stopped generating tokens. This will be `stop` if the model hit a natural stop point or a provided stop sequence,
    `length` if the maximum number of tokens specified in the request was reached, or `tool_calls` if the model called a tool."""
    logprobs: NotRequired[ChatCompletionChoiceLogprobs]
    """Optional. Log probability information for the choice."""


class ChatCompletion(TypedDict):
    """
    Represents the complete response from a chat completion request.
    
    This class encapsulates the full response from the Venice AI chat completions
    endpoint, including metadata about the completion, all generated choices,
    usage statistics, and optional additional features like web search citations.
    
    Used as the return type for non-streaming chat completion requests, providing
    all information needed to process the model's response, track usage, and
    handle any additional features that were enabled during the request.
    """
    
    id: str
    """A unique identifier for the chat completion."""
    object: Literal["chat.completion"]
    """The object type, which is always `chat.completion`."""
    created: int
    """The Unix timestamp (in seconds) of when the chat completion was created."""
    model: str
    """The model used for the chat completion."""
    choices: List[ChatCompletionChoice]
    """A list of chat completion choices. Can be more than one if `n` is greater than 1."""
    usage: Optional[UsageData]
    """Optional. Usage statistics for the completion request."""
    web_search_citations: NotRequired[List[Any]]
    """Optional. Citations for web search results, if web search was enabled for the request."""
    system_fingerprint: NotRequired[str]
    """Optional. This fingerprint represents the backend configuration that the model runs with.
    You can use this value to track changes in the backend configuration that may impact results."""


# --- Streaming Types ---

class ChatCompletionChunkToolCallFunction(TypedDict, total=False):
    """
    Represents function call details within a streaming chat completion chunk.
    
    This class provides incremental function call information during streaming
    responses, where function arguments may be built up progressively across
    multiple chunks. The arguments field may contain partial JSON strings that
    need to be accumulated and parsed once the function call is complete.
    
    Used within :class:`ChatCompletionChunkToolCall` to provide real-time
    updates about function calls as they are generated by the model during
    streaming responses.
    """
    
    name: str
    """The name of the function."""
    arguments: str
    """The arguments to the function, which may be a partial JSON string during streaming."""


class ChatCompletionChunkToolCall(TypedDict, total=False):
    """
    Represents an incremental tool call within a streaming chat completion chunk.
    
    This class provides progressive updates about tool calls during streaming
    responses, allowing clients to process function calls as they are generated
    rather than waiting for the complete response. Tool call information may
    be spread across multiple chunks and needs to be accumulated.
    
    Used within :class:`ChatCompletionChunkChoiceDelta` to provide real-time
    tool call updates during streaming chat completions.
    """
    
    id: str
    """The ID of the tool call."""
    type: Literal["function"]
    """The type of the tool call. Currently, only `function` is supported."""
    function: ChatCompletionChunkToolCallFunction
    """The details of the function call."""


class ChatCompletionChunkChoiceDelta(TypedDict, total=False):
    """
    Contains the incremental changes for a choice in a streaming chat completion.
    
    This class represents the delta (incremental update) for a single choice
    during streaming responses, containing new content, role information, and
    tool call updates. Each chunk provides only the new information since the
    last chunk, requiring accumulation to build the complete response.
    
    Used within :class:`ChatCompletionChunkChoice` to provide progressive
    updates during streaming chat completions, enabling real-time processing
    of model output as it is generated.
    """
    
    role: Literal["assistant"]
    """The role of the author of this message, typically `assistant`."""
    content: Optional[str]
    """Optional. The incremental content of the delta message."""
    tool_calls: NotRequired[List[ChatCompletionChunkToolCall]]
    """Optional. A list of tool calls made by the model, if any. These are incremental during streaming."""


class ChatCompletionChunkChoice(TypedDict):
    """
    Represents a single choice within a streaming chat completion chunk.
    
    This class encapsulates one choice's incremental updates during streaming
    responses, including the delta changes, completion metadata, and optional
    log probability information. Multiple choices may be present when the ``n``
    parameter is greater than 1.
    
    Each chunk choice provides incremental updates that must be accumulated
    with previous chunks to build the complete response for that choice.
    """
    
    index: int
    """The index of the choice in the list of choices."""
    delta: ChatCompletionChunkChoiceDelta
    """The incremental changes (delta) to the message content or tool calls."""
    finish_reason: Optional[Literal["stop", "length", "tool_calls"]]
    """The reason the model stopped generating tokens for this choice.
    This will be `stop` if the model hit a natural stop point or a provided stop sequence,
    `length` if the maximum number of tokens specified in the request was reached, or `tool_calls` if the model called a tool."""
    logprobs: NotRequired[ChatCompletionChoiceLogprobs]
    """Optional. Log probability information for the choice."""


class ChatCompletionChunk(TypedDict):
    """
    Represents a single chunk in a streaming chat completion response.
    
    This class encapsulates one incremental update in a streaming chat completion,
    containing delta information for all choices, metadata about the completion,
    and optional usage statistics. Streaming responses consist of multiple chunks
    that must be processed sequentially to build the complete response.
    
    Used as the unit of data in streaming chat completions, enabling real-time
    processing of model output as it is generated, with each chunk providing
    incremental updates to the overall response.
    """
    
    id: str
    """A unique identifier for the chat completion chunk."""
    object: Literal["chat.completion.chunk"]
    """The object type, which is always `chat.completion.chunk`."""
    created: int
    """The Unix timestamp (in seconds) of when the chat completion chunk was created."""
    model: str
    """The model used for the chat completion."""
    choices: List[ChatCompletionChunkChoice]
    """A list of chat completion choices. Can be more than one if `n` is greater than 1."""
    usage: NotRequired[UsageData]
    """Optional. An object describing the usage statistics for the completion request.
    Only present if `stream_options.include_usage` is set to `true`."""


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