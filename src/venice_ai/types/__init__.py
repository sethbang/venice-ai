"""Venice AI Types Package.

This package contains Pydantic models representing data structures used for API
requests and responses in the venice-ai library. It serves as the public interface
for type definitions, providing convenient access to models from various sub-modules.

The package includes types for:

* **Chat completions**: Message parameters, completion responses, streaming chunks,
  tool calls, and related chat functionality
* **Models**: Model specifications, capabilities, constraints, and pricing information
* **Images**: Image generation requests and responses
* **Audio**: Audio processing and transcription types
* **Embeddings**: Text embedding requests and responses
* **API Keys**: API key management structures
* **Billing**: Billing and usage tracking types
* **Characters**: Character-based AI interaction models

All types are designed to work seamlessly with the Venice.ai API and provide
comprehensive type safety and validation through Pydantic.
"""

# Types package for Venice AI
from .chat import (
    ChatCompletion,
    ChatCompletionChoice,
    ChatCompletionChoiceLogprobs,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkChoiceDelta,
    ChatCompletionChunkToolCall,
    ChatCompletionChunkToolCallFunction,
    ChatCompletionMessage,
    ChatCompletionTokenLogprob,
    ChatCompletionTopLogprob,
    CreateChatCompletionRequest,
    FunctionDefinition,
    MessageParam,
    ResponseFormat,
    StreamOptions,
    Tool,
    ToolCall,
    ToolCallFunction,
    ToolChoice,
    ToolChoiceFunction,
    ToolChoiceObject,
    UsageData,
    VeniceParameters,
)
from .models import (
    ModelPricing,
    ModelCapabilities,
    ModelConstraintsTemperature,
    ModelConstraintsTopP,
    ModelConstraints,
    ModelSpec,
    Model,
    ModelList,
    ModelType,
)
from .characters import (
    Character,
    CharacterList,
)
from .image import (
    GenerateImageRequest,
    SimpleGenerateImageRequest,
    UpscaleImageRequest,
    TimingInfo,
    ImageResponse,
    ImageDataItem,
    SimpleImageResponse,
    ImageStyleList,
    ImageStyleEnum,
)
from .api_keys import (
    ConsumptionLimit,
    ApiKeyUsage,
    ApiKey,
    ApiKeyCreateRequest,
    ApiKeyCreateResponse,
    ApiKeyList,
    ApiKeyRateLimitItem,
    ApiTier,
    Balances,
    RateLimitInfo,
    RateLimitLog,
    RateLimitLogList,
    ApiKeyGenerateWeb3KeyGetResponse,
    ApiKeyGenerateWeb3KeyCreateRequest,
    ApiKeyGenerateWeb3KeyCreateResponse,
)
from .audio import (
    Voice,
    ResponseFormat as AudioResponseFormat,
    CreateSpeechRequest,
    VoiceDetail,
    VoiceList,
)
from .embeddings import (
    CreateEmbeddingRequest,
    Embedding,
    EmbeddingUsage,
    EmbeddingList,
)
from .billing import (
    InferenceDetails,
    BillingUsageEntry,
    BillingUsagePagination,
    BillingUsageResponse,
    BillingUsageRequestParams,
    BillingFormatEnum,
)

__all__ = [
    "ChatCompletion",
    "ChatCompletionChoice",
    "ChatCompletionChoiceLogprobs",
    "ChatCompletionChunk",
    "ChatCompletionChunkChoice",
    "ChatCompletionChunkChoiceDelta",
    "ChatCompletionChunkToolCall",
    "ChatCompletionChunkToolCallFunction",
    "ChatCompletionMessage",
    "ChatCompletionTokenLogprob",
    "ChatCompletionTopLogprob",
    "CreateChatCompletionRequest",
    "FunctionDefinition",
    "MessageParam",
    "ModelCapabilities",
    "ModelConstraints",
    "ModelConstraintsTemperature",
    "ModelConstraintsTopP",
    "ModelList",
    "Model",
    "ModelPricing",
    "ModelSpec",
    "ModelType",
    "ModelTraitList",
    "ModelCompatibilityList",
    "ResponseFormat",
    "StreamOptions",
    "Tool",
    "ToolCall",
    "ToolCallFunction",
    "ToolChoice",
    "ToolChoiceFunction",
    "ToolChoiceObject",
    "UsageData",
    "VeniceParameters",
    "Character",
    "CharacterList",
    # Image types
    "GenerateImageRequest",
    "SimpleGenerateImageRequest",
    "UpscaleImageRequest",
    "TimingInfo",
    "ImageResponse",
    "ImageDataItem",
    "SimpleImageResponse",
    "ImageStyleList",
    "ImageStyleEnum",
    # API Keys types
    "ConsumptionLimit",
    "ApiKeyUsage",
    "ApiKey",
    "ApiKeyCreateRequest",
    "ApiKeyCreateResponse",
    "ApiKeyList",
    "ApiKeyRateLimitItem",
    "ApiTier",
    "Balances",
    "RateLimitInfo",
    "RateLimitLog",
    "RateLimitLogList",
    "ApiKeyGenerateWeb3KeyGetResponse",
    "ApiKeyGenerateWeb3KeyCreateRequest",
    "ApiKeyGenerateWeb3KeyCreateResponse",
    # Audio types
    "Voice",
    "AudioResponseFormat",
    "CreateSpeechRequest",
    "VoiceDetail",
    "VoiceList",
    # Embeddings types
    "CreateEmbeddingRequest",
    "Embedding",
    "EmbeddingUsage",
    "EmbeddingList",
    # Billing types
    "InferenceDetails",
    "BillingUsageEntry",
    "BillingUsagePagination",
    "BillingUsageResponse",
    "BillingUsageRequestParams",
    "BillingFormatEnum",
]