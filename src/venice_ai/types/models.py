from typing import Any, Dict, List, Literal, Optional, TypedDict

__all__ = [
    "PricingUnit",
    "PricingDetail",
    "ModelPricing",
    "ModelCapabilities",
    "ModelConstraintsTemperature",
    "ModelConstraintsTopP",
    "ModelConstraints",
    "ModelSpec",
    "Model",
    "ModelList",
    "ModelType",
    "ModelTraitList",
    "ModelCompatibilityList",
]

ModelType = Literal["embedding", "image", "text", "tts", "upscale"]
"""
Type alias for valid model types in the Venice.ai API.

Defines the available categories of AI models that can be filtered when
listing models, traits, or compatibility mappings. Each type represents
a different class of AI functionality:

- ``"embedding"``: Models that generate vector embeddings from text
- ``"image"``: Models for image generation and manipulation
- ``"text"``: Models for text generation and chat completions
- ``"tts"``: Text-to-speech models for audio generation
- ``"upscale"``: Models for image upscaling and enhancement
"""

class PricingUnit(TypedDict):
    """Represents a pricing unit with both USD and VCU values.
    
    :param usd: Cost in US dollars.
    :type usd: float
    :param vcu: Cost in Venice Compute Units.
    :type vcu: float
    """
    usd: float
    vcu: float

class PricingDetail(TypedDict):
    """Represents pricing details for input and output.
    
    :param input: Pricing for input (per 1000 tokens for text models).
    :type input: PricingUnit
    :param output: Pricing for output (per 1000 tokens for text models).
    :type output: PricingUnit
    """
    input: PricingUnit
    output: PricingUnit

class ModernPricing(TypedDict):
    """Represents the modern pricing structure with required input/output."""
    input: PricingUnit
    output: PricingUnit

class ModelPricing(ModernPricing, total=False):
    """Represents pricing information for an AI model.
    
    Defines the cost structure for using a model, including costs per token,
    image, or time unit depending on the model type. Used within the :class:`Model`
    class to provide billing information.
    
    The pricing structure now supports both USD and VCU (Venice Compute Units)
    for accurate cost tracking and billing.
    
    :param input: Pricing for input operations.
    :type input: PricingUnit
    :param output: Pricing for output operations.
    :type output: PricingUnit
    :param input_cost_per_mtok: Legacy: Cost for input per 1000 tokens (USD only).
    :type input_cost_per_mtok: float
    :param output_cost_per_mtok: Legacy: Cost for output per 1000 tokens (USD only).
    :type output_cost_per_mtok: float
    :param input_cost_per_image: Cost for input per image.
    :type input_cost_per_image: float
    :param output_cost_per_image: Cost for output per image.
    :type output_cost_per_image: float
    :param input_cost_per_second: Cost for input per second (e.g., audio).
    :type input_cost_per_second: float
    :param output_cost_per_second: Cost for output per second (e.g., audio).
    :type output_cost_per_second: float
    """
    # Legacy fields for backward compatibility
    input_cost_per_mtok: float
    output_cost_per_mtok: float
    input_cost_per_image: float
    output_cost_per_image: float
    input_cost_per_second: float
    output_cost_per_second: float

class ModelCapabilities(TypedDict, total=False):
    """Defines the functional capabilities and limitations of an AI model.
    
    Specifies what features a model supports, including code optimization,
    quantization method, reasoning, vision, and various other capabilities.
    Uses camelCase field names to match the API response format.
    
    :param optimizedForCode: Indicates if the model is optimized for code generation.
    :type optimizedForCode: bool
    :param quantization: The quantization method used (e.g., "fp16", "fp8").
    :type quantization: str
    :param supportsFunctionCalling: Indicates if the model supports function calling.
    :type supportsFunctionCalling: bool
    :param supportsReasoning: Indicates if the model supports reasoning capabilities.
    :type supportsReasoning: bool
    :param supportsResponseSchema: Indicates if the model supports structured response schemas.
    :type supportsResponseSchema: bool
    :param supportsVision: Indicates if the model supports vision/image understanding.
    :type supportsVision: bool
    :param supportsWebSearch: Indicates if the model supports web search integration.
    :type supportsWebSearch: bool
    :param supportsLogProbs: Indicates if the model supports log probability output.
    :type supportsLogProbs: bool
    :param streaming: Legacy field - Indicates if the model supports streaming responses.
    :type streaming: bool
    :param async_: Legacy field - Indicates if the model supports asynchronous operations.
    :type async_: bool
    :param max_tokens: Legacy field - Maximum number of tokens the model can process.
    :type max_tokens: int
    :param supports_functions: Legacy field - Use supportsFunctionCalling instead.
    :type supports_functions: bool
    """
    # Current API fields (camelCase as returned by API)
    optimizedForCode: bool
    quantization: str
    supportsFunctionCalling: bool
    supportsReasoning: bool
    supportsResponseSchema: bool
    supportsVision: bool
    supportsWebSearch: bool
    supportsLogProbs: bool
    
    # Legacy fields for backward compatibility
    streaming: bool
    async_: bool  # Field name is async_ due to async being a keyword
    max_tokens: int
    supports_functions: bool  # Deprecated, use supportsFunctionCalling

class ModelConstraintsTemperature(TypedDict, total=False):
    """Defines valid range and default value for the temperature parameter.
    
    Specifies the constraints for the temperature parameter that controls
    randomness in model outputs. Used within :class:`ModelConstraints`.
    Note: The API may only return 'default' without min/max values.
    
    :param default: Default temperature value (always present).
    :type default: float
    :param min: Minimum allowed temperature value (optional).
    :type min: float
    :param max: Maximum allowed temperature value (optional).
    :type max: float
    """
    default: float
    min: float
    max: float

class ModelConstraintsTopP(TypedDict, total=False):
    """Defines valid range and default value for the top_p parameter.
    
    Specifies the constraints for the top_p parameter that controls nucleus
    sampling in model outputs. Used within :class:`ModelConstraints`.
    Note: The API may only return 'default' without min/max values.
    
    :param default: Default top_p value (always present).
    :type default: float
    :param min: Minimum allowed top_p value (optional).
    :type min: float
    :param max: Maximum allowed top_p value (optional).
    :type max: float
    """
    default: float
    min: float
    max: float

class ModelConstraints(TypedDict):
    """Defines parameter constraints and valid ranges for a model.
    
    Contains the allowable ranges and default values for various model
    parameters like temperature and top_p. Used within the :class:`Model`
    class to specify parameter limits.
    
    :param temperature: Constraints for the temperature parameter.
    :type temperature: ModelConstraintsTemperature
    :param top_p: Constraints for the top_p parameter.
    :type top_p: ModelConstraintsTopP
    """
    temperature: ModelConstraintsTemperature
    top_p: ModelConstraintsTopP

class ModelSpec(TypedDict, total=False):
    """Defines the specifications for a model including pricing and capabilities.
    
    Contains detailed information about a model's pricing structure, capabilities,
    constraints, and other specifications. This is the main container for model
    metadata in the API response.
    
    :param pricing: Pricing information for the model with USD and VCU costs.
    :type pricing: ModelPricing
    :param availableContextTokens: Maximum context window size in tokens.
    :type availableContextTokens: int
    :param capabilities: Model capabilities and feature support.
    :type capabilities: ModelCapabilities
    :param constraints: Parameter constraints for the model.
    :type constraints: ModelConstraints
    :param name: Human-readable name of the model.
    :type name: str
    :param modelSource: URL or reference to the model source.
    :type modelSource: str
    :param offline: Whether the model is currently offline.
    :type offline: bool
    :param traits: List of model traits (e.g., "default", "fastest").
    :type traits: List[str]
    :param beta: Indicates if this is a beta model (optional).
    :type beta: bool
    """
    pricing: ModelPricing
    availableContextTokens: int
    capabilities: ModelCapabilities  # Changed from Dict[str, Any] to use proper type
    constraints: ModelConstraints
    name: str
    modelSource: str
    offline: bool
    traits: List[str]
    beta: bool  # New optional field for beta models

class Model(TypedDict):
    """Represents a single AI model available through the Venice.ai API.
    
    Contains comprehensive information about an AI model including its
    identification and specifications. The model_spec field contains all
    the detailed information about pricing, capabilities, and constraints.
    
    :param id: Unique identifier for the model.
    :type id: str
    :param object: Object type, always ``"model"``.
    :type object: Literal["model"]
    :param created: Unix timestamp (seconds) of when the model was created.
    :type created: int
    :param owned_by: Organization or user that owns the model.
    :type owned_by: str
    :param type: Type of the model (e.g., ``"text"``, ``"image"``).
    :type type: ModelType
    :param model_spec: Detailed specifications including pricing, capabilities, and constraints.
    :type model_spec: ModelSpec
    """
    id: str
    object: Literal["model"]
    created: int
    owned_by: str
    type: ModelType
    model_spec: ModelSpec

class ModelList(TypedDict):
    """Represents a collection of AI models returned by the list models endpoint.
    
    Contains a list of available :class:`Model` objects along with metadata
    about the collection. Typically returned when querying for available
    models through the API.
    
    :param object: Object type, always ``"list"``.
    :type object: Literal["list"]
    :param data: A list of available :class:`Model` objects.
    :type data: List[Model]
    :param type: Optional. The type of models in the list, if filtered.
    :type type: Optional[ModelType]
    """
    object: Literal["list"]
    data: List[Model]
    type: Optional[ModelType]

class ModelTraitList(TypedDict):
    """Represents a mapping of model traits to their corresponding model IDs.
    
    Provides a way to map semantic model traits (like "default", "fastest",
    "most_accurate") to specific model IDs. Used for trait-based model
    selection through the API.
    
    :param object: Object type, always ``"list"``.
    :type object: Literal["list"]
    :param data: Mapping of trait names to model IDs (e.g., ``{"default": "llama-3.3-70b"}``).
    :type data: Dict[str, str]
    :param type: Optional. The type of models in the mapping, if filtered.
    :type type: Optional[ModelType]
    """
    object: Literal["list"]
    data: Dict[str, str]
    type: Optional[ModelType]

class ModelCompatibilityList(TypedDict):
    """Represents a mapping of external model names to Venice.ai model IDs.
    
    Provides compatibility mappings that allow users to reference models
    using external naming conventions (e.g., OpenAI model names) while
    automatically resolving to the corresponding Venice.ai model IDs.
    
    :param object: Object type, always ``"list"``.
    :type object: Literal["list"]
    :param data: Mapping of external model names to Venice model IDs (e.g., ``{"gpt-4o": "llama-3.3-70b"}``).
    :type data: Dict[str, str]
    :param type: Optional. The type of models in the mapping, if filtered.
    :type type: Optional[ModelType]
    """
    object: Literal["list"]
    data: Dict[str, str]
    type: Optional[ModelType]