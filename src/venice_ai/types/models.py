from typing import Dict, List, Literal, Optional, TypedDict

__all__ = [
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

class ModelPricing(TypedDict, total=False):
    """Represents pricing information for an AI model.
    
    Defines the cost structure for using a model, including costs per token,
    image, or time unit depending on the model type. Used within the :class:`Model`
    class to provide billing information.
    
    :param input_cost_per_mtok: Cost for input per 1000 tokens.
    :type input_cost_per_mtok: float
    :param output_cost_per_mtok: Cost for output per 1000 tokens.
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
    input_cost_per_mtok: float
    output_cost_per_mtok: float
    input_cost_per_image: float
    output_cost_per_image: float
    input_cost_per_second: float
    output_cost_per_second: float

class ModelCapabilities(TypedDict):
    """Defines the functional capabilities and limitations of an AI model.
    
    Specifies what features a model supports, such as streaming responses,
    asynchronous operations, token limits, and function calling. Used within
    the :class:`Model` class to describe model features.
    
    :param streaming: Indicates if the model supports streaming responses.
    :type streaming: bool
    :param async_: Indicates if the model supports asynchronous operations.
        Note: Field name is ``async_`` due to ``async`` being a Python keyword.
    :type async_: bool
    :param max_tokens: Maximum number of tokens the model can process in a single request.
    :type max_tokens: int
    :param supports_functions: Indicates if the model supports function calling.
    :type supports_functions: bool
    """
    streaming: bool
    async_: bool  # Field name is async_ due to async being a keyword
    max_tokens: int
    supports_functions: bool

class ModelConstraintsTemperature(TypedDict):
    """Defines valid range and default value for the temperature parameter.
    
    Specifies the constraints for the temperature parameter that controls
    randomness in model outputs. Used within :class:`ModelConstraints`.
    
    :param default: Default temperature value.
    :type default: float
    :param min: Minimum allowed temperature value.
    :type min: float
    :param max: Maximum allowed temperature value.
    :type max: float
    """
    default: float
    min: float
    max: float

class ModelConstraintsTopP(TypedDict):
    """Defines valid range and default value for the top_p parameter.
    
    Specifies the constraints for the top_p parameter that controls nucleus
    sampling in model outputs. Used within :class:`ModelConstraints`.
    
    :param default: Default top_p value.
    :type default: float
    :param min: Minimum allowed top_p value.
    :type min: float
    :param max: Maximum allowed top_p value.
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

class ModelSpec(TypedDict):
    """Defines the input and output format specifications for a model.
    
    Specifies what data formats a model expects as input and produces as
    output. Used within the :class:`Model` class to describe model I/O
    requirements.
    
    :param input_format: Expected input format (e.g., ``"text"``, ``"image_url"``).
    :type input_format: str
    :param output_format: Output format (e.g., ``"text"``, ``"image_url"``).
    :type output_format: str
    """
    input_format: str
    output_format: str

class Model(TypedDict):
    """Represents a single AI model available through the Venice.ai API.
    
    Contains comprehensive information about an AI model including its
    identification, capabilities, pricing, constraints, and specifications.
    Typically returned by the list models endpoint and used throughout
    the API to reference specific models.
    
    :param id: Unique identifier for the model.
    :type id: str
    :param object: Object type, always ``"model"``.
    :type object: Literal["model"]
    :param created: Unix timestamp (seconds) of when the model was created.
    :type created: int
    :param owned_by: Organization or user that owns the model.
    :type owned_by: str
    :param name: Name of the model.
    :type name: str
    :param description: Description of the model.
    :type description: str
    :param type: Type of the model (e.g., ``"text"``, ``"image"``).
    :type type: ModelType
    :param pricing: Pricing information for the model.
    :type pricing: ModelPricing
    :param capabilities: Capabilities of the model.
    :type capabilities: ModelCapabilities
    :param constraints: Constraints for model parameters.
    :type constraints: ModelConstraints
    :param spec: Specification details for the model.
    :type spec: ModelSpec
    """
    id: str
    object: Literal["model"]
    created: int
    owned_by: str
    name: str
    description: str
    type: ModelType
    pricing: ModelPricing
    capabilities: ModelCapabilities
    constraints: ModelConstraints
    spec: ModelSpec

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