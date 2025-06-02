"""
Type definitions for Venice AI image-related API endpoints.
"""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, TypedDict, Union



class GenerateImageRequest(TypedDict, total=False):
    """Represents parameters for an image generation request to the `/image/generate` endpoint.
    
    This model defines the structure for requesting image generation using Venice AI's
    native image generation API. It provides comprehensive control over generation
    parameters including model selection, prompts, dimensions, and various quality
    and style settings.
    
    :param model: ID of the model to use for image generation (e.g., ``"venice-sd35"``).
    :type model: str
    :param prompt: Text prompt describing the image to generate.
    :type prompt: str
    :param cfg_scale: Optional. Classifier Free Guidance scale (1.0-30.0). Higher values adhere more strictly to the prompt.
    :type cfg_scale: float
    :param embed_exif_metadata: Optional. Whether to embed generation metadata in EXIF data.
    :type embed_exif_metadata: bool
    :param format: Optional. Output image format.
    :type format: Literal["jpeg", "png", "webp"]
    :param height: Optional. Height of the generated image in pixels.
    :type height: int
    :param hide_watermark: Optional. Whether to hide the Venice AI watermark from the generated image.
    :type hide_watermark: bool
    :param lora_strength: Optional. Strength of LoRA model adaptation (0-100).
    :type lora_strength: int
    :param negative_prompt: Optional. Text describing what to avoid in the generated image.
    :type negative_prompt: str
    :param return_binary: Optional. If ``True``, return raw image bytes instead of JSON response with base64 data.
    :type return_binary: bool
    :param safe_mode: Optional. Whether to enable content filtering for safer outputs.
    :type safe_mode: bool
    :param seed: Optional. Random seed for reproducible image generation results.
    :type seed: int
    :param steps: Optional. Number of diffusion steps. Higher values generally improve quality but increase generation time.
    :type steps: int
    :param style_preset: Optional. Style preset ID to apply to the generated image.
    :type style_preset: str
    :param width: Optional. Width of the generated image in pixels.
    :type width: int
    """
    
    model: str
    prompt: str
    cfg_scale: float
    embed_exif_metadata: bool
    format: Literal["jpeg", "png", "webp"]
    height: int
    hide_watermark: bool
    lora_strength: int
    negative_prompt: str
    return_binary: bool
    safe_mode: bool
    seed: int
    steps: int
    style_preset: str
    width: int


class SimpleGenerateImageRequest(TypedDict, total=False):
    """Represents parameters for an OpenAI-compatible image generation request to the `/images/generations` endpoint.
    
    This model provides a simplified interface for image generation that maintains
    compatibility with OpenAI's image generation API. It offers streamlined parameters
    for common image generation tasks while supporting Venice AI's enhanced features
    like custom quality settings and output formats.
    
    :param prompt: Text prompt describing the image to generate.
    :type prompt: str
    :param background: Optional. Background style for the generated image.
    :type background: Optional[Literal["transparent", "opaque", "auto"]]
    :param model: ID of the model to use for image generation.
    :type model: str
    :param moderation: Optional. Content moderation level to apply during generation.
    :type moderation: Optional[Literal["low", "auto"]]
    :param n: Optional. Number of images to generate (typically 1-10).
    :type n: Optional[int]
    :param output_compression: Optional. Output image compression level (0-100, where 100 is highest quality).
    :type output_compression: Optional[int]
    :param output_format: Optional. Output image format.
    :type output_format: Literal["jpeg", "png", "webp"]
    :param quality: Optional. Image quality setting that affects generation parameters.
    :type quality: Optional[Literal["auto", "high", "medium", "low", "hd", "standard"]]
    :param response_format: Optional. Format of the response data (base64 JSON or URL).
    :type response_format: Optional[Literal["b64_json", "url"]]
    :param size: Optional. Dimensions of the generated image in pixels.
    :type size: Optional[Literal["auto", "256x256", "512x512", "1024x1024", "1536x1024", "1024x1536", "1792x1024", "1024x1792"]]
    :param style: Optional. Artistic style of the generated image.
    :type style: Optional[Literal["vivid", "natural"]]
    :param user: Optional. User identifier for tracking and analytics purposes.
    :type user: str
    """
    
    prompt: str
    background: Optional[Literal["transparent", "opaque", "auto"]]
    model: str
    moderation: Optional[Literal["low", "auto"]]
    n: Optional[int]
    output_compression: Optional[int]
    output_format: Literal["jpeg", "png", "webp"]
    quality: Optional[Literal["auto", "high", "medium", "low", "hd", "standard"]]
    response_format: Optional[Literal["b64_json", "url"]]
    size: Optional[Literal["auto", "256x256", "512x512", "1024x1024", "1536x1024", "1024x1536", "1792x1024", "1024x1792"]]
    style: Optional[Literal["vivid", "natural"]]
    user: str


class UpscaleImageRequest(TypedDict, total=False):
    """Represents parameters for an image upscaling request to the `/image/upscale` endpoint.
    
    This model defines the structure for requesting image upscaling and enhancement
    operations. It allows for scaling existing images to higher resolutions while
    optionally applying AI-powered enhancements to improve quality and detail.
    
    Note: The 'image' data is sent base64-encoded within the JSON payload.
    
    :param enhance: Optional. Whether to enhance image quality during upscaling (``"true"`` or ``"false"``).
    :type enhance: Literal["true", "false"]
    :param enhance_creativity: Optional. Creativity level for enhancement (0.0-1.0, where 1.0 is most creative).
    :type enhance_creativity: Optional[float]
    :param enhance_prompt: Optional. Text prompt to guide the enhancement process.
    :type enhance_prompt: str
    :param replication: Optional. Replication factor for matching the original image (0.0-1.0, where 1.0 matches exactly).
    :type replication: Optional[float]
    :param scale: Optional. Scaling factor for upscaling (e.g., ``2.0`` for 2x upscaling).
    :type scale: float
    """
    
    enhance: Literal["true", "false"]
    enhance_creativity: Optional[float]
    enhance_prompt: str
    replication: Optional[float]
    scale: float


class TimingInfo(TypedDict):
    """Represents timing metrics for image generation and processing operations.
    
    This model provides detailed performance information about various stages
    of image generation requests, enabling monitoring and optimization of
    processing times across different components of the Venice AI pipeline.
    All timing values are measured in seconds.
    
    :param inferenceDuration: Duration of the actual inference/generation process in seconds.
    :type inferenceDuration: float
    :param inferencePreprocessingTime: Time spent on preprocessing operations before inference begins, in seconds.
    :type inferencePreprocessingTime: float
    :param inferenceQueueTime: Time spent waiting in the inference queue before processing starts, in seconds.
    :type inferenceQueueTime: float
    :param total: Total time taken for the entire request from start to completion, in seconds.
    :type total: float
    """
    
    inferenceDuration: float
    inferencePreprocessingTime: float
    inferenceQueueTime: float
    total: float


class ImageResponse(TypedDict):
    """Represents the response structure from the `/image/generate` endpoint.
    
    This model defines the complete response format for Venice AI's native image
    generation API, containing the generated images as base64-encoded data along
    with metadata including timing information and the original request parameters.
    
    :param id: Unique identifier for the image generation request, used for tracking and reference.
    :type id: str
    :param images: List of base64-encoded image data strings representing the generated images.
    :type images: List[str]
    :param request: Optional. Echo of the original request parameters that were used for generation.
    :type request: Optional[Dict[str, Any]]
    :param timing: Detailed timing information and performance metrics for the request.
    :type timing: TimingInfo
    :param created: ISO 8601 timestamp indicating when the image generation was completed.
    :type created: str
    """
    
    id: str
    images: List[str]
    request: Optional[Dict[str, Any]]
    timing: TimingInfo
    created: str


class ImageDataItem(TypedDict, total=False):
    """Represents an individual image data item within a :class:`SimpleImageResponse`.
    
    This model defines the structure for a single generated image in OpenAI-compatible
    responses, providing either base64-encoded image data or a URL reference to the
    generated image depending on the requested response format.
    
    Contains either base64 encoded image data or a URL to the image, but not both.
    
    :param b64_json: Base64-encoded image data as a JSON string (when ``response_format`` is ``"b64_json"``).
    :type b64_json: str
    :param url: URL pointing to the generated image (when ``response_format`` is ``"url"``).
    :type url: str
    """
    
    b64_json: str
    url: str


class SimpleImageResponse(TypedDict):
    """Represents the response structure from the `/images/generations` (OpenAI-compatible) endpoint.
    
    This model provides an OpenAI-compatible response format for image generation
    requests, containing a list of generated images and creation timestamp. It
    maintains compatibility with existing OpenAI client libraries and workflows.
    
    :param created: Unix timestamp (seconds since epoch) indicating when the image generation was initiated.
    :type created: int
    :param data: List of generated image data items, each containing either base64 data or URL.
    :type data: List[ImageDataItem]
    """
    
    created: int
    data: List[ImageDataItem]


class ImageStyleList(TypedDict):
    """Represents the response structure from the `/image/styles` endpoint.
    
    This model defines the format for retrieving available image style presets
    from the Venice AI API. These styles can be used with the ``style_preset``
    parameter in image generation requests to influence the artistic direction
    and visual characteristics of generated images.
    
    :param data: List of available image style preset names that can be used in generation requests.
    :type data: List[str]
    :param object: Type of the response object, always ``"list"`` to indicate this is a list response.
    :type object: Literal["list"]
    """
    
    data: List[str]
    object: Literal["list"]


class ImageStyleEnum(str, Enum):
    """
    Represents common or example styles for image generation.

    This enum provides a static, curated list of frequently used image styles.
    For a comprehensive and dynamically updated list of all available styles,
    it is recommended to use the
    :meth:`venice_ai.resources.image.Image.get_available_styles` method (or its
    asynchronous counterpart
    :meth:`venice_ai.resources.image.AsyncImage.get_available_styles`)
    to fetch the current styles directly from the API.

    **Example static values:**
    """
    
    # Common styles mentioned in documentation, example data and type definitions
    THREE_D_MODEL = "3D Model"  #: 3D rendered model style
    ANALOG_FILM = "Analog Film"  #: Vintage analog film photography style
    ANIME = "Anime"  #: Japanese anime/manga artistic style
    CINEMATIC = "Cinematic"  #: Movie-like cinematic style with dramatic lighting
    COMIC_BOOK = "Comic Book"  #: Comic book illustration style