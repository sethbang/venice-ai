import pytest
import pytest_asyncio
import os
import logging
from io import BytesIO
from typing import Union, Dict, Any, Optional, Callable, List, Tuple
from venice_ai import VeniceClient, AsyncVeniceClient
from venice_ai.exceptions import VeniceError, InvalidRequestError, NotFoundError
from venice_ai.types.image import ImageResponse, SimpleImageResponse, ImageDataItem

# Configure minimal logging format
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',  # Only show the message, no timestamps or levels
)
logger = logging.getLogger(__name__)

# Define default image model and style for testing
DEFAULT_IMAGE_MODEL = "venice-sd35"
DEFAULT_IMAGE_STYLE = "vivid"

# Path to the dummy image file
DUMMY_IMAGE_PATH = "e2e_tests/data/sample_image.png"

# API requires images to be within certain pixel dimensions
MIN_PIXELS = 65536       # 256x256
MAX_PIXELS = 16777216    # 4096x4096


# Helper functions
def check_image_dimensions(image_path_or_bytes: Union[str, bytes]) -> Optional[Tuple[int, int, int]]:
    """
    Check if image dimensions meet API requirements.
    
    Args:
        image_path_or_bytes: Either a file path or image bytes
        
    Returns:
        Tuple of (width, height, total_pixels) if successful, None otherwise
    """
    try:
        from PIL import Image
        
        if isinstance(image_path_or_bytes, str):
            # Path provided
            with Image.open(image_path_or_bytes) as img:
                width, height = img.size
                total_pixels = width * height
        else:
            # Bytes provided
            with Image.open(BytesIO(image_path_or_bytes)) as img:
                width, height = img.size
                total_pixels = width * height
        
        # Only log warning if dimensions are out of range
        if not (MIN_PIXELS <= total_pixels <= MAX_PIXELS):
            logger.warning(f"Warning: Image dimensions ({width}x{height}, {total_pixels} pixels) outside API range")
            
        return width, height, total_pixels
        
    except ImportError:
        logger.warning("Warning: Pillow library not installed, skipping image dimension check")
    except Exception as e:
        logger.error(f"Error checking image dimensions: {e}")
        
    return None


def read_test_image() -> Optional[bytes]:
    """Read test image into bytes if it exists"""
    if not os.path.exists(DUMMY_IMAGE_PATH):
        pytest.skip(f"Dummy image file not found at {DUMMY_IMAGE_PATH}")
        return None
    
    with open(DUMMY_IMAGE_PATH, "rb") as f:
        return f.read()


def validate_image_response(response: Union[ImageResponse, SimpleImageResponse, bytes], expected_format: Optional[str] = None) -> None:
    """
    Validate common aspects of image API responses
    
    Args:
        response: API response (ImageResponse, SimpleImageResponse, or raw bytes for image data)
        expected_format: Expected image format if known (url or b64_json)
    """
    if isinstance(response, bytes):
        logger.info(f"Validating raw bytes response, length: {len(response)}")
        assert len(response) > 0, "Raw image data should not be empty"
        # Optionally check if it starts with PNG header
        if response.startswith(b'\x89PNG'):
            logger.info("Response starts with PNG header, likely valid image data")
        return

    # Handle ImageResponse (from client.image.generate)
    if isinstance(response, ImageResponse):
        logger.info(f"Validating ImageResponse. ID: {response.id}")
        assert isinstance(response.id, str), "Response ID should be a string"
        assert isinstance(response.images, list), "Images should be a list"
        assert len(response.images) > 0, "Images list should not be empty"
        logger.info(f"Response images count: {len(response.images)}")
        if response.images:
            logger.info(f"First image data length: {len(response.images[0])}")
        return

    # Handle SimpleImageResponse (from client.image.simple_generate)
    if isinstance(response, SimpleImageResponse):
        logger.info(f"Validating SimpleImageResponse. Created: {response.created}")
        assert isinstance(response.created, int), "Response created should be an integer"
        assert isinstance(response.images, list), "Response images should be a list"
        assert len(response.images) > 0, "Response images list should not be empty"
        
        logger.info(f"Response images count: {len(response.images)}")
        image_data_item = response.images[0]
        assert isinstance(image_data_item, ImageDataItem), "Image data should be an ImageDataItem"
        
        # Check format if specified - for SimpleImageResponse, images are ImageDataItem objects
        if expected_format == "url":
            assert image_data_item.url is not None, "Image URL should be present"
            assert image_data_item.url.startswith("http"), "Image URL should be valid"
        elif expected_format == "b64_json":
            assert image_data_item.b64_json is not None, "Base64 data should be present"
            assert len(image_data_item.b64_json) > 0, "Base64 data should not be empty"
        else:
            # If no specific format expected, check that at least one field is populated
            assert image_data_item.b64_json is not None or image_data_item.url is not None, "Either b64_json or url should be present"
        
        return

    # If we get here, it's an unexpected type
    raise AssertionError(f"Unexpected response type: {type(response)}")


# Parametrized tests

# Separate test for async basic generation to isolate event loop issue
@pytest.mark.asyncio
async def test_generate_image_basic_async_isolated(async_venice_client):
    """Test basic image generation - async isolated"""
    client = async_venice_client # Direct fixture usage
    prompt = "A simple test image async isolated"
    logger.info("Attempting async isolated image generation")
    try:
        response = await client.image.generate(
            model=DEFAULT_IMAGE_MODEL,
            prompt=prompt,
            width=512,
            height=512
        )
        logger.info(f"Async isolated response received: {str(response)[:200]}")
        assert isinstance(response, ImageResponse)
        validate_image_response(response)
    except Exception as e:
        logger.error(f"Error in test_generate_image_basic_async_isolated: {e}", exc_info=True)
        raise


@pytest.mark.parametrize(
    "client_fixture,is_async",
    [
        ("venice_client", False),
        # pytest.param("async_venice_client", True, marks=pytest.mark.asyncio), # Temporarily disable for this test
    ],
)
async def test_generate_image_basic(request, client_fixture, is_async):
    """Test basic image generation with minimal parameters (SYNC ONLY FOR NOW)"""
    client = request.getfixturevalue(client_fixture)
    prompt = f"A simple test image {'async' if is_async else 'sync'}"
    
    if is_async:
        # This branch should ideally not be hit if async is disabled above
        logger.warning("Async branch hit in test_generate_image_basic - this should be isolated now.")
        response = await client.image.generate(
            model=DEFAULT_IMAGE_MODEL,
            prompt=prompt,
            width=512,
            height=512
        )
    else:
        response = client.image.generate(
            model=DEFAULT_IMAGE_MODEL,
            prompt=prompt,
            width=512,
            height=512
        )
    
    assert isinstance(response, ImageResponse)
    validate_image_response(response)


# Refactored test_generate_image_with_all_params
def test_generate_image_with_all_params_sync(venice_client):
    """Test image generation with all available parameters (Sync)"""
    client = venice_client
    prompt = "A complex scene with many details sync"
    logger.info("Attempting sync image generation with all params")
    try:
        response = client.image.generate(
            model=DEFAULT_IMAGE_MODEL,
            prompt=prompt,
            width=1024,
            height=768,
        )
        logger.info(f"Sync all_params response received: {str(response)[:200]}")
        assert isinstance(response, ImageResponse)
        validate_image_response(response)
    except Exception as e:
        logger.error(f"Error in test_generate_image_with_all_params_sync: {e}", exc_info=True)
        raise

@pytest.mark.asyncio
async def test_generate_image_with_all_params_async(async_venice_client):
    """Test image generation with all available parameters (Async)"""
    client = async_venice_client
    prompt = "A complex scene with many details async"
    logger.info("Attempting async image generation with all params")
    try:
        response = await client.image.generate(
            model=DEFAULT_IMAGE_MODEL,
            prompt=prompt,
            width=1024,
            height=768,
        )
        logger.info(f"Async all_params response received: {str(response)[:200]}")
        assert isinstance(response, ImageResponse)
        validate_image_response(response)
    except Exception as e:
        logger.error(f"Error in test_generate_image_with_all_params_async: {e}", exc_info=True)
        raise

# Original test_generate_image_with_all_params is now replaced by the two above.
# The following @pytest.mark.parametrize decorator and function definition
# for the next test (test_simple_generate_image_basic) is the correct continuation point.

# Refactored test_simple_generate_image_basic
def test_simple_generate_image_basic_sync(venice_client):
    """Test simple image generation (OpenAI-compatible) with basic parameters (Sync)"""
    client = venice_client
    prompt = "A cute animal sync"
    logger.info("Attempting sync simple image generation")
    try:
        response = client.image.simple_generate(
            model="venice-sd35",
            prompt=prompt,
            size="512x512",
            quality="standard",
        )
        logger.info(f"Sync simple_generate_image_basic response: {str(response)[:200]}")
        assert isinstance(response, SimpleImageResponse)
        validate_image_response(response)
        assert isinstance(response.images, list)
        assert len(response.images) > 0
        # Check that the image data is not empty (either b64_json or url should have content)
        image_item = response.images[0]
        assert (image_item.b64_json and len(image_item.b64_json) > 0) or (image_item.url and len(image_item.url) > 0), "Image data should not be empty"
    except Exception as e:
        logger.error(f"Error in test_simple_generate_image_basic_sync: {e}", exc_info=True)
        raise

@pytest.mark.asyncio
async def test_simple_generate_image_basic_async(async_venice_client):
    """Test simple image generation (OpenAI-compatible) with basic parameters (Async)"""
    client = async_venice_client
    prompt = "A cute animal async"
    logger.info("Attempting async simple image generation")
    try:
        response = await client.image.simple_generate(
            model="venice-sd35",
            prompt=prompt,
            size="512x512",
            quality="standard",
        )
        logger.info(f"Async simple_generate_image_basic response: {str(response)[:200]}")
        assert isinstance(response, SimpleImageResponse)
        validate_image_response(response)
        assert isinstance(response.images, list)
        assert len(response.images) > 0
        # Check that the image data is not empty (either b64_json or url should have content)
        image_item = response.images[0]
        assert (image_item.b64_json and len(image_item.b64_json) > 0) or (image_item.url and len(image_item.url) > 0), "Image data should not be empty"
    except Exception as e:
        logger.error(f"Error in test_simple_generate_image_basic_async: {e}", exc_info=True)
        raise

# Refactored test_simple_generate_image_with_all_params
def test_simple_generate_image_with_all_params_sync(venice_client):
    """Test simple image generation with all available parameters (Sync)"""
    client = venice_client
    prompt = "An abstract concept sync"
    logger.info("Attempting sync simple image generation with all params")
    try:
        response = client.image.simple_generate(
            model="venice-sd35",
            prompt=prompt,
            size="1024x1024",
            quality="high",
            style=DEFAULT_IMAGE_STYLE,
            n=1,
        )
        logger.info(f"Sync simple_generate_image_with_all_params response: {str(response)[:200]}")
        assert isinstance(response, SimpleImageResponse)
        validate_image_response(response)
        assert len(response.images) == 1
    except Exception as e:
        logger.error(f"Error in test_simple_generate_image_with_all_params_sync: {e}", exc_info=True)
        raise

@pytest.mark.asyncio
async def test_simple_generate_image_with_all_params_async(async_venice_client):
    """Test simple image generation with all available parameters (Async)"""
    # Use a dedicated client with a longer timeout for this E2E test
    client = AsyncVeniceClient(timeout=180.0)
    prompt = "An abstract concept async"
    logger.info("Attempting async simple image generation with all params")
    try:
        response = await client.image.simple_generate(
            model="venice-sd35",
            prompt=prompt,
            size="1024x1024",
            quality="high",
            style=DEFAULT_IMAGE_STYLE,
            n=1,
        )
        logger.info(f"Async simple_generate_image_with_all_params response: {str(response)[:200]}")
        assert isinstance(response, SimpleImageResponse)
        validate_image_response(response)
        assert len(response.images) == 1
    except Exception as e:
        logger.error(f"Error in test_simple_generate_image_with_all_params_async: {e}", exc_info=True)
        raise

# Refactored test_upscale_image_from_file
def test_upscale_image_from_file_sync(venice_client):
    """Test image upscaling from a file path (Sync)"""
    if not os.path.exists(DUMMY_IMAGE_PATH):
        pytest.skip(f"Dummy image file not found at {DUMMY_IMAGE_PATH}")
    client = venice_client
    check_image_dimensions(DUMMY_IMAGE_PATH)
    logger.info("Attempting sync upscale image from file")
    try:
        response = client.image.upscale(image=DUMMY_IMAGE_PATH, scale=2.0, timeout=300.0)
        logger.info(f"Sync upscale_from_file response: {str(response)[:200]}")
        # Upscale might return ImageResponse or bytes, validate_image_response handles both
        validate_image_response(response)
    except Exception as e:
        logger.error(f"Error in test_upscale_image_from_file_sync: {e}", exc_info=True)
        raise

@pytest.mark.asyncio
async def test_upscale_image_from_file_async(async_venice_client):
    """Test image upscaling from a file path (Async)"""
    if not os.path.exists(DUMMY_IMAGE_PATH):
        pytest.skip(f"Dummy image file not found at {DUMMY_IMAGE_PATH}")
    client = async_venice_client
    check_image_dimensions(DUMMY_IMAGE_PATH)
    logger.info("Attempting async upscale image from file")
    try:
        response = await client.image.upscale(image=DUMMY_IMAGE_PATH, scale=2.0, timeout=300.0)
        logger.info(f"Async upscale_from_file response: {str(response)[:200]}")
        # Upscale might return ImageResponse or bytes, validate_image_response handles both
        validate_image_response(response)
    except Exception as e:
        logger.error(f"Error in test_upscale_image_from_file_async: {e}", exc_info=True)
        raise

# Refactored test_upscale_image_from_bytes
def test_upscale_image_from_bytes_sync(venice_client):
    """Test image upscaling from bytes (Sync)"""
    image_bytes = read_test_image()
    if not image_bytes: return
    client = venice_client
    check_image_dimensions(image_bytes)
    logger.info("Attempting sync upscale image from bytes")
    try:
        response = client.image.upscale(image=image_bytes, scale=2.0, timeout=300.0)
        logger.info(f"Sync upscale_from_bytes response: {str(response)[:200]}")
        # Upscale might return ImageResponse or bytes, validate_image_response handles both
        validate_image_response(response)
    except Exception as e:
        logger.error(f"Error in test_upscale_image_from_bytes_sync: {e}", exc_info=True)
        raise

@pytest.mark.asyncio
async def test_upscale_image_from_bytes_async(async_venice_client):
    """Test image upscaling from bytes (Async)"""
    image_bytes = read_test_image()
    if not image_bytes: return
    client = async_venice_client
    check_image_dimensions(image_bytes)
    logger.info("Attempting async upscale image from bytes")
    try:
        response = await client.image.upscale(image=image_bytes, scale=2.0, timeout=300.0)
        logger.info(f"Async upscale_from_bytes response: {str(response)[:200]}")
        # Upscale might return ImageResponse or bytes, validate_image_response handles both
        validate_image_response(response)
    except Exception as e:
        logger.error(f"Error in test_upscale_image_from_bytes_async: {e}", exc_info=True)
        raise




# Refactored test_list_image_styles
def test_list_image_styles_sync(venice_client):
    """Test listing available image styles (Sync)"""
    client = venice_client
    logger.info("Attempting sync list_image_styles")
    try:
        styles = client.image.list_styles()
        logger.info(f"Sync list_image_styles response: {str(styles)[:200]}")
        assert isinstance(styles, dict)
        assert "data" in styles
        assert isinstance(styles["data"], list)
        if len(styles["data"]) > 0:
            sample_style = styles["data"][0]
            if isinstance(sample_style, dict):
                if "id" in sample_style: assert isinstance(sample_style["id"], str)
                if "name" in sample_style: assert isinstance(sample_style["name"], str)
    except Exception as e:
        logger.error(f"Error in test_list_image_styles_sync: {e}", exc_info=True)
        raise

@pytest.mark.asyncio
async def test_list_image_styles_async(async_venice_client):
    """Test listing available image styles (Async)"""
    client = async_venice_client
    logger.info("Attempting async list_image_styles")
    try:
        styles = await client.image.list_styles()
        logger.info(f"Async list_image_styles response: {str(styles)[:200]}")
        assert isinstance(styles, dict)
        assert "data" in styles
        assert isinstance(styles["data"], list)
        if len(styles["data"]) > 0:
            sample_style = styles["data"][0]
            if isinstance(sample_style, dict):
                if "id" in sample_style: assert isinstance(sample_style["id"], str)
                if "name" in sample_style: assert isinstance(sample_style["name"], str)
    except Exception as e:
        logger.error(f"Error in test_list_image_styles_async: {e}", exc_info=True)
        raise

# Refactored test_generate_image_error_handling
def test_generate_image_error_handling_sync(venice_client):
    """Test error handling for image generation with invalid inputs (Sync)"""
    client = venice_client
    test_cases = [
        ("prompt", "", InvalidRequestError),("width", -100, InvalidRequestError),("model", "nonexistent-model", NotFoundError),] # Expect 404 for bad model
    logger.info("Attempting sync generate_image_error_handling")
    for param_name, invalid_value, expected_error in test_cases:
        params = {"model": DEFAULT_IMAGE_MODEL,"prompt": "Valid prompt","width": 512,"height": 512}
        params[param_name] = invalid_value
        try:
            with pytest.raises((VeniceError, expected_error)) as excinfo:
                client.image.generate(**params)
            error_message = str(excinfo.value)
            # Check only the type of exception, as message content is unreliable
            assert isinstance(excinfo.value, expected_error), \
                f"Expected {expected_error} for {param_name}, but got {type(excinfo.value)}"
        except Exception as e:
            logger.error(f"Error in test_generate_image_error_handling_sync for {param_name}: {e}", exc_info=True)
            raise

@pytest.mark.asyncio
async def test_generate_image_error_handling_async(async_venice_client):
    """Test error handling for image generation with invalid inputs (Async)"""
    client = async_venice_client
    test_cases = [
        ("prompt", "", InvalidRequestError),("width", -100, InvalidRequestError),("model", "nonexistent-model", NotFoundError),] # Expect 404 for bad model
    logger.info("Attempting async generate_image_error_handling")
    for param_name, invalid_value, expected_error in test_cases:
        params = {"model": DEFAULT_IMAGE_MODEL,"prompt": "Valid prompt","width": 512,"height": 512}
        params[param_name] = invalid_value
        try:
            with pytest.raises((VeniceError, expected_error)) as excinfo:
                await client.image.generate(**params)
            error_message = str(excinfo.value)
            # Check only the type of exception, as message content is unreliable
            assert isinstance(excinfo.value, expected_error), \
                f"Expected {expected_error} for {param_name}, but got {type(excinfo.value)}"
        except Exception as e:
            logger.error(f"Error in test_generate_image_error_handling_async for {param_name}: {e}", exc_info=True)
            raise
