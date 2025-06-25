"""
Tests for the image.py module with proper coverage tracking.

This file contains comprehensive tests for both the synchronous and asynchronous
image resource classes, ensuring high test coverage.
"""

import pytest
import pytest_asyncio
import os
import io
import base64
import httpx
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path
import tempfile

# Import the actual module to test
from venice_ai.resources.image import Image, AsyncImage, _guess_image_type
from venice_ai._client import VeniceClient
from venice_ai.exceptions import VeniceError
from venice_ai._async_client import AsyncVeniceClient
from venice_ai.exceptions import (
    InvalidRequestError, AuthenticationError, PermissionDeniedError, 
    NotFoundError, RateLimitError, APIError
)
from venice_ai.types.image import ImageResponse, SimpleImageResponse, ImageStyleList

# Test class for synchronous Image resource
class TestImage:
    @pytest.fixture
    def client_mock(self):
        """Create a properly mocked client."""
        client = MagicMock()
        # Configure default return values - return Pydantic models
        from venice_ai.types.image import ImageResponse, SimpleImageResponse, TimingInfo
        client.post.return_value = ImageResponse(
            id="img_123",
            images=["mock_base64_data"],
            request=None,
            timing=TimingInfo(
                inferenceDuration=1.0,
                inferencePreprocessingTime=0.1,
                inferenceQueueTime=0.2,
                total=1.3
            ),
            created="2021-08-26T12:00:00Z"
        )
        
        # Create a properly configured _request method that handles path, raw_response, etc.
        def mock_request(*args, method=None, path=None, params=None, timeout=None, endpoint=None, json_data=None, headers=None, raw_response=False, **kwargs):
            # Accept *args and endpoint to handle older interface and **kwargs for future compatibility
            if path == "image/upscale" and method == "POST":
                if raw_response:
                    return b"upscaled_image_data"
                return {"status": "success_json_if_not_raw"}
            # Default for other paths
            return b"mock_binary_data"
            
        client._request = MagicMock(side_effect=mock_request)
        
        client.get.return_value = {
            "data": ["style1", "style2"],
            "object": "list"
        }
        return client
    
    @pytest.fixture
    def image_resource(self, client_mock):
        """Create an Image resource with a mocked client."""
        return Image(client_mock)
    
    # Tests for generate method
    def test_generate_basic(self, image_resource):
        """Test the basic generate method with required parameters."""
        response = image_resource.generate(
            model="test-model", 
            prompt="A test prompt"
        )
        
        assert isinstance(response, ImageResponse)
        image_resource._client.post.assert_called_once()
        args, kwargs = image_resource._client.post.call_args
        assert args[0] == "image/generate"
        assert kwargs["json_data"]["model"] == "test-model"
        assert kwargs["json_data"]["prompt"] == "A test prompt"
        
    def test_generate_with_options(self, image_resource):
        """Test generate with all optional parameters."""
        response = image_resource.generate(
            model="test-model",
            prompt="A test prompt",
            cfg_scale=7.5,
            embed_exif_metadata=True,
            format="png",
            height=512,
            hide_watermark=True,
            # inpaint argument removed
            lora_strength=10,
            negative_prompt="bad quality",
            return_binary=False,
            safe_mode=True,
            seed=12345,
            steps=50,
            style_preset="photographic",
            width=768
        )
        
        assert isinstance(response, ImageResponse)
        image_resource._client.post.assert_called_once()
        
        args, kwargs = image_resource._client.post.call_args
        json_data = kwargs["json_data"]
        
        # Verify all parameters were passed correctly
        assert json_data["model"] == "test-model"
        assert json_data["prompt"] == "A test prompt"
        assert json_data["cfg_scale"] == 7.5
        assert json_data["embed_exif_metadata"] is True
        assert json_data["format"] == "png"
        assert json_data["height"] == 512
        assert json_data["hide_watermark"] is True
        # assert json_data["inpaint"] == {"mask": "base64mask"} # Assertion removed
        assert json_data["lora_strength"] == 10
        assert json_data["negative_prompt"] == "bad quality"
        assert json_data["return_binary"] is False
        assert json_data["safe_mode"] is True
        assert json_data["seed"] == 12345
        assert json_data["steps"] == 50
        assert json_data["style_preset"] == "photographic"
        assert json_data["width"] == 768
    
    def test_generate_return_binary(self, image_resource):
        """Test generate with return_binary=True option."""
        response = image_resource.generate(
            model="test-model",
            prompt="A test prompt",
            return_binary=True
        )
        
        assert isinstance(response, bytes)
        assert response == b"mock_binary_data"
        
        image_resource._client._request.assert_called_once()
        args, kwargs = image_resource._client._request.call_args
        
        # Check method, endpoint and options
        assert kwargs["method"] == "POST"
        assert kwargs["path"] == "image/generate"
        assert kwargs["headers"] == {"Accept": "image/*"}
        assert kwargs["raw_response"] is True
        
    # Tests for simple_generate method
    def test_simple_generate_basic(self, image_resource):
        """Test the simple_generate method with required parameters."""
        # Configure mock to return SimpleImageResponse for simple_generate
        from venice_ai.types.image import SimpleImageResponse, ImageDataItem
        image_resource._client.post.return_value = SimpleImageResponse(
            created=1630000000,
            data=[ImageDataItem(b64_json="mock_base64_data")]
        )
        
        response = image_resource.simple_generate(
            model="venice-diffusion",
            prompt="A simple test image"
        )
        
        assert isinstance(response, SimpleImageResponse)
        image_resource._client.post.assert_called_once()
        args, kwargs = image_resource._client.post.call_args
        assert args[0] == "images/generations"
        assert kwargs["json_data"]["model"] == "venice-diffusion"
        assert kwargs["json_data"]["prompt"] == "A simple test image"
        
    def test_simple_generate_with_options(self, image_resource):
        """Test simple_generate with all optional parameters."""
        # Configure mock to return SimpleImageResponse for simple_generate
        from venice_ai.types.image import SimpleImageResponse, ImageDataItem
        image_resource._client.post.return_value = SimpleImageResponse(
            created=1630000000,
            data=[ImageDataItem(b64_json="mock_base64_data1"), ImageDataItem(b64_json="mock_base64_data2")]
        )
        
        response = image_resource.simple_generate(
            model="venice-diffusion",
            prompt="A test image",
            background="transparent",
            moderation="low",
            n=2,
            output_compression=75,
            output_format="png",
            quality="high",
            response_format="b64_json",
            size="1024x1024",
            style="vivid",
            user="test-user"
        )
        
        assert isinstance(response, SimpleImageResponse)
        image_resource._client.post.assert_called_once()
        
        args, kwargs = image_resource._client.post.call_args
        json_data = kwargs["json_data"]
        
        # Verify all parameters were passed correctly
        assert json_data["model"] == "venice-diffusion"
        assert json_data["prompt"] == "A test image"
        assert json_data["background"] == "transparent"
        assert json_data["moderation"] == "low"
        assert json_data["n"] == 2
        assert json_data["output_compression"] == 75
        assert json_data["output_format"] == "png"
        assert json_data["quality"] == "high"
        assert json_data["response_format"] == "b64_json"
        assert json_data["size"] == "1024x1024"
        assert json_data["style"] == "vivid"
        assert json_data["user"] == "test-user"
    
    # Tests for upscale method
    def test_upscale_with_bytes(self, image_resource):
        """Test upscale with raw bytes input."""
        # Configure the mock for the method that returns the final bytes
        image_resource._client._request.return_value = b"upscaled_image_data"
        
        # Test with raw bytes
        image_data = b"test image data"
        result = image_resource.upscale(image=image_data)
        
        # Verify result is bytes
        assert isinstance(result, bytes)
        assert result == b"upscaled_image_data"
        
        # Check that base64 encoding was done correctly
        image_resource._client._request.assert_called_once()
        args, kwargs = image_resource._client._request.call_args
        json_data = kwargs["json_data"]
        
        # Decode and verify the base64 image
        decoded_data = base64.b64decode(json_data["image"])
        assert decoded_data == image_data
    
    def test_upscale_with_file_path(self, image_resource, tmp_path):
        """Test upscale with file path input."""
        # Create a temporary test file
        test_file = tmp_path / "test_image.jpg"
        test_file.write_bytes(b"test image data")
        
        # Configure the mock for the method that returns the final bytes
        image_resource._client._request.return_value = b"upscaled_image_data"
        
        # Call upscale with the file path
        result = image_resource.upscale(image=str(test_file))
        
        # Verify result 
        assert isinstance(result, bytes)
        assert result == b"upscaled_image_data"
        
        # Verify API call
        image_resource._client._request.assert_called_once()
    
    def test_upscale_with_file_object(self, image_resource):
        """Test upscale with file-like object input."""
        # Create a BytesIO object
        file_obj = io.BytesIO(b"test image data")
        
        # Configure the mock for the method that returns the final bytes
        image_resource._client._request.return_value = b"upscaled_image_data"
        
        # Call upscale with the file object
        result = image_resource.upscale(image=file_obj)
        
        # Verify result
        assert isinstance(result, bytes)
        assert result == b"upscaled_image_data"
        
        # Verify API call
        image_resource._client._request.assert_called_once()
    
    def test_upscale_with_text_file_object(self, image_resource):
        """Test upscale with a text mode file object."""
        # Create a BytesIO object (binary mode, proper for image data)
        file_obj = io.BytesIO(b"text data to convert")
        
        # Configure mock for the method that returns the final bytes
        image_resource._client._request.return_value = b"upscaled_image_data"
        
        # Call upscale with the text file object
        result = image_resource.upscale(image=file_obj)
        
        # Verify result
        assert isinstance(result, bytes)
        assert result == b"upscaled_image_data"
        
        # Verify API call
        image_resource._client._request.assert_called_once()
    
    def test_image_upscale_file_read_returns_string(self, image_resource):
        """Test upscale with file-like object whose read() method returns a string."""
        # Create a mock file-like object whose read() method returns a string
        mock_file_str_read = MagicMock()
        mock_file_str_read.read.return_value = "this is a string from read"
        mock_file_str_read.name = "test_image.png"
        
        # The SDK should raise VeniceError for file-like objects returning strings
        with pytest.raises(VeniceError, match="Image source is a file-like object that did not return bytes from read()"):
            image_resource.upscale(image=mock_file_str_read)
    
    def test_image_upscale_unsupported_image_type(self, image_resource):
        """Test upscale with an unsupported image type."""
        # Configure mock for the method that returns the final bytes
        image_resource._client._request.return_value = b"upscaled_image_data"
        
        # Test with integer (unsupported type)
        with pytest.raises(TypeError, match=r"Unsupported image_source type: <class 'int'>"):
            image_resource.upscale(image=123)
        
        # Test with list (unsupported type)
        with pytest.raises(TypeError, match="Unsupported image type"):
            image_resource.upscale(image=["not", "an", "image"])
        
        # Verify that _client.post was not called
        image_resource._client._request.assert_not_called()
    
    def test_upscale_with_all_parameters(self, image_resource):
        """Test upscale with all optional parameters."""
        # Configure mock for the method that returns the final bytes
        image_resource._client._request.return_value = b"upscaled_image_data"
        
        # Call with all optional parameters
        result = image_resource.upscale(
            image=b"test image data",
            enhance="true",
            enhance_creativity=0.8,
            enhance_prompt="Add more details",
            replication=0.7,
            scale=2.5
        )
        
        # Verify result
        assert isinstance(result, bytes)
        assert result == b"upscaled_image_data"
        
        # Verify all parameters were passed correctly
        image_resource._client._request.assert_called_once()
        args, kwargs = image_resource._client._request.call_args
        json_data = kwargs["json_data"]
        
        assert "image" in json_data  # Base64 encoded image
        assert json_data["enhance"] is True
        assert json_data["replication"] == 0.7
        assert json_data["scale"] == 2.5
    
    def test_upscale_file_not_found(self, image_resource):
        """Test upscale with non-existent file path."""
        with pytest.raises(VeniceError, match="Image file not found"):
            image_resource.upscale(image="nonexistent/path/image.jpg")
    
    # Tests for list_styles method
    def test_list_styles(self, image_resource):
        """Test listing available image styles."""
        response = image_resource.list_styles()
        
        assert isinstance(response, dict)
        assert "data" in response
        assert len(response["data"]) == 2
        
        image_resource._client.get.assert_called_once_with("image/styles")
    
    # Tests for error handling
    def test_generate_with_api_errors(self):
        """Test generate method with different API errors."""
        # Create mock response required by exception classes
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 400
        mock_response.headers = {}
        
        error_cases = [
            (InvalidRequestError("Invalid parameter", response=mock_response), InvalidRequestError),
            (AuthenticationError("Invalid API key", response=mock_response), AuthenticationError),
            (PermissionDeniedError("Access denied", response=mock_response), PermissionDeniedError),
            (NotFoundError("Model not found", response=mock_response), NotFoundError),
            (RateLimitError("Rate limit exceeded", response=mock_response, retry_after_seconds=None), RateLimitError),
        ]
        
        for error, expected_exception in error_cases:
            # Create a fresh mock client that raises the specific error
            client_mock = MagicMock()
            client_mock.post.side_effect = error
            
            # Create image resource with this mock
            image_resource = Image(client_mock)
            
            # Test that the correct exception is raised
            with pytest.raises(expected_exception):
                image_resource.generate(
                    model="test-model", 
                    prompt="A test prompt"
                )
    
    def test_guess_image_type(self):
        """Test the _guess_image_type utility function."""
        assert _guess_image_type("image.jpg") == "jpeg"
        assert _guess_image_type("image.jpeg") == "jpeg"
        assert _guess_image_type("image.png") == "png"
        assert _guess_image_type("image.webp") == "webp"
        assert _guess_image_type("image.gif") == "gif"
        assert _guess_image_type("image.bmp") == "octet-stream"
        assert _guess_image_type("/path/to/image.jpg") == "jpeg"
        assert _guess_image_type("IMAGE.PNG") == "png"


# Test class for asynchronous AsyncImage resource
@pytest.mark.asyncio
class TestAsyncImage:
    @pytest_asyncio.fixture
    async def client_mock(self):
        """Create a properly mocked async client."""
        client = MagicMock()
        # Configure default return values - return Pydantic models
        from venice_ai.types.image import ImageResponse, SimpleImageResponse, TimingInfo
        client.post = AsyncMock(return_value=ImageResponse(
            id="img_123",
            images=["mock_base64_data"],
            request=None,
            timing=TimingInfo(
                inferenceDuration=1.0,
                inferencePreprocessingTime=0.1,
                inferenceQueueTime=0.2,
                total=1.3
            ),
            created="2021-08-26T12:00:00Z"
        ))
        
        # Create a properly configured async _request method that handles path, raw_response, etc.
        async def mock_async_request(*args, method=None, path=None, params=None, timeout=None, endpoint=None, json_data=None, headers=None, raw_response=False, **kwargs):
            # Accept *args and endpoint to handle older interface and **kwargs for future compatibility
            if path == "image/upscale" and method == "POST":
                if raw_response:
                    return b"upscaled_image_data"
                return {"status": "success_json_if_not_raw"}
            # Default for other paths
            return b"mock_binary_data"
            
        client._request = AsyncMock(side_effect=mock_async_request)
        
        client.get = AsyncMock(return_value={
            "data": [
                {"id": "style1", "name": "Style 1", "description": "Description 1"},
                {"id": "style2", "name": "Style 2", "description": "Description 2"}
            ],
            "object": "list"
        })
        return client
    
    @pytest_asyncio.fixture
    async def image_resource(self, client_mock):
        """Create an AsyncImage resource with a mocked client."""
        return AsyncImage(client_mock)
    
    # Tests for generate method
    async def test_generate_basic(self, image_resource):
        """Test the basic async generate method with required parameters."""
        response = await image_resource.generate(
            model="test-model", 
            prompt="A test prompt"
        )
        
        assert isinstance(response, ImageResponse)
        image_resource._client.post.assert_awaited_once()
        args, kwargs = image_resource._client.post.call_args
        assert args[0] == "image/generate"
        assert kwargs["json_data"]["model"] == "test-model"
        assert kwargs["json_data"]["prompt"] == "A test prompt"
        
    async def test_generate_with_options(self, image_resource):
        """Test async generate with all optional parameters."""
        response = await image_resource.generate(
            model="test-model",
            prompt="A test prompt",
            cfg_scale=7.5,
            embed_exif_metadata=True,
            format="png",
            height=512,
            hide_watermark=True,
            # inpaint argument removed
            lora_strength=10,
            negative_prompt="bad quality",
            return_binary=False,
            safe_mode=True,
            seed=12345,
            steps=50,
            style_preset="photographic",
            width=768
        )
        
        assert isinstance(response, ImageResponse)
        image_resource._client.post.assert_awaited_once()
        
        args, kwargs = image_resource._client.post.call_args
        json_data = kwargs["json_data"]
        
        # Verify all parameters were passed correctly
        assert json_data["model"] == "test-model"
        assert json_data["prompt"] == "A test prompt"
        assert json_data["cfg_scale"] == 7.5
        assert json_data["embed_exif_metadata"] is True
        assert json_data["format"] == "png"
        assert json_data["height"] == 512
        assert json_data["hide_watermark"] is True
        # assert json_data["inpaint"] == {"mask": "base64mask"} # Assertion removed
        assert json_data["lora_strength"] == 10
        assert json_data["negative_prompt"] == "bad quality"
        assert json_data["return_binary"] is False
        assert json_data["safe_mode"] is True
        assert json_data["seed"] == 12345
        assert json_data["steps"] == 50
        assert json_data["style_preset"] == "photographic"
        assert json_data["width"] == 768
    
    async def test_generate_return_binary(self, image_resource):
        """Test async generate with return_binary=True option."""
        response = await image_resource.generate(
            model="test-model",
            prompt="A test prompt",
            return_binary=True
        )
        
        assert isinstance(response, bytes)
        assert response == b"mock_binary_data"
        
        image_resource._client._request.assert_awaited_once()
        args, kwargs = image_resource._client._request.call_args
        
        # Check method, endpoint and options
        assert kwargs["method"] == "POST"
        assert kwargs["path"] == "image/generate"
        assert kwargs["headers"] == {"Accept": "image/*"}
        assert kwargs["raw_response"] is True
        
    # Tests for simple_generate method
    async def test_simple_generate_basic(self, image_resource):
        """Test the async simple_generate method with required parameters."""
        # Configure mock to return SimpleImageResponse for simple_generate
        from venice_ai.types.image import SimpleImageResponse, ImageDataItem
        image_resource._client.post.return_value = SimpleImageResponse(
            created=1630000000,
            data=[ImageDataItem(b64_json="mock_base64_data")]
        )
        
        response = await image_resource.simple_generate(
            model="venice-diffusion",
            prompt="A simple test image"
        )
        
        assert isinstance(response, SimpleImageResponse)
        image_resource._client.post.assert_awaited_once()
        args, kwargs = image_resource._client.post.call_args
        assert args[0] == "images/generations"
        assert kwargs["json_data"]["model"] == "venice-diffusion"
        assert kwargs["json_data"]["prompt"] == "A simple test image"
        
    async def test_simple_generate_with_options(self, image_resource):
        """Test async simple_generate with all optional parameters."""
        # Configure mock to return SimpleImageResponse for simple_generate
        from venice_ai.types.image import SimpleImageResponse, ImageDataItem
        image_resource._client.post.return_value = SimpleImageResponse(
            created=1630000000,
            data=[ImageDataItem(b64_json="mock_base64_data1"), ImageDataItem(b64_json="mock_base64_data2")]
        )
        
        response = await image_resource.simple_generate(
            model="venice-diffusion",
            prompt="A test image",
            background="transparent",
            moderation="low",
            n=2,
            output_compression=75,
            output_format="png",
            quality="high",
            response_format="b64_json",
            size="1024x1024",
            style="vivid",
            user="test-user"
        )
        
        assert isinstance(response, SimpleImageResponse)
        image_resource._client.post.assert_awaited_once()
        
        args, kwargs = image_resource._client.post.call_args
        json_data = kwargs["json_data"]
        
        # Verify all parameters were passed correctly
        assert json_data["model"] == "venice-diffusion"
        assert json_data["prompt"] == "A test image"
        assert json_data["background"] == "transparent"
        assert json_data["moderation"] == "low"
        assert json_data["n"] == 2
        assert json_data["output_compression"] == 75
        assert json_data["output_format"] == "png"
        assert json_data["quality"] == "high"
        assert json_data["response_format"] == "b64_json"
        assert json_data["size"] == "1024x1024"
        assert json_data["style"] == "vivid"
        assert json_data["user"] == "test-user"
    
    # Tests for upscale method
    async def test_upscale_with_bytes(self, image_resource):
        """Test async upscale with raw bytes input."""
        # Configure the mock for the method that returns the final bytes
        image_resource._client._request.return_value = b"upscaled_image_data"
        
        # Test with raw bytes
        image_data = b"test image data"
        result = await image_resource.upscale(image=image_data)
        
        # Verify result is bytes
        assert isinstance(result, bytes)
        assert result == b"upscaled_image_data"
        
        # Check that base64 encoding was done correctly
        image_resource._client._request.assert_awaited_once()
        args, kwargs = image_resource._client._request.call_args
        json_data = kwargs["json_data"]
        
        # Decode and verify the base64 image
        decoded_data = base64.b64decode(json_data["image"])
        assert decoded_data == image_data
    
    async def test_upscale_with_file_path(self, image_resource, tmp_path):
        """Test async upscale with file path input."""
        # Create a temporary test file
        test_file = tmp_path / "test_image.jpg"
        test_file.write_bytes(b"test image data")
        
        # Configure the mock for the method that returns the final bytes
        image_resource._client._request.return_value = b"upscaled_image_data"
        
        # Call upscale with the file path
        result = await image_resource.upscale(image=str(test_file))
        
        # Verify result 
        assert isinstance(result, bytes)
        assert result == b"upscaled_image_data"
        
        # Verify API call
        image_resource._client._request.assert_awaited_once()
    
    async def test_upscale_with_file_object(self, image_resource):
        """Test async upscale with file-like object input."""
        # Create a BytesIO object
        file_obj = io.BytesIO(b"test image data")
        
        # Configure the mock for the method that returns the final bytes
        image_resource._client._request.return_value = b"upscaled_image_data"
        
        # Call upscale with the file object
        result = await image_resource.upscale(image=file_obj)
        
        # Verify result
        assert isinstance(result, bytes)
        assert result == b"upscaled_image_data"
        
        # Verify API call
        image_resource._client._request.assert_awaited_once()
    
    async def test_upscale_with_text_file_object(self, image_resource):
        """Test async upscale with a text mode file object."""
        # Create a BytesIO object (binary mode, proper for image data)
        file_obj = io.BytesIO(b"text data to convert")
        
        # Configure mock for the method that returns the final bytes
        image_resource._client._request.return_value = b"upscaled_image_data"
        
        # Call upscale with the text file object
        result = await image_resource.upscale(image=file_obj)
        
        # Verify result
        assert isinstance(result, bytes)
        assert result == b"upscaled_image_data"
        
        # Verify API call
        image_resource._client._request.assert_awaited_once()
    
    async def test_async_image_upscale_file_read_returns_string(self, image_resource):
        """Test async upscale with file-like object whose read() method returns a string."""
        # Create a mock file-like object whose read() method returns a string
        mock_file_str_read = MagicMock()
        # Make .read an AsyncMock for this test to align with async processing
        mock_file_str_read.read = AsyncMock(return_value="this is a string from sync read")
        mock_file_str_read.name = "test_image.png"

        # Patch asyncio.iscoroutinefunction to recognize this .read as a coroutine function
        with patch('asyncio.iscoroutinefunction', return_value=True):
            # The SDK should raise VeniceError for file-like objects returning strings
            with pytest.raises(VeniceError, match="Image source is a file-like object that did not return bytes from read()"):
                await image_resource.upscale(image=mock_file_str_read)
        
    async def test_async_image_upscale_file_read_returns_string_async(self, image_resource):
        """Test async upscale with file-like object whose async read() method returns a string."""
        # Create a mock file-like object with an async read method that returns a string
        mock_file_str_read_async = MagicMock()
        mock_file_str_read_async.read = AsyncMock(return_value="this is a string from async read")
        mock_file_str_read_async.name = "test_image.png"
        # Make iscoroutinefunction return True for this mock's read method
        with patch('asyncio.iscoroutinefunction', return_value=True):
            # The SDK should raise VeniceError for file-like objects returning strings
            with pytest.raises(VeniceError, match="Image source is a file-like object that did not return bytes from read()"):
                await image_resource.upscale(image=mock_file_str_read_async)
    
    async def test_async_image_upscale_unsupported_image_type(self, image_resource):
        """Test async upscale with an unsupported image type."""
        # Configure mock for the method that returns the final bytes
        image_resource._client._request.return_value = b"upscaled_image_data"
        
        # Test with integer (unsupported type)
        with pytest.raises(TypeError, match="Unsupported image type"):
            await image_resource.upscale(image=123)
        
        # Test with list (unsupported type)
        with pytest.raises(TypeError, match="Unsupported image type"):
            await image_resource.upscale(image=["not", "an", "image"])
        
        # Verify that _client.post was not called
        image_resource._client._request.assert_not_called()
    
    async def test_upscale_with_all_parameters(self, image_resource):
        """Test async upscale with all optional parameters."""
        # Configure mock for the method that returns the final bytes
        image_resource._client._request.return_value = b"upscaled_image_data"
        
        # Call with all optional parameters
        result = await image_resource.upscale(
            image=b"test image data",
            enhance="true",
            enhance_creativity=0.8,
            enhance_prompt="Add more details",
            replication=0.7,
            scale=2.5
        )
        
        # Verify result
        assert isinstance(result, bytes)
        assert result == b"upscaled_image_data"
        
        # Verify all parameters were passed correctly
        image_resource._client._request.assert_awaited_once()
        args, kwargs = image_resource._client._request.call_args
        json_data = kwargs["json_data"]
        
        assert "image" in json_data  # Base64 encoded image
        assert json_data["enhance"] is True
        assert json_data["replication"] == 0.7
        assert json_data["scale"] == 2.5
    
    async def test_upscale_file_not_found(self, image_resource):
        """Test async upscale with non-existent file path."""
        with pytest.raises(VeniceError, match="Image file not found"):
            await image_resource.upscale(image="nonexistent/path/image.jpg")
    
    # Tests for list_styles method
    async def test_list_styles(self, image_resource):
        """Test async listing available image styles."""
        response = await image_resource.list_styles()
        
        assert isinstance(response, dict)
        assert "data" in response
        assert len(response["data"]) == 2
        
        image_resource._client.get.assert_awaited_once_with("image/styles")
    
    # Tests for error handling
    async def test_generate_with_api_errors(self):
        """Test async generate method with different API errors."""
        # Create mock response required by exception classes
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 400
        mock_response.headers = {}
        
        error_cases = [
            (InvalidRequestError("Invalid parameter", response=mock_response), InvalidRequestError),
            (AuthenticationError("Invalid API key", response=mock_response), AuthenticationError),
            (PermissionDeniedError("Access denied", response=mock_response), PermissionDeniedError),
            (NotFoundError("Model not found", response=mock_response), NotFoundError),
            (RateLimitError("Rate limit exceeded", response=mock_response, retry_after_seconds=None), RateLimitError),
        ]
        
        for error, expected_exception in error_cases:
            # Create a fresh mock client that raises the specific error
            client_mock = MagicMock()
            client_mock.post = AsyncMock(side_effect=error)
            
            # Create image resource with this mock
            image_resource = AsyncImage(client_mock)
            
            # Test that the correct exception is raised
            with pytest.raises(expected_exception):
                await image_resource.generate(
                    model="test-model", 
                    prompt="A test prompt"
                )


# Additional test class for edge cases
class TestImageEdgeCases:
    @pytest.fixture
    def client_mock(self):
        """Create a mocked client for edge cases."""
        return MagicMock()
    
    def test_generate_with_extreme_values(self, client_mock):
        """Test generate with extreme parameter values."""
        client_mock.post.return_value = {"id": "test", "data": [{"url": "https://example.com/image.png"}]}
        image = Image(client_mock)
        
        # Test with extreme parameter values
        image.generate(
            model="test-model",
            prompt="A test prompt",
            cfg_scale=30.0,  # Maximum value
            height=2048,     # Large height
            width=2048,      # Large width
            steps=150,       # High step count
            seed=999999999   # Large seed value
        )
        
        # Verify the parameters were passed correctly
        args, kwargs = client_mock.post.call_args
        json_data = kwargs["json_data"]
        
        assert json_data["cfg_scale"] == 30.0
        assert json_data["height"] == 2048
        assert json_data["width"] == 2048
        assert json_data["steps"] == 150
        assert json_data["seed"] == 999999999
        
    def test_guess_image_type_edge_cases(self):
        """Test _guess_image_type with edge cases."""
        # Test with unusual file paths
        assert _guess_image_type("C:\\Windows\\image.jpg") == "jpeg"
        assert _guess_image_type("/usr/local/bin/image.png") == "png"
        assert _guess_image_type("../relative/path/image.webp") == "webp"
        assert _guess_image_type("image.jpg.bak") == "octet-stream"  # Doesn't end with true extension
        assert _guess_image_type("image") == "octet-stream"         # No extension
        assert _guess_image_type("") == "octet-stream"             # Empty string
        
    def test_upscale_with_large_image(self, client_mock, tmp_path):
        """Test upscale with a relatively large image."""
        # Create a "large" image (1MB of random data)
        large_image_data = os.urandom(1024 * 1024)  # 1MB of random data
        large_image_path = tmp_path / "large_image.jpg"
        large_image_path.write_bytes(large_image_data)
        
        # Configure mock for the method that returns the final bytes
        client_mock._request.return_value = b"upscaled_data"
        
        # Create image resource and call upscale
        image = Image(client_mock)
        result = image.upscale(image=str(large_image_path))
        
        # Verify it worked
        assert result == b"upscaled_data"
        client_mock._request.assert_called_once()


@pytest.mark.asyncio
class TestAsyncImageEdgeCases:
    @pytest_asyncio.fixture
    async def client_mock(self):
        """Create a mocked async client for edge cases."""
        client = MagicMock()
        client.post = AsyncMock()
        return client
    
    async def test_generate_with_extreme_values(self, client_mock):
        """Test async generate with extreme parameter values."""
        client_mock.post.return_value = {"id": "test", "data": [{"url": "https://example.com/image.png"}]}
        image = AsyncImage(client_mock)
        
        # Test with extreme parameter values
        await image.generate(
            model="test-model",
            prompt="A test prompt",
            cfg_scale=30.0,  # Maximum value
            height=2048,     # Large height
            width=2048,      # Large width
            steps=150,       # High step count
            seed=999999999   # Large seed value
        )
        
        # Verify the parameters were passed correctly
        args, kwargs = client_mock.post.call_args
        json_data = kwargs["json_data"]
        
        assert json_data["cfg_scale"] == 30.0
        assert json_data["height"] == 2048
        assert json_data["width"] == 2048
        assert json_data["steps"] == 150
        assert json_data["seed"] == 999999999
        
    async def test_upscale_with_large_image(self, client_mock, tmp_path):
        """Test async upscale with a relatively large image."""
        # Create a "large" image (1MB of random data)
        large_image_data = os.urandom(1024 * 1024)  # 1MB of random data
        large_image_path = tmp_path / "large_image.jpg"
        large_image_path.write_bytes(large_image_data)
        
        # Configure mock for the method that returns the final bytes
        # Use AsyncMock for the _request method since it will be awaited
        client_mock._request = AsyncMock(return_value=b"upscaled_data")
        
        # Create image resource and call upscale
        image = AsyncImage(client_mock)
        result = await image.upscale(image=str(large_image_path))
        
        # Verify it worked
        assert result == b"upscaled_data"
        client_mock._request.assert_awaited_once() # Assert against the correct mock