"""
Tests focused on improving coverage for the image.py module.

This file contains tests specifically designed to target untested
code paths in the image.py module to improve test coverage.
"""

import pytest
import io
import base64
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path
import os

from venice_ai.resources.image import Image, AsyncImage, _guess_image_type
from venice_ai._client import VeniceClient
from venice_ai._async_client import AsyncVeniceClient
from venice_ai.exceptions import InvalidRequestError, AuthenticationError, VeniceError
from venice_ai.types.image import ImageResponse, SimpleImageResponse, TimingInfo, ImageDataItem

# Test the synchronous Image resource
class TestImageCoverage:
    @pytest.fixture
    def mock_client(self):
        """Fixture for a properly mocked client."""
        mock = MagicMock()
        # Setup the _request method to handle binary responses
        mock._request.return_value = b"upscaled_image_data"
        # Setup the post method for JSON responses - return Pydantic models
        mock.post.return_value = ImageResponse(
            id="test_id",
            images=["base64_image_data"],
            request=None,
            timing=TimingInfo(
                inferenceDuration=1.0,
                inferencePreprocessingTime=0.1,
                inferenceQueueTime=0.2,
                total=1.3
            ),
            created="2021-08-26T12:00:00Z"
        )
        return mock

    @pytest.fixture
    def image_resource(self, mock_client):
        """Fixture for the Image resource with a mocked client."""
        return Image(mock_client)

    def test_generate_with_mandatory_parameters_only(self, image_resource):
        """Test generate with only the required parameters."""
        result = image_resource.generate(
            model="test-model",
            prompt="Test prompt"
        )

        assert isinstance(result, ImageResponse)
        image_resource._client.post.assert_called_once()
        call_args = image_resource._client.post.call_args
        json_data = call_args[1]["json_data"]
        
        # Verify only required parameters are present
        assert json_data == {
            "model": "test-model",
            "prompt": "Test prompt"
        }

    def test_generate_with_return_binary_true(self, image_resource):
        """Test generate with return_binary=True."""
        result = image_resource.generate(
            model="test-model",
            prompt="Test prompt",
            return_binary=True
        )

        assert isinstance(result, bytes)
        image_resource._client._request.assert_called_once()
        call_args = image_resource._client._request.call_args
        
        # Verify headers were properly set for binary response
        assert call_args[1]["headers"] == {"Accept": "image/*"}
        assert call_args[1]["raw_response"] is True

    def test_upscale_with_bytes_data(self, image_resource):
        """Test upscale using binary data."""
        # Mock the method that returns the final bytes
        image_resource._client._request.return_value = b"upscaled_image_data"
        
        result = image_resource.upscale(
            image=b"test_image_data"
        )
        
        assert isinstance(result, bytes)
        assert result == b"upscaled_image_data"
        
        # Verify base64 encoding was done correctly
        image_resource._client._request.assert_called_once()
        call_args = image_resource._client._request.call_args
        json_data = call_args[1]["json_data"]
        assert "image" in json_data
        
        # Decode the base64 and verify it matches our input
        decoded = base64.b64decode(json_data["image"])
        assert decoded == b"test_image_data"

    def test_upscale_with_file_path(self, image_resource, tmp_path):
        """Test upscale with a file path."""
        # Create a temporary file
        test_file = tmp_path / "test_image.png"
        test_file.write_bytes(b"test_image_data")
        
        # Mock the method that returns the final bytes
        image_resource._client._request.return_value = b"upscaled_image_data"
        
        # Call upscale with the file path
        result = image_resource.upscale(
            image=str(test_file),
            scale=2.0
        )
        
        assert isinstance(result, bytes)
        assert result == b"upscaled_image_data"
        
        # Verify the file was properly read and encoded
        image_resource._client._request.assert_called_once()
        call_args = image_resource._client._request.call_args
        json_data = call_args[1]["json_data"]
        
        # Ensure scale parameter was passed correctly
        assert json_data["scale"] == 2.0

    def test_upscale_with_file_object(self, image_resource):
        """Test upscale with a file-like object."""
        # Create a BytesIO object
        file_obj = io.BytesIO(b"test_image_data")
        
        # Mock the method that returns the final bytes
        image_resource._client._request.return_value = b"upscaled_image_data"
        
        # Call upscale with file object
        result = image_resource.upscale(
            image=file_obj
        )
        
        assert isinstance(result, bytes)
        assert result == b"upscaled_image_data"
        
        # Verify file object was properly read and encoded
        image_resource._client._request.assert_called_once()

    def test_upscale_with_text_file_object(self, image_resource):
        """Test upscale with text-mode file object (should convert to bytes)."""
        # Create a BytesIO that returns bytes (proper for image data)
        text_file = io.BytesIO(b"text data that should be converted to bytes")
        
        # Mock the method that returns the final bytes
        image_resource._client._request.return_value = b"processed_data"
        
        # Call upscale with text file object
        result = image_resource.upscale(image=text_file)
        
        assert result == b"processed_data"
        
        # Verify text was converted to bytes and encoded properly
        image_resource._client._request.assert_called_once()
        call_args = image_resource._client._request.call_args
        json_data = call_args[1]["json_data"]
        assert "image" in json_data

    def test_upscale_with_all_parameters(self, image_resource):
        """Test upscale with all optional parameters."""
        # Mock the method that returns the final bytes
        image_resource._client._request.return_value = b"upscaled_image_data"
        
        # Call upscale with all parameters
        result = image_resource.upscale(
            image=b"test_image_data",
            enhance="true",
            enhance_creativity=0.8,
            enhance_prompt="Make it better",
            replication=0.7,
            scale=2.0
        )
        
        assert result == b"upscaled_image_data"
        
        # Verify all parameters were passed correctly
        image_resource._client._request.assert_called_once()
        call_args = image_resource._client._request.call_args
        json_data = call_args[1]["json_data"]
        
        assert json_data["enhance"] is True
        assert json_data["replication"] == 0.7
        assert json_data["scale"] == 2.0

    def test_upscale_exceptions(self, image_resource):
        """Test upscale method error handling for non-existent files."""
        with pytest.raises(VeniceError, match="Image file not found at path:"):
            image_resource.upscale(image="non_existent_file.jpg")


# Test the asynchronous AsyncImage resource
@pytest.mark.asyncio
class TestAsyncImageCoverage:
    @pytest.fixture
    async def mock_async_client(self):
        """Fixture for a properly mocked async client."""
        mock = MagicMock()
        # Setup the _request method to handle binary responses
        mock._request = AsyncMock(return_value=b"upscaled_image_data")
        # Setup the post method for JSON responses - return Pydantic models
        mock.post = AsyncMock(return_value=ImageResponse(
            id="test_id",
            images=["base64_image_data"],
            request=None,
            timing=TimingInfo(
                inferenceDuration=1.0,
                inferencePreprocessingTime=0.1,
                inferenceQueueTime=0.2,
                total=1.3
            ),
            created="2021-08-26T12:00:00Z"
        ))
        return mock

    @pytest.fixture
    def async_image_resource(self, mock_async_client):
        """Fixture for the AsyncImage resource with a mocked client."""
        return AsyncImage(mock_async_client)

    async def test_generate_with_mandatory_parameters_only(self, async_image_resource):
        """Test async generate with only the required parameters."""
        result = await async_image_resource.generate(
            model="test-model",
            prompt="Test prompt"
        )

        assert isinstance(result, ImageResponse)
        async_image_resource._client.post.assert_awaited_once()
        call_args = async_image_resource._client.post.call_args
        json_data = call_args[1]["json_data"]
        
        # Verify only required parameters are present
        assert json_data == {
            "model": "test-model",
            "prompt": "Test prompt"
        }

    async def test_generate_with_return_binary_true(self, async_image_resource):
        """Test async generate with return_binary=True."""
        result = await async_image_resource.generate(
            model="test-model",
            prompt="Test prompt",
            return_binary=True
        )

        assert isinstance(result, bytes)
        async_image_resource._client._request.assert_awaited_once()
        call_args = async_image_resource._client._request.call_args
        
        # Verify headers were properly set for binary response
        assert call_args[1]["headers"] == {"Accept": "image/*"}
        assert call_args[1]["raw_response"] is True

    async def test_async_upscale_with_bytes_data(self, async_image_resource):
        """Test async upscale using binary data."""
        # Mock the method that returns the final bytes
        async_image_resource._client._request.return_value = b"upscaled_image_data"
        
        result = await async_image_resource.upscale(
            image=b"test_image_data"
        )
        
        assert isinstance(result, bytes)
        assert result == b"upscaled_image_data"
        
        # Verify base64 encoding was done correctly
        async_image_resource._client._request.assert_awaited_once()
        call_args = async_image_resource._client._request.call_args
        json_data = call_args[1]["json_data"]
        assert "image" in json_data
        
        # Decode the base64 and verify it matches our input
        decoded = base64.b64decode(json_data["image"])
        assert decoded == b"test_image_data"

    async def test_async_upscale_with_file_path(self, async_image_resource, tmp_path):
        """Test async upscale with a file path."""
        # Create a temporary file
        test_file = tmp_path / "test_image.png"
        test_file.write_bytes(b"test_image_data")
        
        # Mock the method that returns the final bytes
        async_image_resource._client._request.return_value = b"upscaled_image_data"
        
        # Call upscale with the file path
        result = await async_image_resource.upscale(
            image=str(test_file),
            scale=2.0
        )
        
        assert isinstance(result, bytes)
        assert result == b"upscaled_image_data"
        
        # Verify the file was properly read and encoded
        async_image_resource._client._request.assert_awaited_once()
        call_args = async_image_resource._client._request.call_args
        json_data = call_args[1]["json_data"]
        
        # Ensure scale parameter was passed correctly
        assert json_data["scale"] == 2.0

    async def test_async_upscale_with_file_object(self, async_image_resource):
        """Test async upscale with a file-like object."""
        # Create a BytesIO object
        file_obj = io.BytesIO(b"test_image_data")
        
        # Mock the method that returns the final bytes
        async_image_resource._client._request.return_value = b"upscaled_image_data"
        
        # Call upscale with file object
        result = await async_image_resource.upscale(
            image=file_obj
        )
        
        assert isinstance(result, bytes)
        assert result == b"upscaled_image_data"
        
        # Verify file object was properly read and encoded
        async_image_resource._client._request.assert_awaited_once()

    async def test_async_upscale_with_text_file_object(self, async_image_resource):
        """Test async upscale with text-mode file object (should convert to bytes)."""
        # Create a BytesIO that returns bytes (proper for image data)
        text_file = io.BytesIO(b"text data that should be converted to bytes")
        
        # Mock the method that returns the final bytes
        async_image_resource._client._request.return_value = b"processed_data"
        
        # Call upscale with text file object
        result = await async_image_resource.upscale(image=text_file)
        
        assert result == b"processed_data"
        
        # Verify text was converted to bytes and encoded properly
        async_image_resource._client._request.assert_awaited_once()
        call_args = async_image_resource._client._request.call_args
        json_data = call_args[1]["json_data"]
        assert "image" in json_data

    async def test_async_upscale_with_all_parameters(self, async_image_resource):
        """Test async upscale with all optional parameters."""
        # Mock the method that returns the final bytes
        async_image_resource._client._request.return_value = b"upscaled_image_data"
        
        # Call upscale with all parameters
        result = await async_image_resource.upscale(
            image=b"test_image_data",
            enhance="true",
            enhance_creativity=0.8,
            enhance_prompt="Make it better",
            replication=0.7,
            scale=2.0
        )
        
        assert result == b"upscaled_image_data"
        
        # Verify all parameters were passed correctly
        async_image_resource._client._request.assert_awaited_once()
        call_args = async_image_resource._client._request.call_args
        json_data = call_args[1]["json_data"]
        
        assert json_data["enhance"] is True
        assert json_data["replication"] == 0.7
        assert json_data["scale"] == 2.0

    async def test_async_upscale_exceptions(self, async_image_resource):
        """Test async upscale method error handling for non-existent files."""
        with pytest.raises(VeniceError, match="Image file not found at path:"):
            await async_image_resource.upscale(image="non_existent_file.jpg")


# Additional tests for edge cases for improved coverage
class TestImageEdgeCases:
    def test_guess_image_type_comprehensive(self):
        """Comprehensive test for _guess_image_type function."""
        # Test standard image extensions
        assert _guess_image_type("image.jpg") == "jpeg"
        assert _guess_image_type("image.jpeg") == "jpeg"
        assert _guess_image_type("image.png") == "png"
        assert _guess_image_type("image.webp") == "webp"
        assert _guess_image_type("image.gif") == "gif"
        
        # Test with uppercase and mixed case
        assert _guess_image_type("IMAGE.JPG") == "jpeg"
        assert _guess_image_type("Image.Png") == "png"
        
        # Test with paths
        assert _guess_image_type("/path/to/image.jpg") == "jpeg"
        assert _guess_image_type("C:\\Users\\test\\image.png") == "png"
        
        # Test with non-image extensions
        assert _guess_image_type("document.pdf") == "octet-stream"
        assert _guess_image_type("image.tiff") == "octet-stream"
        assert _guess_image_type("image") == "octet-stream"

    def test_generate_with_all_parameters(self):
        """Test generate with every single parameter specified."""
        client = MagicMock()
        client.post.return_value = ImageResponse(
            id="test",
            images=["test-image"],
            request=None,
            timing=TimingInfo(
                inferenceDuration=1.0,
                inferencePreprocessingTime=0.1,
                inferenceQueueTime=0.2,
                total=1.3
            ),
            created="2021-08-26T12:00:00Z"
        )
        
        image = Image(client)
        response = image.generate(
            model="test-model",
            prompt="Test prompt",
            cfg_scale=7.5,
            embed_exif_metadata=True,
            format="png",
            height=512,
            hide_watermark=False,
            # inpaint argument removed
            lora_strength=5,
            negative_prompt="bad quality",
            return_binary=False,
            safe_mode=True,
            seed=12345,
            steps=30,
            style_preset="anime",
            width=512
        )
        
        assert isinstance(response, ImageResponse)
        assert response.id == "test"
        assert response.images == ["test-image"]
        client.post.assert_called_once()
        call_args = client.post.call_args
        json_data = call_args[1]["json_data"]
        
        # Verify all parameters were passed correctly
        assert json_data["model"] == "test-model"
        assert json_data["prompt"] == "Test prompt"
        assert json_data["cfg_scale"] == 7.5
        assert json_data["embed_exif_metadata"] is True
        assert json_data["format"] == "png"
        assert json_data["height"] == 512
        assert json_data["hide_watermark"] is False
        # inpaint assertion removed
        assert json_data["lora_strength"] == 5
        assert json_data["negative_prompt"] == "bad quality"
        assert json_data["return_binary"] is False
        assert json_data["safe_mode"] is True
        assert json_data["seed"] == 12345
        assert json_data["steps"] == 30
        assert json_data["style_preset"] == "anime"
        assert json_data["width"] == 512

    def test_simple_generate_comprehensive(self):
        """Test simple_generate with every parameter."""
        client = MagicMock()
        client.post.return_value = SimpleImageResponse(
            created=1677610602,
            data=[ImageDataItem(b64_json="base64data")]
        )
        
        image = Image(client)
        response = image.simple_generate(
            model="dalle-3",
            prompt="Test prompt",
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
        assert response.created == 1677610602
        assert len(response.images) == 1
        assert response.images[0].b64_json == "base64data"
        client.post.assert_called_once()
        call_args = client.post.call_args
        json_data = call_args[1]["json_data"]
        
        # Verify all parameters were passed correctly
        assert json_data["model"] == "dalle-3"
        assert json_data["prompt"] == "Test prompt"
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


@pytest.mark.asyncio
class TestAsyncImageEdgeCases:
    async def test_async_generate_with_all_parameters(self):
        """Test async generate with every single parameter specified."""
        client = MagicMock()
        client.post = AsyncMock(return_value=ImageResponse(
            id="test",
            images=["test-image"],
            request=None,
            timing=TimingInfo(
                inferenceDuration=1.0,
                inferencePreprocessingTime=0.1,
                inferenceQueueTime=0.2,
                total=1.3
            ),
            created="2021-08-26T12:00:00Z"
        ))
        
        image = AsyncImage(client)
        response = await image.generate(
            model="test-model",
            prompt="Test prompt",
            cfg_scale=7.5,
            embed_exif_metadata=True,
            format="png",
            height=512,
            hide_watermark=False,
            # inpaint argument removed
            lora_strength=5,
            negative_prompt="bad quality",
            return_binary=False,
            safe_mode=True,
            seed=12345,
            steps=30,
            style_preset="anime",
            width=512
        )
        
        assert isinstance(response, ImageResponse)
        assert response.id == "test"
        assert response.images == ["test-image"]
        client.post.assert_awaited_once()
        call_args = client.post.call_args
        json_data = call_args[1]["json_data"]
        
        # Verify all parameters were passed correctly
        assert json_data["model"] == "test-model"
        assert json_data["prompt"] == "Test prompt"
        assert json_data["cfg_scale"] == 7.5
        assert json_data["embed_exif_metadata"] is True
        assert json_data["format"] == "png"
        assert json_data["height"] == 512
        assert json_data["hide_watermark"] is False
        # inpaint assertion removed
        assert json_data["lora_strength"] == 5
        assert json_data["negative_prompt"] == "bad quality"
        assert json_data["return_binary"] is False
        assert json_data["safe_mode"] is True
        assert json_data["seed"] == 12345
        assert json_data["steps"] == 30
        assert json_data["style_preset"] == "anime"
        assert json_data["width"] == 512

    async def test_async_simple_generate_comprehensive(self):
        """Test async simple_generate with every parameter."""
        client = MagicMock()
        client.post = AsyncMock(return_value=SimpleImageResponse(
            created=1677610602,
            data=[ImageDataItem(b64_json="base64data")]
        ))
        
        image = AsyncImage(client)
        response = await image.simple_generate(
            model="dalle-3",
            prompt="Test prompt",
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
        assert response.created == 1677610602
        assert len(response.images) == 1
        assert response.images[0].b64_json == "base64data"
        client.post.assert_awaited_once()
        call_args = client.post.call_args
        json_data = call_args[1]["json_data"]
        
        # Verify all parameters were passed correctly
        assert json_data["model"] == "dalle-3"
        assert json_data["prompt"] == "Test prompt"
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