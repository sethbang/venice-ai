import pytest
import pytest_asyncio
from unittest.mock import patch, MagicMock, AsyncMock
import base64
import io
from pathlib import Path
import os

from venice_ai.resources.image import Image, AsyncImage, _guess_image_type
from venice_ai.exceptions import (
    InvalidRequestError,
    AuthenticationError,
    PermissionDeniedError,
    NotFoundError,
    RateLimitError,
    APIError,
    VeniceError # Added VeniceError
)

class TestImageSync:
    @pytest.fixture
    def image_resource(self, mocker):
        client_mock = mocker.Mock()
        return Image(client_mock)

    def test_generate_with_all_parameters(self, image_resource, mocker):
        """Test generate with all optional parameters."""
        mock_response = {
            "created": 1677610602,
            "data": [
                {
                    "url": "https://example.com/image1.png",
                    "b64_json": "base64image"
                }
            ]
        }
        mocker.patch.object(image_resource._client, 'post', return_value=mock_response)
        
        result = image_resource.generate(
            model="stable-diffusion-v2",
            prompt="A beautiful sunset",
            cfg_scale=7.5,
            embed_exif_metadata=True,
            format="png",
            height=768,
            hide_watermark=True,
            # inpaint argument removed
            lora_strength=10,
            negative_prompt="clouds, rain",
            return_binary=False,
            safe_mode=True,
            seed=123456,
            steps=50,
            style_preset="photographic",
            width=1024
        )
        
        assert result == mock_response
        image_resource._client.post.assert_called_once()
        call_args = image_resource._client.post.call_args
        assert call_args[0][0] == "image/generate"
        
        # Verify the json_data has all the parameters
        json_data = call_args[1]["json_data"]
        assert json_data["model"] == "stable-diffusion-v2"
        assert json_data["prompt"] == "A beautiful sunset"
        assert json_data["cfg_scale"] == 7.5
        assert json_data["embed_exif_metadata"] is True
        assert json_data["format"] == "png"
        assert json_data["height"] == 768
        assert json_data["hide_watermark"] is True
        # inpaint assertion removed
        assert json_data["lora_strength"] == 10
        assert json_data["negative_prompt"] == "clouds, rain"
        assert json_data["return_binary"] is False
        assert json_data["safe_mode"] is True
        assert json_data["seed"] == 123456
        assert json_data["steps"] == 50
        assert json_data["style_preset"] == "photographic"
        assert json_data["width"] == 1024

    def test_generate_with_binary_response(self, image_resource, mocker):
        """Test generate with binary response format."""
        mock_binary_data = b"binary_image_data"
        mocker.patch.object(image_resource._client, '_request', return_value=mock_binary_data)
        
        result = image_resource.generate(
            model="stable-diffusion-v2",
            prompt="A beautiful sunset",
            return_binary=True
        )
        
        assert result == mock_binary_data
        image_resource._client._request.assert_called_once()
        call_args = image_resource._client._request.call_args
        assert call_args[0][0] == "POST"
        assert call_args[0][1] == "image/generate"
        
        # Verify headers for binary response
        assert call_args[1].get("raw_response", False) is True

    def test_simple_generate_with_all_parameters(self, image_resource, mocker):
        """Test simple_generate with all optional parameters."""
        mock_response = {
            "created": 1677610602,
            "data": [
                {
                    "url": "https://example.com/image1.png",
                    "b64_json": "base64image"
                }
            ]
        }
        mocker.patch.object(image_resource._client, 'post', return_value=mock_response)
        
        result = image_resource.simple_generate(
            model="venice-diffusion",
            prompt="A beautiful sunset",
            background="transparent",
            moderation="low",
            n=2,
            output_compression=50,
            output_format="png",
            quality="high",
            response_format="b64_json",
            size="1024x1024",
            style="vivid",
            user="user123"
        )
        
        assert result == mock_response
        image_resource._client.post.assert_called_once()
        call_args = image_resource._client.post.call_args
        assert call_args[0][0] == "images/generations"
        
        # Verify the json_data has all the parameters
        json_data = call_args[1]["json_data"]
        assert json_data["model"] == "venice-diffusion"
        assert json_data["prompt"] == "A beautiful sunset"
        assert json_data["background"] == "transparent"
        assert json_data["moderation"] == "low"
        assert json_data["n"] == 2
        assert json_data["output_compression"] == 50
        assert json_data["output_format"] == "png"
        assert json_data["quality"] == "high"
        assert json_data["response_format"] == "b64_json"
        assert json_data["size"] == "1024x1024"
        assert json_data["style"] == "vivid"
        assert json_data["user"] == "user123"

    def test_upscale_with_file_path(self, image_resource, mocker, tmp_path):
        """Test upscale with file path."""
        # Create a temporary test image file
        test_image_path = tmp_path / "test_image.jpg"
        with open(test_image_path, 'wb') as f:
            f.write(b"fake image data")
        
        mock_response = b"upscaled_image_data"
        def custom_mock_request_sync(*args, **kwargs):
            # You can add print statements here to debug args/kwargs if needed
            # print(f"SYNC MOCK _request called with: args={args}, kwargs={kwargs}")
            return mock_response

        mocker.patch.object(image_resource._client, '_request', side_effect=custom_mock_request_sync)
        
        result = image_resource.upscale(
            image=str(test_image_path),
            enhance="true",
            enhance_creativity=0.8,
            enhance_prompt="Add more details",
            replication=0.7,
            scale=2.0
        )
        
        assert result == mock_response
        image_resource._client._request.assert_called_once()
        call_args = image_resource._client._request.call_args
        assert call_args[1]['method'] == "POST"
        assert call_args[1]['path'] == "image/upscale"
        
        # Verify the json_data has all the parameters
        json_data = call_args[1]["json_data"]
        assert "image" in json_data  # Base64 encoded image
        assert json_data["enhance"] is True
        assert json_data["replication"] == 0.7
        assert json_data["scale"] == 2.0

    def test_upscale_with_bytes(self, image_resource, mocker):
        """Test upscale with bytes."""
        image_bytes = b"fake image data"
        mock_response = b"upscaled_image_data"
        def custom_mock_request_sync(*args, **kwargs):
            # You can add print statements here to debug args/kwargs if needed
            # print(f"SYNC MOCK _request called with: args={args}, kwargs={kwargs}")
            return mock_response

        mocker.patch.object(image_resource._client, '_request', side_effect=custom_mock_request_sync)
        
        result = image_resource.upscale(image=image_bytes)
        
        assert result == mock_response
        image_resource._client._request.assert_called_once()
        call_args = image_resource._client._request.call_args
        assert call_args[1]['method'] == "POST"
        assert call_args[1]['path'] == "image/upscale"
        
        # Verify there is a base64 encoded image in the request
        json_data = call_args[1]["json_data"]
        assert "image" in json_data
        # The image should be base64 encoded
        decoded_bytes = base64.b64decode(json_data["image"])
        assert decoded_bytes == image_bytes

    def test_upscale_with_file_object(self, image_resource, mocker):
        """Test upscale with file-like object."""
        image_bytes = b"fake image data"
        file_obj = io.BytesIO(image_bytes)
        mock_response = b"upscaled_image_data"
        def custom_mock_request_sync(*args, **kwargs):
            # You can add print statements here to debug args/kwargs if needed
            # print(f"SYNC MOCK _request called with: args={args}, kwargs={kwargs}")
            return mock_response

        mocker.patch.object(image_resource._client, '_request', side_effect=custom_mock_request_sync)
        
        result = image_resource.upscale(image=file_obj)
        
        assert result == mock_response
        image_resource._client._request.assert_called_once()
        call_args = image_resource._client._request.call_args
        assert call_args[1]['method'] == "POST"
        assert call_args[1]['path'] == "image/upscale"
        
        # Verify there is a base64 encoded image in the request
        json_data = call_args[1]["json_data"]
        assert "image" in json_data

    def test_upscale_file_not_found(self, image_resource):
        """Test upscale with non-existent file path."""
        with pytest.raises(VeniceError, match="Image file not found"):
            image_resource.upscale(image="non_existent_file.jpg")

    def test_upscale_with_text_mode_file(self, image_resource, mocker, tmp_path):
        """Test upscale with text mode file."""
        # Create a temporary text file
        test_file_path = tmp_path / "test_file.txt"
        with open(test_file_path, 'w') as f:
            f.write("This is text data")
        
        # Create a mock file object that returns string instead of bytes
        file_obj = MagicMock()
        file_obj.read.return_value = "This is text data"
        
        # This test is designed to fail because _prepare_image_content expects bytes
        # from file-like objects. The implementation logic explicitly raises a VeniceError
        # if BytesIO.read() returns a string.

        with pytest.raises(VeniceError, match="Image source is a file-like object that did not return bytes from read()"):
            image_resource.upscale(image=file_obj)

    def test_list_styles(self, image_resource, mocker):
        """Test listing available image styles."""
        mock_response = {
            "data": [
                {"id": "photographic", "name": "Photographic", "description": "Realistic photographic style"},
                {"id": "digital-art", "name": "Digital Art", "description": "Digital art style"}
            ]
        }
        mocker.patch.object(image_resource._client, 'get', return_value=mock_response)
        
        result = image_resource.list_styles()
        
        assert result == mock_response
        image_resource._client.get.assert_called_once_with("image/styles")

    def test_guess_image_type(self):
        """Test _guess_image_type function."""
        assert _guess_image_type("image.jpg") == "jpeg"
        assert _guess_image_type("image.jpeg") == "jpeg"
        assert _guess_image_type("image.png") == "png"
        assert _guess_image_type("image.webp") == "webp"
        assert _guess_image_type("image.gif") == "gif"
        assert _guess_image_type("image.unknown") == "octet-stream"
        # Test case insensitivity
        assert _guess_image_type("IMAGE.PNG") == "png"


@pytest.mark.asyncio
class TestImageAsync:
    @pytest_asyncio.fixture
    async def image_resource(self, mocker):
        client_mock = mocker.Mock()
        client_mock.post = AsyncMock()
        client_mock.get = AsyncMock()
        return AsyncImage(client_mock)

    async def test_generate_with_all_parameters(self, image_resource):
        """Test async generate with all optional parameters."""
        mock_response = {
            "created": 1677610602,
            "data": [
                {
                    "url": "https://example.com/image1.png",
                    "b64_json": "base64image"
                }
            ]
        }
        image_resource._client.post.return_value = mock_response
        
        result = await image_resource.generate(
            model="stable-diffusion-v2",
            prompt="A beautiful sunset",
            cfg_scale=7.5,
            embed_exif_metadata=True,
            format="png",
            height=768,
            hide_watermark=True,
            # inpaint argument removed
            lora_strength=10,
            negative_prompt="clouds, rain",
            return_binary=False,
            safe_mode=True,
            seed=123456,
            steps=50,
            style_preset="photographic",
            width=1024
        )
        
        assert result == mock_response
        image_resource._client.post.assert_awaited_once()
        call_args = image_resource._client.post.call_args
        assert call_args[0][0] == "image/generate"
        
        # Verify the json_data has all the parameters
        json_data = call_args[1]["json_data"]
        assert json_data["model"] == "stable-diffusion-v2"
        assert json_data["prompt"] == "A beautiful sunset"
        assert json_data["cfg_scale"] == 7.5
        assert json_data["embed_exif_metadata"] is True
        assert json_data["format"] == "png"
        assert json_data["height"] == 768
        assert json_data["hide_watermark"] is True
        # inpaint assertion removed
        assert json_data["lora_strength"] == 10
        assert json_data["negative_prompt"] == "clouds, rain"
        assert json_data["return_binary"] is False
        assert json_data["safe_mode"] is True
        assert json_data["seed"] == 123456
        assert json_data["steps"] == 50
        assert json_data["style_preset"] == "photographic"
        assert json_data["width"] == 1024

    async def test_generate_with_binary_response(self, image_resource):
        """Test async generate with binary response format."""
        mock_binary_data = b"binary_image_data"
        
        # Need to mock the special _request method for raw binary
        image_resource._client._request = AsyncMock(return_value=mock_binary_data)
        
        result = await image_resource.generate(
            model="stable-diffusion-v2",
            prompt="A beautiful sunset",
            return_binary=True
        )
        
        assert result == mock_binary_data
        image_resource._client._request.assert_awaited_once()
        call_args = image_resource._client._request.call_args
        assert call_args[0][0] == "POST"
        assert call_args[0][1] == "image/generate"
        
        # Verify proper headers for binary response
        assert "headers" in call_args[1]
        assert call_args[1]["headers"] == {"Accept": "image/*"}

    async def test_simple_generate_with_all_parameters(self, image_resource):
        """Test async simple_generate with all optional parameters."""
        mock_response = {
            "created": 1677610602,
            "data": [
                {
                    "url": "https://example.com/image1.png",
                    "b64_json": "base64image"
                }
            ]
        }
        image_resource._client.post.return_value = mock_response
        
        result = await image_resource.simple_generate(
            model="venice-diffusion",
            prompt="A beautiful sunset",
            background="transparent",
            moderation="low",
            n=2,
            output_compression=50,
            output_format="png",
            quality="high",
            response_format="b64_json",
            size="1024x1024",
            style="vivid",
            user="user123"
        )
        
        assert result == mock_response
        image_resource._client.post.assert_awaited_once()
        call_args = image_resource._client.post.call_args
        assert call_args[0][0] == "images/generations"
        
        # Verify the json_data has all the parameters
        json_data = call_args[1]["json_data"]
        assert json_data["model"] == "venice-diffusion"
        assert json_data["prompt"] == "A beautiful sunset"
        assert json_data["background"] == "transparent"
        assert json_data["moderation"] == "low"
        assert json_data["n"] == 2
        assert json_data["output_compression"] == 50
        assert json_data["output_format"] == "png"
        assert json_data["quality"] == "high"
        assert json_data["response_format"] == "b64_json"
        assert json_data["size"] == "1024x1024"
        assert json_data["style"] == "vivid"
        assert json_data["user"] == "user123"

    async def test_upscale_with_file_path(self, image_resource, mocker, tmp_path):
        """Test async upscale with file path."""
        # Create a temporary test image file
        test_image_path = tmp_path / "test_image.jpg"
        with open(test_image_path, 'wb') as f:
            f.write(b"fake image data")
        
        expected_bytes = b"upscaled_image_data"

        async def custom_mock_request_async(*args, **kwargs):
            # print(f"ASYNC MOCK _request called with: args={args}, kwargs={kwargs}")
            return expected_bytes # This is the httpx.Response object (actually bytes in this test)

        # Use AsyncMock for _request method with the async side_effect
        mock_request = mocker.patch.object(
            image_resource._client,
            '_request',
            new_callable=AsyncMock,
            side_effect=custom_mock_request_async
        )
        
        result = await image_resource.upscale(
            image=str(test_image_path),
            enhance="true",
            enhance_creativity=0.8,
            enhance_prompt="Add more details",
            replication=0.7,
            scale=2.0
        )
        
        assert result == expected_bytes
        
        # Construct expected payload with "image" field containing base64-encoded data
        # We can't know the exact base64 string, so use mocker.ANY for that field
        expected_payload = {
            "image": mocker.ANY,  # Base64 encoded image data
            "enhance": True,  # API expects string "true", not boolean True
            "replication": 0.7,
            "scale": 2.0
        }
        
        # Use assert_awaited_once_with with all expected parameters
        mock_request.assert_awaited_once_with(
            method="POST",
            path="image/upscale",
            json_data=expected_payload,
            headers=mocker.ANY,
            raw_response=True,
            timeout=mocker.ANY
        )
        
        # Verify the base64 encoded image can be decoded
        actual_json_data = mock_request.call_args[1]["json_data"]
        assert "image" in actual_json_data
        decoded_bytes = base64.b64decode(actual_json_data["image"])
        assert len(decoded_bytes) > 0  # Ensure there's some decoded content

    async def test_upscale_with_bytes(self, image_resource, mocker):
        """Test async upscale with bytes."""
        image_bytes = b"fake image data"
        expected_bytes = b"upscaled_image_data"
        # Use mocker.patch.object with AsyncMock and correct return_value
        mock_request = mocker.patch.object(
            image_resource._client,
            '_request',
            new_callable=AsyncMock,
            return_value=expected_bytes
        )
        
        result = await image_resource.upscale(image=image_bytes)
        
        assert result == expected_bytes
        
        # Calculate the expected base64 encoding of our input bytes for verification
        expected_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Construct expected payload
        expected_payload = {
            "image": expected_base64,
            "scale": 2.0  # Default scale value
        }
        
        # Use assert_awaited_once_with with all expected parameters
        mock_request.assert_awaited_once_with(
            method="POST",
            path="image/upscale",
            json_data=expected_payload,
            headers=mocker.ANY,
            raw_response=True,
            timeout=mocker.ANY
        )

    async def test_upscale_with_file_object(self, image_resource, mocker):
        """Test async upscale with file-like object."""
        # Create a BytesIO object
        image_bytes = b"fake image data"
        file_obj = io.BytesIO(image_bytes)
        expected_bytes = b"upscaled_image_data"
        # Use mocker.patch.object with AsyncMock and correct return_value
        mock_request = mocker.patch.object(
            image_resource._client,
            '_request',
            new_callable=AsyncMock,
            return_value=expected_bytes
        )
        
        result = await image_resource.upscale(image=file_obj)
        
        assert result == expected_bytes
        
        # Calculate the expected base64 encoding of our input bytes for verification
        expected_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Construct expected payload
        expected_payload = {
            "image": expected_base64,
            "scale": 2.0  # Default scale value
        }
        
        # Use assert_awaited_once_with with all expected parameters
        mock_request.assert_awaited_once_with(
            method="POST",
            path="image/upscale",
            json_data=expected_payload,
            headers=mocker.ANY,
            raw_response=True,
            timeout=mocker.ANY
        )

    async def test_upscale_file_not_found(self, image_resource):
        """Test async upscale with non-existent file path."""
        with pytest.raises(VeniceError, match="Image file not found"):
            await image_resource.upscale(image="non_existent_file.jpg")

    async def test_list_styles(self, image_resource):
        """Test async listing available image styles."""
        mock_response = {
            "data": [
                {"id": "photographic", "name": "Photographic", "description": "Realistic photographic style"},
                {"id": "digital-art", "name": "Digital Art", "description": "Digital art style"}
            ]
        }
        image_resource._client.get.return_value = mock_response
        
        result = await image_resource.list_styles()
        
        assert result == mock_response
        image_resource._client.get.assert_awaited_once_with("image/styles")