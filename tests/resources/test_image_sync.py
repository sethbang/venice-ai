"""
Tests for the synchronous Image resource.
"""

import pytest
import httpx
import io
from typing import Dict, List, Literal, Optional, TypedDict

from venice_ai import VeniceClient
from venice_ai.types.image import (
    GenerateImageRequest, ImageResponse, SimpleGenerateImageRequest,
    SimpleImageResponse, UpscaleImageRequest, ImageStyleList, TimingInfo
)
from venice_ai.exceptions import APIError, AuthenticationError, VeniceError


# Define mock response structures for testing
class MockImageResponse(TypedDict):
    id: str
    images: List[str]
    request: Dict
    timing: Dict


class MockSimpleImageDataItem(TypedDict, total=False):
    b64_json: str
    url: str


class MockSimpleImageResponse(TypedDict):
    created: int
    data: List[MockSimpleImageDataItem]


class MockImageStyleList(TypedDict):
    data: List[str]
    object: Literal["list"]


def test_generate_success(httpx_mock):
    """Tests successful image generation."""
    mock_response_data: MockImageResponse = {
        "id": "img-12345",
        "images": ["base64_encoded_image_data"],
        "request": {
            "model": "test-model",
            "prompt": "A beautiful sunset"
        },
        "timing": {
            "inferenceDuration": 1.5,
            "inferencePreprocessingTime": 0.2,
            "inferenceQueueTime": 0.1,
            "total": 1.8
        }
    }

    httpx_mock.add_response(
        method="POST",
        url="https://api.venice.ai/api/v1/image/generate",
        json=mock_response_data,
        status_code=200,
    )

    client = VeniceClient(api_key="test-key")
    response = client.image.generate(
        model="test-model",
        prompt="A beautiful sunset"
    )

    assert isinstance(response, ImageResponse)
    assert response.id == "img-12345"
    assert len(response.images) == 1
    assert response.request is not None
    assert response.request["model"] == "test-model"
    assert response.request["prompt"] == "A beautiful sunset"


def test_generate_with_options(httpx_mock):
    """Tests image generation with additional options."""
    mock_response_data: MockImageResponse = {
        "id": "img-67890",
        "images": ["base64_encoded_image_data"],
        "request": {
            "model": "test-model",
            "prompt": "A snowy mountain",
            "negative_prompt": "clouds, fog",
            "width": 1024,
            "height": 1024,
            "steps": 50
        },
        "timing": {
            "inferenceDuration": 2.5,
            "inferencePreprocessingTime": 0.3,
            "inferenceQueueTime": 0.2,
            "total": 3.0
        }
    }

    httpx_mock.add_response(
        method="POST",
        url="https://api.venice.ai/api/v1/image/generate",
        json=mock_response_data,
        status_code=200,
    )

    client = VeniceClient(api_key="test-key")
    response = client.image.generate(
        model="test-model",
        prompt="A snowy mountain",
        negative_prompt="clouds, fog",
        width=1024,
        height=1024,
        steps=50
    )

    assert isinstance(response, ImageResponse)
    assert response.id == "img-67890"
    assert response.request is not None
    assert response.request["model"] == "test-model"
    assert response.request["prompt"] == "A snowy mountain"
    assert response.request["negative_prompt"] == "clouds, fog"
    assert response.request["width"] == 1024
    assert response.request["height"] == 1024
    assert response.request["steps"] == 50


def test_generate_return_binary(httpx_mock):
    """Tests image generation with return_binary=True."""
    mock_image_data = b"fake image bytes"

    httpx_mock.add_response(
        method="POST",
        url="https://api.venice.ai/api/v1/image/generate",
        content=mock_image_data,
        status_code=200,
        headers={"Content-Type": "image/png"},
    )

    client = VeniceClient(api_key="test-key")
    response = client.image.generate(
        model="test-model",
        prompt="A beautiful sunset",
        return_binary=True
    )

    assert isinstance(response, bytes)
    assert response == mock_image_data

def test_simple_generate_success(httpx_mock):
    """Tests successful simple image generation."""
    mock_response_data: MockSimpleImageResponse = {
        "created": 1683900000,
        "data": [
            {
                "b64_json": "base64_encoded_image_data"
            }
        ]
    }

    httpx_mock.add_response(
        method="POST",
        url="https://api.venice.ai/api/v1/images/generations",
        json=mock_response_data,
        status_code=200,
    )

    client = VeniceClient(api_key="test-key")
    response = client.image.simple_generate(
        model="dalle-3",
        prompt="A beautiful sunset"
    )

    assert isinstance(response, SimpleImageResponse)
    assert response.created == 1683900000
    assert len(response.images) == 1
    assert response.images[0].b64_json == "base64_encoded_image_data"


def test_simple_generate_with_options(httpx_mock):
    """Tests simple image generation with additional options."""
    mock_response_data: MockSimpleImageResponse = {
        "created": 1683900000,
        "data": [
            {
                "url": "https://api.venice.ai/images/generations/img-12345"
            },
            {
                "url": "https://api.venice.ai/images/generations/img-67890"
            }
        ]
    }

    httpx_mock.add_response(
        method="POST",
        url="https://api.venice.ai/api/v1/images/generations",
        json=mock_response_data,
        status_code=200,
    )

    client = VeniceClient(api_key="test-key")
    response = client.image.simple_generate(
        model="dalle-3",
        prompt="A beautiful mountain",
        n=2,
        size="1024x1024",
        response_format="url",
        quality="high"
    )

    assert isinstance(response, SimpleImageResponse)
    assert response.created == 1683900000
    assert len(response.images) == 2
    assert response.images[0].url == "https://api.venice.ai/images/generations/img-12345"
    assert response.images[1].url == "https://api.venice.ai/images/generations/img-67890"


def test_upscale_success(httpx_mock):
    """Tests successful image upscaling."""
    mock_response_data = {
        "id": "upscale-12345",
        "upscaled_image": "base64_encoded_upscaled_image",
        "original_width": 512,
        "original_height": 512,
        "upscaled_width": 1024,
        "upscaled_height": 1024
    }

    # Convert dict to JSON bytes to simulate raw_response=True behavior
    import json
    mock_response_bytes = json.dumps(mock_response_data).encode('utf-8')

    httpx_mock.add_response(
        method="POST",
        url="https://api.venice.ai/api/v1/image/upscale",
        content=mock_response_bytes,
        status_code=200,
        headers={"Content-Type": "application/json"}
    )

    client = VeniceClient(api_key="test-key")
    # Normally we'd pass a file path, bytes or file-like object
    # For testing, we'll use a simple bytes object
    test_image = b"test image data"
    response = client.image.upscale(
        image=test_image,
        scale=2.0
    )

    # Check that response is bytes
    assert isinstance(response, bytes)
    
    # Parse response bytes as JSON for validation
    parsed_response = json.loads(response.decode('utf-8'))
    assert parsed_response["id"] == "upscale-12345"
    assert "upscaled_image" in parsed_response
    assert parsed_response["original_width"] == 512
    assert parsed_response["upscaled_width"] == 1024


def test_upscale_with_filepath(httpx_mock, tmp_path):
    """Tests image upscaling with a file path input."""
    mock_response_data = {
        "id": "upscale-filepath-123",
        "upscaled_image": "base64_encoded_upscaled_image_filepath",
        "original_width": 512,
        "original_height": 512,
        "upscaled_width": 1024,
        "upscaled_height": 1024
    }

    # Convert dict to JSON bytes to simulate raw_response=True behavior
    import json
    mock_response_bytes = json.dumps(mock_response_data).encode('utf-8')

    httpx_mock.add_response(
        method="POST",
        url="https://api.venice.ai/api/v1/image/upscale",
        content=mock_response_bytes,
        status_code=200,
        headers={"Content-Type": "application/json"}
    )

    client = VeniceClient(api_key="test-key")
    # Create a dummy image file
    dummy_image_path = tmp_path / "test_image.png"
    dummy_image_path.write_bytes(b"dummy png data")

    response = client.image.upscale(
        image=str(dummy_image_path),
        scale=2.0
    )

    # Check that response is bytes
    assert isinstance(response, bytes)
    
    # Parse response bytes as JSON for validation
    parsed_response = json.loads(response.decode('utf-8'))
    assert parsed_response["id"] == "upscale-filepath-123"
    assert "upscaled_image" in parsed_response

def test_upscale_with_file_like_object(httpx_mock):
    """Tests image upscaling with a file-like object input."""
    mock_response_data = {
        "id": "upscale-filelike-456",
        "upscaled_image": "base64_encoded_upscaled_image_filelike",
        "original_width": 512,
        "original_height": 512,
        "upscaled_width": 1024,
        "upscaled_height": 1024
    }

    # Convert dict to JSON bytes to simulate raw_response=True behavior
    import json
    mock_response_bytes = json.dumps(mock_response_data).encode('utf-8')

    httpx_mock.add_response(
        method="POST",
        url="https://api.venice.ai/api/v1/image/upscale",
        content=mock_response_bytes,
        status_code=200,
        headers={"Content-Type": "application/json"}
    )

    client = VeniceClient(api_key="test-key")
    # Use a BytesIO object as a file-like object
    test_image_file = io.BytesIO(b"test file-like data")
    response = client.image.upscale(
        image=test_image_file,
        scale=2.0
    )

    # Check that response is bytes
    assert isinstance(response, bytes)
    
    # Parse response bytes as JSON for validation
    parsed_response = json.loads(response.decode('utf-8'))
    assert parsed_response["id"] == "upscale-filelike-456"
    assert "upscaled_image" in parsed_response

def test_upscale_file_not_found():
    """Tests image upscaling with a non-existent file path."""
    client = VeniceClient(api_key="test-key")
    with pytest.raises(VeniceError) as excinfo:
        client.image.upscale(image="non_existent_file.jpg", scale=2.0)

    assert "Image file not found" in str(excinfo.value)

def test_list_styles_success(httpx_mock):
    """Tests successful retrieval of available image styles."""
    mock_response_data: MockImageStyleList = {
        "data": ["3D Model", "Analog Film", "Anime", "Cinematic", "Comic Book"],
        "object": "list"
    }

    httpx_mock.add_response(
        method="GET",
        url="https://api.venice.ai/api/v1/image/styles",
        json=mock_response_data,
        status_code=200,
    )

    client = VeniceClient(api_key="test-key")
    response = client.image.list_styles()

    assert isinstance(response, dict)
    assert response["object"] == "list"
    assert isinstance(response["data"], list)
    assert len(response["data"]) == 5
    assert "Anime" in response["data"]


def test_list_styles_api_error(httpx_mock):
    """Tests API error handling for listing image styles."""
    httpx_mock.add_response(
        method="GET",
        url="https://api.venice.ai/api/v1/image/styles",
        status_code=500,
        json={"error": {"message": "Internal Server Error", "type": "api_error"}},
    )

    from venice_ai._client_with_retries import VeniceClientWithRetries
    client = VeniceClientWithRetries(api_key="test-key", max_retries=0)

    with pytest.raises(APIError) as excinfo:
        client.image.list_styles()

    assert excinfo.value.response is not None
    assert excinfo.value.response.status_code == 500
    assert "Internal Server Error" in str(excinfo.value)

def test_generate_api_error(httpx_mock):
    """Tests API error handling for image generation."""
    httpx_mock.add_response(
        method="POST",
        url="https://api.venice.ai/api/v1/image/generate",
        status_code=401,
        json={"error": {"message": "Invalid API key", "type": "authentication_error"}},
    )

    client = VeniceClient(api_key="invalid-key")

    with pytest.raises(AuthenticationError) as excinfo:
        client.image.generate(model="test-model", prompt="A beautiful sunset")

    assert excinfo.value.response is not None
    assert excinfo.value.response.status_code == 401
    assert "Invalid API key" in str(excinfo.value)

def test__guess_image_type():
    """Tests the internal _guess_image_type function."""
    from venice_ai.resources.image import _guess_image_type

    assert _guess_image_type("image.jpg") == "jpeg"
    assert _guess_image_type("image.jpeg") == "jpeg"
    assert _guess_image_type("image.png") == "png"
    assert _guess_image_type("image.webp") == "webp"
    assert _guess_image_type("image.gif") == "gif"
    assert _guess_image_type("image.bmp") == "octet-stream"  # Test fallback
    assert _guess_image_type("IMAGE.JPG") == "jpeg"  # Test case insensitivity


class TestImageResourceMissedLines:
    """Test cases for covering missed lines in synchronous Image resource."""

    def test_prepare_image_content_file_like_with_name(self, httpx_mock):
        """
        Test _prepare_image_content with a file-like object that has a 'name' attribute.
        Covers lines 81-82 in image.py.
        """
        from venice_ai.resources.image import Image # Import Image resource

        client = VeniceClient(api_key="test-key")
        image_resource = Image(client)

        mock_file_content = b"dummy image data"
        file_like_object = io.BytesIO(mock_file_content)
        file_like_object.name = "test_image.png" # Add name attribute

        content = image_resource._prepare_image_content(file_like_object)

        assert content == mock_file_content

    def test_prepare_image_content_unsupported_type(self):
        """
        Test _prepare_image_content with an unsupported image type.
        Covers line 99 in image.py.
        """
        from venice_ai.resources.image import Image # Import Image resource
        
        client = VeniceClient(api_key="test-key")
        image_resource = Image(client)

        with pytest.raises(VeniceError, match="Unsupported image type"):
            image_resource._prepare_image_content(12345) # type: ignore[arg-type] # Pass an integer (unsupported)

    def test_upscale_invalid_enhance_type(self, httpx_mock):
        """
        Test upscale method with an invalid type for the 'enhance' parameter.
        Covers line 375 in image.py.
        """
        client = VeniceClient(api_key="test-key")
        test_image_bytes = b"dummy image data"

        # This call should ideally raise a VeniceError or TypeError due to invalid 'enhance' type.
        # The actual behavior depends on how strictly the Pydantic model or internal checks validate.
        # We're testing the branch where it's not a bool and not "true"/"false".
        with pytest.raises(VeniceError) as excinfo:
            client.image.upscale(image=test_image_bytes, scale=2.0, enhance=123) # type: ignore[arg-type] # Invalid enhance type
        
        # Check if the error message indicates an issue with the 'enhance' parameter
        # Pydantic V2-like error message for boolean
        assert "Input should be a valid boolean" in str(excinfo.value)


    def test_upscale_invalid_upscale_factor_type(self, httpx_mock):
        """
        Test upscale method with an invalid type for the 'scale' parameter.
        Covers line 400 in image.py (related to scale validation).
        """
        client = VeniceClient(api_key="test-key")
        test_image_bytes = b"dummy image data"

        with pytest.raises(VeniceError) as excinfo:
            client.image.upscale(image=test_image_bytes, scale="not_a_float") # type: ignore[arg-type] # Invalid scale type
        
        # Pydantic V2-like error message for float/number
        assert "Input should be a valid number" in str(excinfo.value)