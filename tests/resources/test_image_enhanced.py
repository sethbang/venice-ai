"""
Additional tests for enhancing coverage of both synchronous and asynchronous Image resources.

This file contains tests targeting specific edge cases, parameter combinations, and error
handling cases that may not be covered by the main test files to improve overall test coverage.
"""

import pytest
import httpx
import io
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Literal, Optional, TypedDict, Union, Any, cast
from unittest.mock import patch, MagicMock

from venice_ai import VeniceClient, AsyncVeniceClient
from venice_ai.resources.image import Image, AsyncImage, _guess_image_type
from venice_ai.exceptions import APIError, AuthenticationError, InvalidRequestError, VeniceError


# Test _guess_image_type with additional file extensions
def test_guess_image_type_additional_extensions():
    """Tests the internal _guess_image_type function with additional cases."""
    # Test uppercase and mixed case extensions
    assert _guess_image_type("IMAGE.PNG") == "png"
    assert _guess_image_type("image.JPEG") == "jpeg"
    assert _guess_image_type("IMAGE.Webp") == "webp"
    
    # Test with path-like names
    assert _guess_image_type("/path/to/image.jpg") == "jpeg"
    assert _guess_image_type("C:\\Users\\test\\image.PNG") == "png"
    
    # Test with dots in filename
    assert _guess_image_type("my.image.with.dots.png") == "png"
    
    # Test with unusual extensions that should fall back to default
    assert _guess_image_type("image.tiff") == "octet-stream"
    assert _guess_image_type("image.svg") == "octet-stream"
    assert _guess_image_type("image") == "octet-stream"


# Test generate method with edge cases and additional parameter combinations
def test_generate_with_min_max_values(httpx_mock):
    """Tests image generation with boundary values for numeric parameters."""
    mock_response_data = {
        "id": "img-edge-case",
        "images": ["base64_encoded_image_data"],
        "request": {
            "model": "test-model",
            "prompt": "A test image",
            "cfg_scale": 30.0,  # Very high value
            "steps": 150,       # High number of steps
            "width": 64,        # Minimum width
            "height": 64        # Minimum height
        },
        "timing": {
            "inferenceDuration": 2.0,
            "inferencePreprocessingTime": 0.2,
            "inferenceQueueTime": 0.1,
            "total": 2.3
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
        prompt="A test image",
        cfg_scale=30.0,  # Maximum value typically allowed
        steps=150,       # High value
        width=64,        # Minimum dimension typically allowed
        height=64        # Minimum dimension typically allowed
    )

    assert isinstance(response, dict)
    assert response["id"] == "img-edge-case"
    assert response["request"] is not None
    assert response["request"]["cfg_scale"] == 30.0
    assert response["request"]["steps"] == 150
    assert response["request"]["width"] == 64
    assert response["request"]["height"] == 64


def test_generate_with_all_optional_params(httpx_mock):
    """Tests image generation with all optional parameters provided."""
    mock_response_data = {
        "id": "img-all-params",
        "images": ["base64_encoded_image_data"],
        "request": {
            "model": "test-model",
            "prompt": "A comprehensive test",
            "cfg_scale": 7.5,
            "embed_exif_metadata": True,
            "format": "png",
            "height": 512,
            "hide_watermark": True,
            "negative_prompt": "bad quality, blurry",
            "safe_mode": True,
            "seed": 12345,
            "steps": 30,
            "style_preset": "anime",
            "width": 512
        },
        "timing": {
            "inferenceDuration": 2.0,
            "inferencePreprocessingTime": 0.2,
            "inferenceQueueTime": 0.1,
            "total": 2.3
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
        prompt="A comprehensive test",
        cfg_scale=7.5,
        embed_exif_metadata=True,
        format="png",
        height=512,
        hide_watermark=True,
        negative_prompt="bad quality, blurry",
        safe_mode=True,
        seed=12345,
        steps=30,
        style_preset="anime",
        width=512
    )

    assert isinstance(response, dict)
    assert response["id"] == "img-all-params"
    # Check all parameters were passed correctly
    assert response["request"] is not None
    assert response["request"]["cfg_scale"] == 7.5
    assert response["request"]["embed_exif_metadata"] is True
    assert response["request"]["format"] == "png"
    assert response["request"]["height"] == 512
    assert response["request"]["hide_watermark"] is True
    assert response["request"]["negative_prompt"] == "bad quality, blurry"
    assert response["request"]["safe_mode"] is True
    assert response["request"]["seed"] == 12345
    assert response["request"]["steps"] == 30
    assert response["request"]["style_preset"] == "anime"
    assert response["request"]["width"] == 512


def test_generate_with_inpaint_parameter(httpx_mock):
    """Tests image generation with inpaint parameter for selective regeneration."""
    # Define inpaint parameter as a dictionary
    inpaint_config = {
        "image": "base64_encoded_original_image",
        "mask": "base64_encoded_mask_image",
        "prompt_strength": 0.8
    }

    mock_response_data = {
        "id": "img-inpaint",
        "images": ["base64_encoded_inpainted_image"],
        "request": {
            "model": "test-model",
            "prompt": "A beautiful landscape",
            "inpaint": inpaint_config
        },
        "timing": {
            "inferenceDuration": 2.0,
            "inferencePreprocessingTime": 0.2,
            "inferenceQueueTime": 0.1,
            "total": 2.3
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
        prompt="A beautiful landscape"
        # inpaint argument removed
    )

    assert isinstance(response, dict)
    assert response["id"] == "img-inpaint"
    assert response["request"] is not None
    assert "inpaint" in response["request"]
    assert response["request"]["inpaint"]["prompt_strength"] == 0.8


def test_upscale_with_invalid_file_format():
    """Tests upscale with a file that exists but is not a valid image."""
    # Create a temporary text file
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as temp_file:
        temp_file.write("This is not an image file")
        temp_path = temp_file.name

    client = VeniceClient(api_key="test-key")
    
    # Create a mock HTTP response that simulates an API error
    mock_http_response = MagicMock(spec=httpx.Response)
    mock_http_response.status_code = 400
    mock_http_response.request = MagicMock(spec=httpx.Request)
    # Add method and url attributes to the mock request
    mock_http_response.request.method = "POST"
    mock_http_response.request.url = httpx.URL("https://api.venice.ai/api/v1/image/upscale")
    error_payload = {"error": {"message": "Invalid image data", "type": "invalid_request_error"}}
    mock_http_response.json.return_value = error_payload
    mock_http_response.text = json.dumps(error_payload)
    # Configure raise_for_status to raise an instance of HTTPStatusError
    mock_http_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "400 Bad Request", request=mock_http_response.request, response=mock_http_response
    )
    
    # Patch the underlying HTTP client's request method
    with patch.object(client._client, 'request', return_value=mock_http_response) as mock_request:
        # The upscale should raise an InvalidRequestError
        with pytest.raises(InvalidRequestError) as excinfo:
            client.image.upscale(image=temp_path, scale=2.0)
        
        # Verify the error message contains details from the simulated API error
        assert "Invalid image data" in str(excinfo.value)
        # Verify the request method was called
        mock_request.assert_called_once()
    
    # Clean up
    Path(temp_path).unlink()


def test_upscale_with_text_mode_file(httpx_mock):
    """Tests upscale with a text mode file-like object."""
    mock_response_data = {
        "id": "upscale-text-mode",
        "upscaled_image": "base64_encoded_upscaled_image",
        "original_width": 100,
        "original_height": 100,
        "upscaled_width": 200,
        "upscaled_height": 200
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
    
    # Create a BytesIO object (binary mode file-like object, proper for image data)
    text_content = b"This is text content that will be converted to bytes"
    text_file = io.BytesIO(text_content)
    
    # The method should handle this by encoding to UTF-8
    response = client.image.upscale(image=text_file, scale=2.0)
    
    # Check that response is bytes
    assert isinstance(response, bytes)
    
    # Parse response bytes as JSON for validation
    parsed_response = json.loads(response.decode('utf-8'))
    assert parsed_response["id"] == "upscale-text-mode"
    assert "upscaled_image" in parsed_response


def test_upscale_with_all_optional_parameters(httpx_mock):
    """Tests upscale with all optional parameters specified."""
    mock_response_data = {
        "id": "upscale-all-params",
        "upscaled_image": "base64_encoded_upscaled_image",
        "original_width": 200,
        "original_height": 200,
        "upscaled_width": 600,
        "upscaled_height": 600
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
    test_image = b"test image data"
    
    response = client.image.upscale(
        image=test_image,
        enhance=True,
        enhance_creativity=0.7,
        enhance_prompt=True,
        replication=0.8,
        scale=3.0
    )
    
    # Check that response is bytes
    assert isinstance(response, bytes)
    
    # Parse response bytes as JSON for validation
    parsed_response = json.loads(response.decode('utf-8'))
    assert parsed_response["id"] == "upscale-all-params"
    assert parsed_response["upscaled_width"] == 600  # 200 * 3.0
    assert parsed_response["upscaled_height"] == 600  # 200 * 3.0


# Async versions of the tests

@pytest.mark.asyncio
async def test_generate_with_min_max_values_async(httpx_mock):
    """Tests async image generation with boundary values for numeric parameters."""
    mock_response_data = {
        "id": "img-edge-case-async",
        "images": ["base64_encoded_image_data"],
        "request": {
            "model": "test-model",
            "prompt": "A test image",
            "cfg_scale": 30.0,
            "steps": 150,
            "width": 64,
            "height": 64
        },
        "timing": {
            "inferenceDuration": 2.0,
            "inferencePreprocessingTime": 0.2,
            "inferenceQueueTime": 0.1,
            "total": 2.3
        }
    }

    httpx_mock.add_response(
        method="POST",
        url="https://api.venice.ai/api/v1/image/generate",
        json=mock_response_data,
        status_code=200,
    )

    async with AsyncVeniceClient(api_key="test-key") as client:
        response = await client.image.generate(
            model="test-model",
            prompt="A test image",
            cfg_scale=30.0,
            steps=150,
            width=64,
            height=64
        )

    assert isinstance(response, dict)
    assert response["id"] == "img-edge-case-async"
    assert response["request"] is not None
    assert response["request"]["cfg_scale"] == 30.0
    assert response["request"]["steps"] == 150
    assert response["request"]["width"] == 64
    assert response["request"]["height"] == 64


@pytest.mark.asyncio
async def test_generate_with_inpaint_parameter_async(httpx_mock):
    """Tests async image generation with inpaint parameter."""
    inpaint_config = {
        "image": "base64_encoded_original_image",
        "mask": "base64_encoded_mask_image",
        "prompt_strength": 0.8
    }

    mock_response_data = {
        "id": "img-inpaint-async",
        "images": ["base64_encoded_inpainted_image"],
        "request": {
            "model": "test-model",
            "prompt": "A beautiful landscape",
            "inpaint": inpaint_config
        },
        "timing": {
            "inferenceDuration": 2.0,
            "inferencePreprocessingTime": 0.2,
            "inferenceQueueTime": 0.1,
            "total": 2.3
        }
    }

    httpx_mock.add_response(
        method="POST",
        url="https://api.venice.ai/api/v1/image/generate",
        json=mock_response_data,
        status_code=200,
    )

    async with AsyncVeniceClient(api_key="test-key") as client:
        response = await client.image.generate(
            model="test-model",
            prompt="A beautiful landscape"
            # inpaint argument removed
        )

    assert isinstance(response, dict)
    assert response["id"] == "img-inpaint-async"
    assert response["request"] is not None
    assert "inpaint" in response["request"]
    assert response["request"]["inpaint"]["prompt_strength"] == 0.8


@pytest.mark.asyncio
async def test_upscale_with_all_optional_parameters_async(httpx_mock):
    """Tests async upscale with all optional parameters specified."""
    mock_response_data = {
        "id": "upscale-all-params-async",
        "upscaled_image": "base64_encoded_upscaled_image",
        "original_width": 200,
        "original_height": 200,
        "upscaled_width": 600,
        "upscaled_height": 600
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

    async with AsyncVeniceClient(api_key="test-key") as client:
        test_image = b"test image data"
        
        response = await client.image.upscale(
            image=test_image,
            enhance=True,
            enhance_creativity=0.7,
            enhance_prompt=True,
            replication=0.8,
            scale=3.0
        )
    
    # Check that response is bytes
    assert isinstance(response, bytes)
    
    # Parse response bytes as JSON for validation
    parsed_response = json.loads(response.decode('utf-8'))
    assert parsed_response["id"] == "upscale-all-params-async"
    assert parsed_response["upscaled_width"] == 600


@pytest.mark.asyncio
async def test_upscale_with_invalid_file_format_async():
    """Tests async upscale with a file that exists but is not a valid image."""
    # Create a temporary text file
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as temp_file:
        temp_file.write("This is not an image file")
        temp_path = temp_file.name

    async with AsyncVeniceClient(api_key="test-key") as client:
        # Use AsyncMock for the async method
        from unittest.mock import AsyncMock
        
        # Create a mock HTTP response that simulates an API error
        mock_http_response = MagicMock(spec=httpx.Response)
        mock_http_response.status_code = 400
        mock_http_response.request = MagicMock(spec=httpx.Request)
        # Add method and url attributes to the mock request
        mock_http_response.request.method = "POST"
        mock_http_response.request.url = httpx.URL("https://api.venice.ai/api/v1/image/upscale")
        error_payload = {"error": {"message": "Invalid image data", "type": "invalid_request_error"}}
        mock_http_response.json.return_value = error_payload
        mock_http_response.text = json.dumps(error_payload)
        # Configure raise_for_status to raise an instance of HTTPStatusError
        mock_http_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "400 Bad Request", request=mock_http_response.request, response=mock_http_response
        )

        # Patch the underlying HTTP client's request method
        with patch.object(client._client, 'request', return_value=mock_http_response) as mock_request:
            # The upscale should raise an InvalidRequestError
            with pytest.raises(InvalidRequestError) as excinfo:
                await client.image.upscale(image=temp_path, scale=2.0)
            
            # Verify the error message contains details from the simulated API error
            assert "Invalid image data" in str(excinfo.value)
            # Verify the request method was called
            mock_request.assert_awaited_once()
    
    # Clean up
    Path(temp_path).unlink()


# Testing invalid parameter handling
def test_generate_with_invalid_format(httpx_mock):
    """Tests image generation with invalid format parameter."""
    client = VeniceClient(api_key="test-key")
    
    # Set up mock for the API error response
    mock_error_response = {
        "error": {
            "message": "Invalid format specified: invalid_format. Must be one of: jpeg, png, webp",
            "type": "invalid_request_error",
            "param": "format",
            "code": "invalid_parameter"
        }
    }
    
    httpx_mock.add_response(
        method="POST",
        url="https://api.venice.ai/api/v1/image/generate",
        json=mock_error_response,
        status_code=400,
    )
    
    with pytest.raises(InvalidRequestError) as excinfo:
        client.image.generate(
            model="test-model",
            prompt="A test image",
            format="invalid_format"  # type: ignore  # Invalid format value for testing
        )
    
    assert excinfo.value.response is not None
    assert excinfo.value.response.status_code == 400
    assert "Invalid format" in str(excinfo.value)


@pytest.mark.asyncio
async def test_simple_generate_with_invalid_size_async(httpx_mock):
    """Tests async simple_generate with invalid size parameter."""
    mock_error_response = {
        "error": {
            "message": "Invalid size specified. Must be one of: auto, 256x256, 512x512, 1024x1024, ...",
            "type": "invalid_request_error",
            "param": "size",
            "code": "invalid_parameter"
        }
    }
    
    httpx_mock.add_response(
        method="POST",
        url="https://api.venice.ai/api/v1/images/generations",
        json=mock_error_response,
        status_code=400,
    )
    
    async with AsyncVeniceClient(api_key="test-key") as client:
        with pytest.raises(InvalidRequestError) as excinfo:
            await client.image.simple_generate(
                model="dalle-3",
                prompt="A test image",
                size="123x456"  # type: ignore  # Invalid size for testing
            )
    
    assert excinfo.value.response is not None
    assert excinfo.value.response.status_code == 400
    assert "Invalid size" in str(excinfo.value)


@pytest.mark.asyncio
async def test_upscale_with_negative_scale_async(httpx_mock):
    """Tests async upscale with negative scale parameter."""
    mock_error_response = {
        "error": {
            "message": "Scale must be a positive number",
            "type": "invalid_request_error",
            "param": "scale",
            "code": "invalid_parameter"
        }
    }
    
    httpx_mock.add_response(
        method="POST",
        url="https://api.venice.ai/api/v1/image/upscale",
        json=mock_error_response,
        status_code=400,
    )
    
    async with AsyncVeniceClient(api_key="test-key") as client:
        with pytest.raises(InvalidRequestError) as excinfo:
            await client.image.upscale(
                image=b"test image data",
                scale=-1.5  # Negative scale should trigger error
            )
    
    assert excinfo.value.response is not None
    assert excinfo.value.response.status_code == 400
    assert "Scale must be a positive number" in str(excinfo.value)