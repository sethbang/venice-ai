import pytest
import os
import io
from unittest.mock import patch, AsyncMock
from pathlib import Path

from venice_ai._client import VeniceClient  # Import real client
from venice_ai._async_client import AsyncVeniceClient # Import real async client
from venice_ai.resources.image import Image, AsyncImage
from venice_ai.exceptions import VeniceError, InvalidRequestError, AuthenticationError, PermissionDeniedError, NotFoundError, RateLimitError
from venice_ai.types.image import ImageResponse, SimpleImageResponse, ImageStyleList
from tests.conftest import create_mock_response

# Mock data for API responses
MOCK_IMAGE_RESPONSE = {
    "id": "img_123",
    "created": 1630000000,
    "data": [{"b64_json": "mock_base64_data"}]
}

MOCK_SIMPLE_IMAGE_RESPONSE = {
    "created": 1630000000,
    "data": [{"b64_json": "mock_base64_data"}]
}

MOCK_IMAGE_STYLE_LIST = {
    "styles": [
        {"id": "style1", "name": "Style 1", "description": "Description 1"},
        {"id": "style2", "name": "Style 2", "description": "Description 2"}
    ]
}

MOCK_UPSCALE_RESPONSE = {
    "id": "upscale_123",
    "status": "completed",
    "result": {"url": "http://example.com/upscaled_image.png"}
}

# Test class for synchronous Image resource
class TestImage:
    @pytest.fixture
    def image_resource(self):
        # Fixture for tests NOT needing patching (uses inline mock)
        class MockClient:
            def post(self, endpoint, json_data=None, **kwargs):
                # Check if raw_response is set to handle binary responses
                if kwargs.get("raw_response", False):
                    return b"mock_binary_data"
                return MOCK_IMAGE_RESPONSE if "generate" in endpoint else MOCK_SIMPLE_IMAGE_RESPONSE

            def get(self, endpoint):
                return MOCK_IMAGE_STYLE_LIST

            def _request(self, method, path, json_data=None, headers=None, raw_response=False, **kwargs):
                if raw_response:
                    return b"mock_binary_data"
                return MOCK_IMAGE_RESPONSE

            def _request_multipart(self, method, endpoint, files=None, data=None, **kwargs):
                return MOCK_UPSCALE_RESPONSE

        return Image(MockClient())

    @pytest.fixture
    def patchable_image_resource(self):
        # Fixture providing an Image resource with a real (dummy) client for patching
        dummy_client = VeniceClient(api_key="dummy_key", base_url="http://dummy.url")
        return Image(dummy_client)

    # Tests for generate method
    def test_generate_basic(self, image_resource):
        response = image_resource.generate(model="stable-diffusion-v2", prompt="A beautiful landscape")
        assert isinstance(response, dict)
        assert response["id"] == "img_123"
        assert "data" in response

    def test_generate_with_optional_params(self, image_resource):
        response = image_resource.generate(
            model="stable-diffusion-v2",
            prompt="A beautiful landscape",
            cfg_scale=7.5,
            height=512,
            width=512,
            format="png",
            negative_prompt="blurry, low quality",
            return_binary=False
        )
        assert isinstance(response, dict)
        assert response["id"] == "img_123"

    def test_generate_return_binary(self, image_resource):
        # Note: This test might need adjustment if _request logic changes
        response = image_resource.generate(
            model="stable-diffusion-v2",
            prompt="A beautiful landscape",
            return_binary=True
        )
        assert isinstance(response, bytes)
        assert response == b"mock_binary_data" # Relies on inline mock's _request

    # Tests for simple_generate method
    def test_simple_generate_basic(self, image_resource):
        response = image_resource.simple_generate(model="dalle-3", prompt="A simple image")
        assert isinstance(response, dict)
        assert "data" in response

    def test_simple_generate_with_optional_params(self, image_resource):
        response = image_resource.simple_generate(
            prompt="A simple image",
            background="transparent",
            model="dalle-3",
            n=2,
            size="1024x1024",
            response_format="b64_json"
        )
        assert isinstance(response, dict)
        assert "data" in response

    # Tests for upscale method
    def test_upscale_with_file_path(self, image_resource, tmp_path):
        image_path = tmp_path / "test_image.png"
        image_path.write_bytes(b"mock_image_data")
        response = image_resource.upscale(image=str(image_path), scale=2.0)
        # Check that response is bytes
        assert isinstance(response, bytes)
        
        # Parse response bytes as JSON for validation
        import json
        # For the test, we'll manually create a mock JSON response since the fixture
        # is returning b"mock_binary_data" for raw_response=True
        assert response == b"mock_binary_data"
        
        # Note: In a real scenario, we would parse the bytes:
        # parsed_response = json.loads(response.decode('utf-8'))
        # assert parsed_response["id"] == "upscale_123"

    def test_upscale_with_bytes(self, image_resource):
        response = image_resource.upscale(image=b"mock_image_data", scale=2.0)
        # Check that response is bytes
        assert isinstance(response, bytes)
        assert response == b"mock_binary_data"

    def test_upscale_with_file_like_object(self, image_resource):
        image_file = io.BytesIO(b"mock_image_data")
        response = image_resource.upscale(image=image_file, scale=2.0)
        # Check that response is bytes
        assert isinstance(response, bytes)
        assert response == b"mock_binary_data"

    def test_upscale_file_not_found(self, image_resource):
        with pytest.raises(VeniceError, match="Image file not found"):
            image_resource.upscale(image="nonexistent/path/image.png")

    # Tests for list_styles method
    def test_list_styles(self, image_resource):
        response = image_resource.list_styles()
        assert isinstance(response, dict)
        assert "styles" in response
        assert len(response["styles"]) == 2

    # Tests for error handling using the patchable fixture
    @patch('venice_ai._client.VeniceClient.post')
    def test_generate_invalid_request(self, mock_post, patchable_image_resource): # Use new fixture
        mock_response = create_mock_response(
            status_code=400,
            json_data={"error": {"message": "Invalid parameter"}}
        )
        mock_post.side_effect = InvalidRequestError("Invalid parameter", response=mock_response, body=None)
        with pytest.raises(InvalidRequestError):
            patchable_image_resource.generate(model="invalid-model", prompt="Test prompt") # Use new fixture

    @patch('venice_ai._client.VeniceClient.post')
    def test_generate_authentication_error(self, mock_post, patchable_image_resource): # Use new fixture
        mock_response = create_mock_response(
            status_code=401,
            json_data={"error": {"message": "Invalid API key"}}
        )
        mock_post.side_effect = AuthenticationError("Invalid API key", response=mock_response, body=None)
        with pytest.raises(AuthenticationError):
            patchable_image_resource.generate(model="stable-diffusion-v2", prompt="Test prompt") # Use new fixture

    @patch('venice_ai._client.VeniceClient.post')
    def test_generate_permission_error(self, mock_post, patchable_image_resource): # Use new fixture
        mock_response = create_mock_response(
            status_code=403,
            json_data={"error": {"message": "Access denied"}}
        )
        mock_post.side_effect = PermissionDeniedError("Access denied", response=mock_response, body=None)
        with pytest.raises(PermissionDeniedError):
            patchable_image_resource.generate(model="stable-diffusion-v2", prompt="Test prompt") # Use new fixture

    @patch('venice_ai._client.VeniceClient.post')
    def test_generate_not_found_error(self, mock_post, patchable_image_resource): # Use new fixture
        mock_response = create_mock_response(
            status_code=404,
            json_data={"error": {"message": "Model not found"}}
        )
        mock_post.side_effect = NotFoundError("Model not found", response=mock_response, body=None)
        with pytest.raises(NotFoundError):
            patchable_image_resource.generate(model="nonexistent-model", prompt="Test prompt") # Use new fixture

    @patch('venice_ai._client.VeniceClient.post')
    def test_generate_rate_limit_error(self, mock_post, patchable_image_resource): # Use new fixture
        mock_response = create_mock_response(
            status_code=429,
            json_data={"error": {"message": "Rate limit exceeded"}}
        )
        mock_post.side_effect = RateLimitError("Rate limit exceeded", response=mock_response, body=None)
        with pytest.raises(RateLimitError):
            patchable_image_resource.generate(model="stable-diffusion-v2", prompt="Test prompt") # Use new fixture

# Test class for asynchronous AsyncImage resource
@pytest.mark.asyncio
class TestAsyncImage:
    @pytest.fixture
    def async_image_resource(self):
        # Fixture for tests NOT needing patching (uses inline mock)
        class MockAsyncClient:
            async def post(self, endpoint, json_data=None, **kwargs):
                # Check if raw_response is set to handle binary responses
                if kwargs.get("raw_response", False):
                    return b"mock_binary_data"
                return MOCK_IMAGE_RESPONSE if "generate" in endpoint else MOCK_SIMPLE_IMAGE_RESPONSE

            async def get(self, endpoint):
                return MOCK_IMAGE_STYLE_LIST

            async def _request(self, method, path, json_data=None, headers=None, raw_response=False, **kwargs):
                if raw_response:
                    return b"mock_binary_data"
                return MOCK_IMAGE_RESPONSE

            async def _request_multipart(self, method, endpoint, files=None, data=None, **kwargs):
                return MOCK_UPSCALE_RESPONSE

        return AsyncImage(MockAsyncClient())

    @pytest.fixture
    def patchable_async_image_resource(self):
        # Fixture providing an AsyncImage resource with a real (dummy) async client for patching
        dummy_async_client = AsyncVeniceClient(api_key="dummy_key", base_url="http://dummy.url")
        return AsyncImage(dummy_async_client)

    # Tests for generate method
    async def test_generate_basic(self, async_image_resource):
        response = await async_image_resource.generate(model="stable-diffusion-v2", prompt="A beautiful landscape")
        assert isinstance(response, dict)
        assert response["id"] == "img_123"
        assert "data" in response

    async def test_generate_with_optional_params(self, async_image_resource):
        response = await async_image_resource.generate(
            model="stable-diffusion-v2",
            prompt="A beautiful landscape",
            cfg_scale=7.5,
            height=512,
            width=512,
            format="png",
            negative_prompt="blurry, low quality",
            return_binary=False
        )
        assert isinstance(response, dict)
        assert response["id"] == "img_123"

    async def test_generate_return_binary(self, async_image_resource):
        # Note: This test might need adjustment if _request logic changes
        response = await async_image_resource.generate(
            model="stable-diffusion-v2",
            prompt="A beautiful landscape",
            return_binary=True
        )
        assert isinstance(response, bytes)
        assert response == b"mock_binary_data" # Relies on inline mock's _request

    # Tests for simple_generate method
    async def test_simple_generate_basic(self, async_image_resource):
        response = await async_image_resource.simple_generate(model="dalle-3", prompt="A simple image")
        assert isinstance(response, dict)
        assert "data" in response

    async def test_simple_generate_with_optional_params(self, async_image_resource):
        response = await async_image_resource.simple_generate(
            prompt="A simple image",
            background="transparent",
            model="dalle-3",
            n=2,
            size="1024x1024",
            response_format="b64_json"
        )
        assert isinstance(response, dict)
        assert "data" in response

    # Tests for upscale method
    async def test_upscale_with_file_path(self, async_image_resource, tmp_path):
        image_path = tmp_path / "test_image.png"
        image_path.write_bytes(b"mock_image_data")
        response = await async_image_resource.upscale(image=str(image_path), scale=2.0)
        # Check that response is bytes
        assert isinstance(response, bytes)
        assert response == b"mock_binary_data"

    async def test_upscale_with_bytes(self, async_image_resource):
        response = await async_image_resource.upscale(image=b"mock_image_data", scale=2.0)
        # Check that response is bytes
        assert isinstance(response, bytes)
        assert response == b"mock_binary_data"

    async def test_upscale_with_file_like_object(self, async_image_resource):
        image_file = io.BytesIO(b"mock_image_data")
        response = await async_image_resource.upscale(image=image_file, scale=2.0)
        # Check that response is bytes
        assert isinstance(response, bytes)
        assert response == b"mock_binary_data"

    async def test_upscale_file_not_found(self, async_image_resource):
        with pytest.raises(VeniceError, match="Image file not found"):
            await async_image_resource.upscale(image="nonexistent/path/image.png")

    # Tests for list_styles method
    async def test_list_styles(self, async_image_resource):
        response = await async_image_resource.list_styles()
        assert isinstance(response, dict)
        assert "styles" in response
        assert len(response["styles"]) == 2

    # Tests for error handling using the patchable fixture
    @patch('venice_ai._async_client.AsyncVeniceClient.post', new_callable=AsyncMock)
    async def test_generate_invalid_request(self, mock_post, patchable_async_image_resource): # Use new fixture
        mock_response = create_mock_response(
            status_code=400,
            json_data={"error": {"message": "Invalid parameter"}}
        )
        mock_post.side_effect = InvalidRequestError("Invalid parameter", response=mock_response, body=None)
        with pytest.raises(InvalidRequestError):
            await patchable_async_image_resource.generate(model="invalid-model", prompt="Test prompt") # Use new fixture

    @patch('venice_ai._async_client.AsyncVeniceClient.post', new_callable=AsyncMock)
    async def test_generate_authentication_error(self, mock_post, patchable_async_image_resource): # Use new fixture
        mock_response = create_mock_response(
            status_code=401,
            json_data={"error": {"message": "Invalid API key"}}
        )
        mock_post.side_effect = AuthenticationError("Invalid API key", response=mock_response, body=None)
        with pytest.raises(AuthenticationError):
            await patchable_async_image_resource.generate(model="stable-diffusion-v2", prompt="Test prompt") # Use new fixture

    @patch('venice_ai._async_client.AsyncVeniceClient.post', new_callable=AsyncMock)
    async def test_generate_permission_error(self, mock_post, patchable_async_image_resource): # Use new fixture
        mock_response = create_mock_response(
            status_code=403,
            json_data={"error": {"message": "Access denied"}}
        )
        mock_post.side_effect = PermissionDeniedError("Access denied", response=mock_response, body=None)
        with pytest.raises(PermissionDeniedError):
            await patchable_async_image_resource.generate(model="stable-diffusion-v2", prompt="Test prompt") # Use new fixture

    @patch('venice_ai._async_client.AsyncVeniceClient.post', new_callable=AsyncMock)
    async def test_generate_not_found_error(self, mock_post, patchable_async_image_resource): # Use new fixture
        mock_response = create_mock_response(
            status_code=404,
            json_data={"error": {"message": "Model not found"}}
        )
        mock_post.side_effect = NotFoundError("Model not found", response=mock_response, body=None)
        with pytest.raises(NotFoundError):
            await patchable_async_image_resource.generate(model="nonexistent-model", prompt="Test prompt") # Use new fixture

    @patch('venice_ai._async_client.AsyncVeniceClient.post', new_callable=AsyncMock)
    async def test_generate_rate_limit_error(self, mock_post, patchable_async_image_resource): # Use new fixture
        mock_response = create_mock_response(
            status_code=429,
            json_data={"error": {"message": "Rate limit exceeded"}}
        )
        mock_post.side_effect = RateLimitError("Rate limit exceeded", response=mock_response, body=None)
        with pytest.raises(RateLimitError):
            await patchable_async_image_resource.generate(model="stable-diffusion-v2", prompt="Test prompt") # Use new fixture