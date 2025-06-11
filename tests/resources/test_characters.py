"""
Comprehensive unit tests for Characters API resources.

This module provides complete test coverage for the Characters API resources,
including both synchronous and asynchronous versions of character listing
functionality. Tests cover various scenarios including successful operations,
error handling, parameter validation, and edge cases.
"""

import pytest
import httpx
from unittest.mock import MagicMock, AsyncMock
from typing import Dict, Any

from venice_ai._client import VeniceClient
from venice_ai._async_client import AsyncVeniceClient
from venice_ai.resources.characters import Characters, AsyncCharacters
from venice_ai.types.characters import CharacterList, Character
from venice_ai.exceptions import APIError


class TestCharactersList:
    """Test cases for synchronous Characters.list method."""

    def test_list_success_basic(self):
        """Test successful character listing with basic parameters."""
        # Create mock response data
        mock_response_data = {
            "object": "list",
            "data": [
                {
                    "slug": "test-character-1",
                    "name": "Test Character 1",
                    "description": "A test character for unit testing",
                    "system_prompt": "You are a helpful test character",
                    "user_prompt": "Hello, I'm a test character",
                    "vision_enabled": True,
                    "image_url": "https://example.com/character1.png",
                    "voice_id": "voice_123",
                    "category_tags": ["test", "character"],
                    "created_at": "2025-01-01T00:00:00Z",
                    "updated_at": "2025-01-01T00:00:00Z"
                },
                {
                    "slug": "test-character-2",
                    "name": "Test Character 2",
                    "description": "Another test character",
                    "system_prompt": "You are another helpful test character",
                    "user_prompt": None,
                    "vision_enabled": False,
                    "image_url": None,
                    "voice_id": None,
                    "category_tags": ["test"],
                    "created_at": "2025-01-02T00:00:00Z",
                    "updated_at": "2025-01-02T00:00:00Z"
                }
            ]
        }
        
        # Create mock client
        mock_client = MagicMock(spec=VeniceClient)
        mock_client.get.return_value = mock_response_data
        
        # Create Characters resource
        characters_resource = Characters(mock_client)
        
        # Call list method
        result = characters_resource.list()
        
        # Verify result
        assert isinstance(result, CharacterList)
        assert result.object == "list"
        assert len(result.data) == 2
        assert isinstance(result.data[0], Character)
        assert result.data[0].slug == "test-character-1"
        assert result.data[0].name == "Test Character 1"
        assert result.data[0].vision_enabled is True
        assert result.data[1].slug == "test-character-2"
        assert result.data[1].vision_enabled is False
        
        # Verify the request was made correctly
        mock_client.get.assert_called_once_with(
            "characters",
            headers=None,
            params=None,
            timeout=None,
        )

    def test_list_with_extra_headers(self):
        """Test character listing with extra headers."""
        mock_response_data = {
            "object": "list",
            "data": []
        }
        
        # Create mock client
        mock_client = MagicMock(spec=VeniceClient)
        mock_client.get.return_value = mock_response_data
        characters_resource = Characters(mock_client)
        
        # Test with extra headers
        extra_headers = httpx.Headers({"X-Custom-Header": "test-value"})
        result = characters_resource.list(extra_headers=extra_headers)
        
        # Verify result
        assert isinstance(result, CharacterList)
        assert result.object == "list"
        assert len(result.data) == 0
        
        # Verify headers were passed correctly
        mock_client.get.assert_called_once_with(
            "characters",
            headers={"x-custom-header": "test-value"},
            params=None,
            timeout=None,
        )

    def test_list_with_extra_query(self):
        """Test character listing with extra query parameters."""
        mock_response_data = {
            "object": "list",
            "data": []
        }
        
        # Create mock client
        mock_client = MagicMock(spec=VeniceClient)
        mock_client.get.return_value = mock_response_data
        characters_resource = Characters(mock_client)
        
        # Test with extra query parameters
        extra_query = {"filter": "test", "limit": 10}
        result = characters_resource.list(extra_query=extra_query)
        
        # Verify result
        assert isinstance(result, CharacterList)
        
        # Verify query parameters were passed correctly
        mock_client.get.assert_called_once_with(
            "characters",
            headers=None,
            params={"filter": "test", "limit": 10},
            timeout=None,
        )

    def test_list_with_timeout(self):
        """Test character listing with custom timeout."""
        mock_response_data = {
            "object": "list",
            "data": []
        }
        
        # Create mock client
        mock_client = MagicMock(spec=VeniceClient)
        mock_client.get.return_value = mock_response_data
        characters_resource = Characters(mock_client)
        
        # Test with timeout
        timeout = 30.0
        result = characters_resource.list(timeout=timeout)
        
        # Verify result
        assert isinstance(result, CharacterList)
        
        # Verify timeout was passed correctly
        mock_client.get.assert_called_once_with(
            "characters",
            headers=None,
            params=None,
            timeout=30.0,
        )

    def test_list_with_all_parameters(self):
        """Test character listing with all optional parameters."""
        mock_response_data = {
            "object": "list",
            "data": [
                {
                    "slug": "comprehensive-test",
                    "name": "Comprehensive Test Character",
                    "description": "Testing all parameters",
                    "system_prompt": "System prompt",
                    "user_prompt": "User prompt",
                    "vision_enabled": True,
                    "image_url": "https://example.com/image.png",
                    "voice_id": "voice_456",
                    "category_tags": ["comprehensive", "test"],
                    "created_at": "2025-01-03T00:00:00Z",
                    "updated_at": "2025-01-03T00:00:00Z"
                }
            ]
        }
        
        # Create mock client
        mock_client = MagicMock(spec=VeniceClient)
        mock_client.get.return_value = mock_response_data
        characters_resource = Characters(mock_client)
        
        # Test with all parameters
        extra_headers = httpx.Headers({"Authorization": "Bearer test"})
        extra_query = {"category": "test"}
        extra_body = {"metadata": "test"}
        timeout = 45.0
        
        result = characters_resource.list(
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            timeout=timeout
        )
        
        # Verify result
        assert isinstance(result, CharacterList)
        assert len(result.data) == 1
        assert result.data[0].slug == "comprehensive-test"
        
        # Verify all parameters were passed correctly
        mock_client.get.assert_called_once_with(
            "characters",
            headers={"authorization": "Bearer test"},
            params={"category": "test"},
            timeout=45.0,
        )

    def test_list_empty_response(self):
        """Test character listing with empty response."""
        mock_response_data = {
            "object": "list",
            "data": []
        }
        
        # Create mock client
        mock_client = MagicMock(spec=VeniceClient)
        mock_client.get.return_value = mock_response_data
        characters_resource = Characters(mock_client)
        
        result = characters_resource.list()
        
        # Verify empty result
        assert isinstance(result, CharacterList)
        assert result.object == "list"
        assert len(result.data) == 0

    def test_list_api_error_handling(self):
        """Test error handling for character listing."""
        # Create mock client
        mock_client = MagicMock(spec=VeniceClient)
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 500
        mock_client.get.side_effect = APIError("API Error occurred", response=mock_response)
        
        characters_resource = Characters(mock_client)
        
        # Verify APIError is propagated
        with pytest.raises(APIError, match="API Error occurred"):
            characters_resource.list()

    def test_list_character_model_validation(self):
        """Test that character data is properly validated and converted to Character models."""
        mock_response_data = {
            "object": "list",
            "data": [
                {
                    "slug": "validation-test",
                    "name": "Validation Test Character",
                    "description": None,  # Test None values
                    "system_prompt": None,
                    "user_prompt": None,
                    "vision_enabled": False,
                    "image_url": None,
                    "voice_id": None,
                    "category_tags": None,
                    "created_at": None,
                    "updated_at": None
                }
            ]
        }
        
        # Create mock client
        mock_client = MagicMock(spec=VeniceClient)
        mock_client.get.return_value = mock_response_data
        characters_resource = Characters(mock_client)
        
        result = characters_resource.list()
        
        # Verify model validation
        assert isinstance(result, CharacterList)
        assert len(result.data) == 1
        character = result.data[0]
        assert isinstance(character, Character)
        assert character.slug == "validation-test"
        assert character.name == "Validation Test Character"
        assert character.description is None
        assert character.vision_enabled is False


class TestAsyncCharactersList:
    """Test cases for asynchronous AsyncCharacters.list method."""

    @pytest.mark.asyncio
    async def test_async_list_success_basic(self):
        """Test successful async character listing with basic parameters."""
        # Create mock response data
        mock_response_data = {
            "object": "list",
            "data": [
                {
                    "slug": "async-test-character-1",
                    "name": "Async Test Character 1",
                    "description": "An async test character",
                    "system_prompt": "You are an async test character",
                    "user_prompt": "Hello from async",
                    "vision_enabled": True,
                    "image_url": "https://example.com/async1.png",
                    "voice_id": "async_voice_123",
                    "category_tags": ["async", "test"],
                    "created_at": "2025-01-01T00:00:00Z",
                    "updated_at": "2025-01-01T00:00:00Z"
                }
            ]
        }
        
        # Create mock client
        mock_client = MagicMock(spec=AsyncVeniceClient)
        mock_client.get = AsyncMock(return_value=mock_response_data)
        
        # Create AsyncCharacters resource
        characters_resource = AsyncCharacters(mock_client)
        
        # Call list method
        result = await characters_resource.list()
        
        # Verify result
        assert isinstance(result, CharacterList)
        assert result.object == "list"
        assert len(result.data) == 1
        assert isinstance(result.data[0], Character)
        assert result.data[0].slug == "async-test-character-1"
        assert result.data[0].name == "Async Test Character 1"
        
        # Verify the request was made correctly
        mock_client.get.assert_called_once_with(
            "characters",
            headers=None,
            params=None,
            timeout=None,
        )

    @pytest.mark.asyncio
    async def test_async_list_with_extra_headers(self):
        """Test async character listing with extra headers."""
        mock_response_data = {
            "object": "list",
            "data": []
        }
        
        # Create mock client
        mock_client = MagicMock(spec=AsyncVeniceClient)
        mock_client.get = AsyncMock(return_value=mock_response_data)
        characters_resource = AsyncCharacters(mock_client)
        
        # Test with extra headers
        extra_headers = httpx.Headers({"X-Async-Header": "async-value"})
        result = await characters_resource.list(extra_headers=extra_headers)
        
        # Verify result
        assert isinstance(result, CharacterList)
        assert result.object == "list"
        assert len(result.data) == 0
        
        # Verify headers were passed correctly
        mock_client.get.assert_called_once_with(
            "characters",
            headers={"x-async-header": "async-value"},
            params=None,
            timeout=None,
        )

    @pytest.mark.asyncio
    async def test_async_list_with_extra_query(self):
        """Test async character listing with extra query parameters."""
        mock_response_data = {
            "object": "list",
            "data": []
        }
        
        # Create mock client
        mock_client = MagicMock(spec=AsyncVeniceClient)
        mock_client.get = AsyncMock(return_value=mock_response_data)
        characters_resource = AsyncCharacters(mock_client)
        
        # Test with extra query parameters
        extra_query = {"async_filter": "test", "async_limit": 5}
        result = await characters_resource.list(extra_query=extra_query)
        
        # Verify result
        assert isinstance(result, CharacterList)
        
        # Verify query parameters were passed correctly
        mock_client.get.assert_called_once_with(
            "characters",
            headers=None,
            params={"async_filter": "test", "async_limit": 5},
            timeout=None,
        )

    @pytest.mark.asyncio
    async def test_async_list_with_timeout(self):
        """Test async character listing with custom timeout."""
        mock_response_data = {
            "object": "list",
            "data": []
        }
        
        # Create mock client
        mock_client = MagicMock(spec=AsyncVeniceClient)
        mock_client.get = AsyncMock(return_value=mock_response_data)
        characters_resource = AsyncCharacters(mock_client)
        
        # Test with timeout
        timeout = 60.0
        result = await characters_resource.list(timeout=timeout)
        
        # Verify result
        assert isinstance(result, CharacterList)
        
        # Verify timeout was passed correctly
        mock_client.get.assert_called_once_with(
            "characters",
            headers=None,
            params=None,
            timeout=60.0,
        )

    @pytest.mark.asyncio
    async def test_async_list_with_all_parameters(self):
        """Test async character listing with all optional parameters."""
        mock_response_data = {
            "object": "list",
            "data": [
                {
                    "slug": "async-comprehensive-test",
                    "name": "Async Comprehensive Test Character",
                    "description": "Testing all async parameters",
                    "system_prompt": "Async system prompt",
                    "user_prompt": "Async user prompt",
                    "vision_enabled": False,
                    "image_url": "https://example.com/async-image.png",
                    "voice_id": "async_voice_789",
                    "category_tags": ["async", "comprehensive"],
                    "created_at": "2025-01-04T00:00:00Z",
                    "updated_at": "2025-01-04T00:00:00Z"
                }
            ]
        }
        
        # Create mock client
        mock_client = MagicMock(spec=AsyncVeniceClient)
        mock_client.get = AsyncMock(return_value=mock_response_data)
        characters_resource = AsyncCharacters(mock_client)
        
        # Test with all parameters
        extra_headers = httpx.Headers({"X-Async-Auth": "Bearer async-token"})
        extra_query = {"async_category": "comprehensive"}
        extra_body = {"async_metadata": "test"}
        timeout = 90.0
        
        result = await characters_resource.list(
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            timeout=timeout
        )
        
        # Verify result
        assert isinstance(result, CharacterList)
        assert len(result.data) == 1
        assert result.data[0].slug == "async-comprehensive-test"
        
        # Verify all parameters were passed correctly
        mock_client.get.assert_called_once_with(
            "characters",
            headers={"x-async-auth": "Bearer async-token"},
            params={"async_category": "comprehensive"},
            timeout=90.0,
        )

    @pytest.mark.asyncio
    async def test_async_list_empty_response(self):
        """Test async character listing with empty response."""
        mock_response_data = {
            "object": "list",
            "data": []
        }
        
        # Create mock client
        mock_client = MagicMock(spec=AsyncVeniceClient)
        mock_client.get = AsyncMock(return_value=mock_response_data)
        characters_resource = AsyncCharacters(mock_client)
        
        result = await characters_resource.list()
        
        # Verify empty result
        assert isinstance(result, CharacterList)
        assert result.object == "list"
        assert len(result.data) == 0

    @pytest.mark.asyncio
    async def test_async_list_api_error_handling(self):
        """Test error handling for async character listing."""
        # Create mock client
        mock_client = MagicMock(spec=AsyncVeniceClient)
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 500
        mock_client.get = AsyncMock(side_effect=APIError("Async API Error occurred", response=mock_response))
        
        characters_resource = AsyncCharacters(mock_client)
        
        # Verify APIError is propagated
        with pytest.raises(APIError, match="Async API Error occurred"):
            await characters_resource.list()

    @pytest.mark.asyncio
    async def test_async_list_character_model_validation(self):
        """Test that async character data is properly validated and converted to Character models."""
        mock_response_data = {
            "object": "list",
            "data": [
                {
                    "slug": "async-validation-test",
                    "name": "Async Validation Test Character",
                    "description": "Testing async validation",
                    "system_prompt": "Async validation prompt",
                    "user_prompt": "Async user validation",
                    "vision_enabled": True,
                    "image_url": "https://example.com/async-validation.png",
                    "voice_id": "async_validation_voice",
                    "category_tags": ["async", "validation"],
                    "created_at": "2025-01-05T00:00:00Z",
                    "updated_at": "2025-01-05T00:00:00Z"
                }
            ]
        }
        
        # Create mock client
        mock_client = MagicMock(spec=AsyncVeniceClient)
        mock_client.get = AsyncMock(return_value=mock_response_data)
        characters_resource = AsyncCharacters(mock_client)
        
        result = await characters_resource.list()
        
        # Verify model validation
        assert isinstance(result, CharacterList)
        assert len(result.data) == 1
        character = result.data[0]
        assert isinstance(character, Character)
        assert character.slug == "async-validation-test"
        assert character.name == "Async Validation Test Character"
        assert character.vision_enabled is True


class TestCharactersEdgeCases:
    """Test edge cases and boundary conditions for Characters resources."""

    def test_list_with_malformed_response(self):
        """Test handling of malformed API response."""
        # Mock response missing required fields
        mock_response_data = {
            "object": "list"
            # Missing "data" field
        }
        
        # Create mock client
        mock_client = MagicMock(spec=VeniceClient)
        mock_client.get.return_value = mock_response_data
        characters_resource = Characters(mock_client)
        
        # Should raise validation error due to missing required field
        with pytest.raises(Exception):  # Pydantic validation error
            characters_resource.list()

    def test_list_with_invalid_character_data(self):
        """Test handling of invalid character data in response."""
        mock_response_data = {
            "object": "list",
            "data": [
                {
                    # Missing required "slug" field
                    "name": "Invalid Character",
                    "description": "Missing slug field"
                }
            ]
        }
        
        # Create mock client
        mock_client = MagicMock(spec=VeniceClient)
        mock_client.get.return_value = mock_response_data
        characters_resource = Characters(mock_client)
        
        # Should raise validation error due to missing required field
        with pytest.raises(Exception):  # Pydantic validation error
            characters_resource.list()

    @pytest.mark.asyncio
    async def test_async_list_with_malformed_response(self):
        """Test async handling of malformed API response."""
        # Mock response with wrong object type
        mock_response_data = {
            "object": "invalid_type",
            "data": []
        }
        
        # Create mock client
        mock_client = MagicMock(spec=AsyncVeniceClient)
        mock_client.get = AsyncMock(return_value=mock_response_data)
        characters_resource = AsyncCharacters(mock_client)
        
        result = await characters_resource.list()
        
        # Should still work as Pydantic will accept the value
        assert isinstance(result, CharacterList)
        assert result.object == "invalid_type"

    def test_list_with_large_response(self):
        """Test handling of large character list response."""
        # Create a large number of characters
        large_character_list = []
        for i in range(100):
            large_character_list.append({
                "slug": f"character-{i}",
                "name": f"Character {i}",
                "description": f"Description for character {i}",
                "system_prompt": f"System prompt {i}",
                "user_prompt": f"User prompt {i}",
                "vision_enabled": i % 2 == 0,
                "image_url": f"https://example.com/character{i}.png",
                "voice_id": f"voice_{i}",
                "category_tags": [f"tag{i}", "test"],
                "created_at": "2025-01-01T00:00:00Z",
                "updated_at": "2025-01-01T00:00:00Z"
            })
        
        mock_response_data = {
            "object": "list",
            "data": large_character_list
        }
        
        # Create mock client
        mock_client = MagicMock(spec=VeniceClient)
        mock_client.get.return_value = mock_response_data
        characters_resource = Characters(mock_client)
        
        result = characters_resource.list()
        
        # Verify large response is handled correctly
        assert isinstance(result, CharacterList)
        assert len(result.data) == 100
        assert all(isinstance(char, Character) for char in result.data)
        assert result.data[0].slug == "character-0"
        assert result.data[99].slug == "character-99"

    def test_list_with_unicode_characters(self):
        """Test handling of unicode characters in response data."""
        mock_response_data = {
            "object": "list",
            "data": [
                {
                    "slug": "unicode-test",
                    "name": "测试角色 🤖",
                    "description": "Тестовый персонаж with émojis 🎭",
                    "system_prompt": "あなたは日本語を話すキャラクターです",
                    "user_prompt": "Здравствуйте! 你好! 🌍",
                    "vision_enabled": True,
                    "image_url": "https://example.com/unicode-character.png",
                    "voice_id": "unicode_voice",
                    "category_tags": ["unicode", "测试", "тест"],
                    "created_at": "2025-01-01T00:00:00Z",
                    "updated_at": "2025-01-01T00:00:00Z"
                }
            ]
        }
        
        # Create mock client
        mock_client = MagicMock(spec=VeniceClient)
        mock_client.get.return_value = mock_response_data
        characters_resource = Characters(mock_client)
        
        result = characters_resource.list()
        
        # Verify unicode handling
        assert isinstance(result, CharacterList)
        assert len(result.data) == 1
        character = result.data[0]
        assert character.name == "测试角色 🤖"
        assert character.description is not None and "émojis 🎭" in character.description
        assert "あなたは日本語を話すキャラクターです" == character.system_prompt
        assert character.category_tags is not None and "测试" in character.category_tags