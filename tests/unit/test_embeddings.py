import pytest
from unittest.mock import MagicMock, AsyncMock

from venice_ai.resources.embeddings import Embeddings, AsyncEmbeddings
from venice_ai._client import VeniceClient
from venice_ai._async_client import AsyncVeniceClient

class TestEmbeddings:
    @pytest.fixture
    def mock_client(self):
        client = MagicMock(spec=VeniceClient)
        client.post = MagicMock() # Mock the post method
        return client

    def test_create_with_required_params(self, mock_client):
        """Test create method with only required parameters."""
        embeddings_resource = Embeddings(mock_client)
        model = "test-model"
        input_data = "test input"

        embeddings_resource.create(model=model, input=input_data)

        mock_client.post.assert_called_once_with(
            "embeddings",
            json_data={
                "model": model,
                "input": input_data,
            }
        )

    def test_create_with_all_optional_params(self, mock_client):
        """Test create method with all optional parameters."""
        embeddings_resource = Embeddings(mock_client)
        model = "test-model"
        input_data = ["test input 1", "test input 2"]
        dimensions = 10
        encoding_format = "base64"
        user = "test-user"

        embeddings_resource.create(
            model=model,
            input=input_data,
            dimensions=dimensions,
            encoding_format=encoding_format,
            user=user
        )

        mock_client.post.assert_called_once_with(
            "embeddings",
            json_data={
                "model": model,
                "input": input_data,
                "dimensions": dimensions,
                "encoding_format": encoding_format,
                "user": user,
            }
        )

    def test_create_with_some_optional_params(self, mock_client):
        """Test create method with some optional parameters."""
        embeddings_resource = Embeddings(mock_client)
        model = "test-model"
        input_data = [1, 2, 3]
        user = "another-user"

        embeddings_resource.create(
            model=model,
            input=input_data,
            user=user
        )

        mock_client.post.assert_called_once_with(
            "embeddings",
            json_data={
                "model": model,
                "input": input_data,
                "user": user,
            }
        )

class TestAsyncEmbeddings:
    @pytest.fixture
    def mock_async_client(self):
        client = MagicMock(spec=AsyncVeniceClient)
        client.post = AsyncMock() # Mock the async post method
        return client

    @pytest.mark.asyncio
    async def test_create_with_required_params(self, mock_async_client):
        """Test async create method with only required parameters."""
        embeddings_resource = AsyncEmbeddings(mock_async_client)
        model = "test-model-async"
        input_data = "test input async"

        await embeddings_resource.create(model=model, input=input_data)

        mock_async_client.post.assert_awaited_once_with(
            "embeddings",
            json_data={
                "model": model,
                "input": input_data,
            }
        )

    @pytest.mark.asyncio
    async def test_create_with_all_optional_params(self, mock_async_client):
        """Test async create method with all optional parameters."""
        embeddings_resource = AsyncEmbeddings(mock_async_client)
        model = "test-model-async"
        input_data = ["test input 1 async", "test input 2 async"]
        dimensions = 20
        encoding_format = "float"
        user = "test-user-async"

        await embeddings_resource.create(
            model=model,
            input=input_data,
            dimensions=dimensions,
            encoding_format=encoding_format,
            user=user
        )

        mock_async_client.post.assert_awaited_once_with(
            "embeddings",
            json_data={
                "model": model,
                "input": input_data,
                "dimensions": dimensions,
                "encoding_format": encoding_format,
                "user": user,
            }
        )

    @pytest.mark.asyncio
    async def test_create_with_some_optional_params(self, mock_async_client):
        """Test async create method with some optional parameters."""
        embeddings_resource = AsyncEmbeddings(mock_async_client)
        model = "test-model-async"
        input_data = [[4, 5], [6]]
        dimensions = 30

        await embeddings_resource.create(
            model=model,
            input=input_data,
            dimensions=dimensions
        )

        mock_async_client.post.assert_awaited_once_with(
            "embeddings",
            json_data={
                "model": model,
                "input": input_data,
                "dimensions": dimensions,
            }
        )