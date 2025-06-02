import pytest
import pytest_asyncio
from venice_ai import VeniceClient, AsyncVeniceClient

# Define a default embedding model for testing
DEFAULT_EMBEDDING_MODEL = "llama-3.2-3b"  # Using the specified embedding model

# Functional Tests for Embeddings API

@pytest.mark.xfail(reason="API key lacks embedding authorization")
def test_create_embeddings_single_input_sync(venice_client: VeniceClient):
    """Tests synchronous embedding creation for a single string."""
    input_text = "This is a sample text for embedding generation."
    response = venice_client.embeddings.create(
        model=DEFAULT_EMBEDDING_MODEL,
        input=input_text
    )
    
    # Validate response structure
    assert isinstance(response, dict)
    assert "data" in response
    assert isinstance(response["data"], list)
    assert len(response["data"]) == 1  # Single input should give a single embedding
    
    # Validate embedding data
    embedding = response["data"][0]
    assert "embedding" in embedding
    assert isinstance(embedding["embedding"], list)
    assert len(embedding["embedding"]) > 0  # Embedding vector should not be empty
    # Most embedding models output floating point vectors
    assert all(isinstance(value, float) for value in embedding["embedding"])
    
    # Check usage information
    assert "usage" in response
    assert "total_tokens" in response["usage"]
    assert isinstance(response["usage"]["total_tokens"], int)
    assert response["usage"]["total_tokens"] > 0

@pytest.mark.asyncio
@pytest.mark.xfail(reason="API key lacks embedding authorization")
async def test_create_embeddings_single_input_async(async_venice_client: AsyncVeniceClient):
    """Tests asynchronous embedding creation for a single string."""
    input_text = "This is a sample text for async embedding generation."
    response = await async_venice_client.embeddings.create(
        model=DEFAULT_EMBEDDING_MODEL,
        input=input_text
    )
    
    # Validate response structure
    assert isinstance(response, dict)
    assert "data" in response
    assert isinstance(response["data"], list)
    assert len(response["data"]) == 1
    
    # Validate embedding data
    embedding = response["data"][0]
    assert "embedding" in embedding
    assert isinstance(embedding["embedding"], list)
    assert len(embedding["embedding"]) > 0
    
    # Check usage information
    assert "usage" in response
    assert "total_tokens" in response["usage"]
    assert response["usage"]["total_tokens"] > 0

@pytest.mark.xfail(reason="API key lacks embedding authorization")
def test_create_embeddings_batch_input_sync(venice_client: VeniceClient):
    """Tests synchronous embedding creation for a batch of strings."""
    input_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Embedding models convert text into numerical vectors.",
        "These vectors capture semantic meaning and relationships."
    ]
    
    response = venice_client.embeddings.create(
        model=DEFAULT_EMBEDDING_MODEL,
        input=input_texts
    )
    
    # Validate response structure
    assert isinstance(response, dict)
    assert "data" in response
    assert isinstance(response["data"], list)
    assert len(response["data"]) == len(input_texts)  # Should have one embedding per input
    
    # Validate embedding data for each input
    for i, embedding_data in enumerate(response["data"]):
        assert "embedding" in embedding_data
        assert isinstance(embedding_data["embedding"], list)
        assert len(embedding_data["embedding"]) > 0
    
    # Compare embedding dimensions - all should be the same length
    embedding_lengths = [len(item["embedding"]) for item in response["data"]]
    assert len(set(embedding_lengths)) == 1  # All should be the same length
    
    # Check usage information
    assert "usage" in response
    assert "total_tokens" in response["usage"]
    assert response["usage"]["total_tokens"] > 0
    # Batch processing should use more tokens than a single embedding
    assert response["usage"]["total_tokens"] > 10

@pytest.mark.asyncio
@pytest.mark.xfail(reason="API key lacks embedding authorization")
async def test_create_embeddings_batch_input_async(async_venice_client: AsyncVeniceClient):
    """Tests asynchronous embedding creation for a batch of strings."""
    input_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Embedding models convert text into numerical vectors.",
        "These vectors capture semantic meaning and relationships."
    ]
    
    response = await async_venice_client.embeddings.create(
        model=DEFAULT_EMBEDDING_MODEL,
        input=input_texts
    )
    
    # Validate response structure
    assert isinstance(response, dict)
    assert "data" in response
    assert isinstance(response["data"], list)
    assert len(response["data"]) == len(input_texts)
    
    # Validate embedding data for each input
    for embedding_data in response["data"]:
        assert "embedding" in embedding_data
        assert isinstance(embedding_data["embedding"], list)
        assert len(embedding_data["embedding"]) > 0
    
    # Check usage information
    assert "usage" in response
    assert "total_tokens" in response["usage"]
    assert response["usage"]["total_tokens"] > 0

@pytest.mark.xfail(reason="API key lacks embedding authorization")
def test_create_embeddings_with_model_and_dimensions_sync(venice_client: VeniceClient):
    """Tests synchronous embedding creation with specified model and dimensions."""
    input_text = "This test checks for dimension configuration."
    
    # Some embedding models allow specifying the output dimensions
    # If this isn't supported by the model, this test might need to be skipped
    try:
        response = venice_client.embeddings.create(
            model=DEFAULT_EMBEDDING_MODEL,
            input=input_text,
            dimensions=512  # Request a specific embedding size if supported
        )
        
        # Validate that the embedding has the requested dimensions if supported
        assert isinstance(response, dict)
        assert "data" in response
        assert len(response["data"]) == 1
        embedding = response["data"][0]["embedding"]
        assert isinstance(embedding, list)
        
        # Check if dimensions parameter was honored (if supported)
        # If dimensions isn't supported or is ignored, this might fail
        if "dimensions" in response:
            assert response["dimensions"] == 512
            assert len(embedding) == 512
    except Exception as e:
        # If the model doesn't support dimensions parameter, skip the test
        # This test is marked xfail. If the dimensions parameter is not supported
        # and causes an exception, re-raising it will make the test fail,
        # which will then be caught by the xfail marker.
        raise e

def test_create_embeddings_error_invalid_input_sync(venice_client: VeniceClient):
    """Tests synchronous embedding creation with invalid input."""
    # Test with empty input which should cause an error
    with pytest.raises(Exception) as excinfo:
        venice_client.embeddings.create(
            model=DEFAULT_EMBEDDING_MODEL,
            input=""  # Empty input should raise an error
        )
    
    # Check that an appropriate error is raised
    assert excinfo.value is not None
    
    # Test with invalid input type
    with pytest.raises(Exception) as excinfo:
        venice_client.embeddings.create(
            model=DEFAULT_EMBEDDING_MODEL,
            input=123,  # Invalid input type (not string or list of strings) # type: ignore[arg-type]
        )
    
    assert excinfo.value is not None