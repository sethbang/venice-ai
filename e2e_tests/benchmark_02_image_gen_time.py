import pytest
import time
import asyncio
from venice_ai import VeniceClient
from venice_ai._async_client import AsyncVeniceClient
from venice_ai.exceptions import APIError
from venice_ai.types.image import ImageResponse

# Clients are provided via fixtures

# Test configurations
SIMPLE_PROMPT = "A simple landscape painting."
COMPLEX_PROMPT = (
    "A highly detailed digital artwork of a futuristic cityscape at night, "
    "featuring neon lights, flying cars, intricate architectural designs, "
    "and a diverse crowd of cybernetic humans and robots interacting in a bustling market. "
    "The scene should convey a sense of wonder and technological advancement."
)
MODEL = "venice-sd35"  # Updated to match the model used in other tests

def measure_time(func):
    """Decorator to measure execution time of a function."""
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        return result, end_time - start_time
    return wrapper

def measure_time_async(func):
    """Decorator to measure execution time of an async function."""
    async def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = await func(*args, **kwargs)
        end_time = time.perf_counter()
        return result, end_time - start_time
    return wrapper

# Synchronous Tests
def test_image_gen_time_simple_prompt(venice_client):
    """Test time taken for image generation with a simple prompt."""
    client = venice_client
    @measure_time
    def make_request():
        response = client.image.generate(
            prompt=SIMPLE_PROMPT,
            model=MODEL
        )
        return response
    
    response, duration = make_request()
    print(f"Time for image generation (simple prompt): {duration:.3f} seconds")
    assert response is not None, "Response should not be None"
    assert isinstance(response, ImageResponse), "Response should be an ImageResponse object"
    assert response.images, "Response should have image data"

def test_image_gen_time_complex_prompt(api_key):
    """Test time taken for image generation with a complex prompt."""
    client = VeniceClient(api_key=api_key, timeout=180.0)
    @measure_time
    def make_request():
        response = client.image.generate(
            prompt=COMPLEX_PROMPT,
            model=MODEL
        )
        return response
    
    response, duration = make_request()
    print(f"Time for image generation (complex prompt): {duration:.3f} seconds")
    assert response is not None, "Response should not be None"
    assert isinstance(response, ImageResponse), "Response should be an ImageResponse object"
    assert response.images, "Response should have image data"

# Asynchronous Tests
@pytest.mark.asyncio
async def test_image_gen_time_simple_prompt_async(async_venice_client):
    """Test time taken for image generation with a simple prompt using async client."""
    async_client = async_venice_client
    @measure_time_async
    async def make_request():
        response = await async_client.image.generate(
            prompt=SIMPLE_PROMPT,
            model=MODEL
        )
        return response
    
    response, duration = await make_request()
    print(f"Time for image generation (simple prompt, async): {duration:.3f} seconds")
    assert response is not None, "Response should not be None"
    assert isinstance(response, ImageResponse), "Response should be an ImageResponse object"
    assert response.images, "Response should have image data"

@pytest.mark.asyncio
async def test_image_gen_time_complex_prompt_async(async_venice_client):
    """Test time taken for image generation with a complex prompt using async client."""
    async_client = async_venice_client
    @measure_time_async
    async def make_request():
        response = await async_client.image.generate(
            prompt=COMPLEX_PROMPT,
            model=MODEL
        )
        return response
    
    response, duration = await make_request()
    print(f"Time for image generation (complex prompt, async): {duration:.3f} seconds")
    assert response is not None, "Response should not be None"
    assert isinstance(response, ImageResponse), "Response should be an ImageResponse object"
    assert response.images, "Response should have image data"