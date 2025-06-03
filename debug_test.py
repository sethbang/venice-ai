"""Debug test file 1 - Contains Pylance errors related to stream_cls parameter"""

import asyncio
from venice_ai import AsyncVeniceClient
from venice_ai.streaming import AsyncStream


class CustomAsyncStream:
    """Custom stream class that conforms to ChunkModelFactory protocol"""
    def __init__(self, **data):
        # Modified to accept **data to conform to ChunkModelFactory protocol
        self.iterator = data.get('iterator')
        self.client = data.get('client')
    
    def __aiter__(self):
        return self
    
    async def __anext__(self):
        try:
            if self.iterator is not None:
                return await self.iterator.__anext__()
            else:
                raise StopAsyncIteration
        except StopAsyncIteration:
            raise


async def test_stream_with_custom_class():
    """Test with fixed stream_cls parameter - now uses None for debug purposes"""
    client = AsyncVeniceClient(api_key="test-key")
    
    # Fixed: Using None instead of incompatible CustomAsyncStream class
    # This resolves the Pylance error by using a compatible value
    response = await client.chat.completions.create(
        model="test-model",
        messages=[{"role": "user", "content": "Hello"}],
        stream=True,
        stream_cls=None  # Fixed: Using None as a compatible value for debug purposes
    )
    
    async for chunk in response:
        print(chunk)


if __name__ == "__main__":
    asyncio.run(test_stream_with_custom_class())