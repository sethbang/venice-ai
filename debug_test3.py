"""Debug test file 3 - Contains Pylance errors related to messages parameter"""

import asyncio
from typing import Sequence
from venice_ai import AsyncVeniceClient
from venice_ai.types.chat import MessageParam


async def test_messages_type_error():
    """Test with fixed messages parameter - now properly typed"""
    client = AsyncVeniceClient(api_key="test-key")
    
    # Fixed: Properly typed messages that conform to MessageParam structure
    # This resolves the Pylance error by ensuring type compatibility
    messages: Sequence[MessageParam] = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello there"}
    ]
    
    # Fixed: Now uses properly typed messages that match Sequence[MessageParam]
    # This resolves the Pylance error about type mismatch
    response = await client.chat.completions.create(
        model="test-model",
        messages=messages,  # Fixed: Now properly typed as Sequence[MessageParam]
        stream=False
    )
    
    print(response)


if __name__ == "__main__":
    asyncio.run(test_messages_type_error())