# Venice AI Python Client

[![PyPI version](https://img.shields.io/pypi/v/venice-ai.svg)](https://pypi.org/project/venice-ai/)
[![CI Status](https://github.com/sethbang/venice-ai/actions/workflows/python-publish.yaml/badge.svg)](https://github.com/sethbang/venice-ai/actions/workflows/python-publish.yaml)
[![Coverage Status](https://img.shields.io/codecov/c/github/sethbang/venice-ai.svg)](https://codecov.io/gh/sethbang/venice-ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Versions](https://img.shields.io/pypi/pyversions/venice-ai.svg)](https://pypi.org/project/venice-ai/)

Developed to benchmark and explore the full capabilities of the Venice.ai API, the venice-ai Python package has evolved into a comprehensive client library for developers. This library provides convenient access to Venice.ai's powerful features, including chat completions, image generation, audio synthesis, embeddings, model management, API key management, billing information, and more, with support for both synchronous and asynchronous operations.

## Powered by

<div align="center">
  <a href="https://venice.ai/chat?ref=6sxLV1">
    <img src="./venice-logo-lockup-red.svg" alt="Venice.ai" width="250">
  </a>
  <sub><em>*This is a referral link</em></sub>
</div>

## Table of Contents

- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [API Key Setup](#api-key-setup)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Client Initialization](#client-initialization)
  - [Chat Completions](#chat-completions)
  - [Image Generation](#image-generation)
  - [Audio Synthesis (Text-to-Speech)](#audio-synthesis-text-to-speech)
  - [Embeddings Creation](#embeddings-creation)
  - [Model Management](#model-management)
  - [API Key Management](#api-key-management)
  - [Billing Information](#billing-information)
- [Error Handling](#error-handling)
- [Advanced Usage](#advanced-usage)
  - [Using a Custom `httpx` Client](#using-a-custom-httpx-client)
  - [Understanding Streaming](#understanding-streaming)
  - [Token Estimation](#token-estimation)
- [Showcase Application](#showcase-application)
- [Testing](#testing)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## Features

- Intuitive Pythonic interface for all Venice.ai API endpoints.
- Support for both synchronous and asynchronous operations.
- Access to all major Venice AI model families including text generation, image synthesis, audio processing, and embedding models.
- Streaming capabilities for chat completions and audio.
- Built-in utilities for tasks like token estimation and chat message validation.
- Robust error handling with a custom exception hierarchy.
- Type-hinted for a better developer experience and static analysis.
- Resource-oriented client design (e.g., `client.chat`, `client.image`).
- Comprehensive testing suite with `test_runner.py` for easy execution.
- Detailed API documentation generated with Sphinx.

## Getting Started

### Prerequisites

Python 3.11 or higher.

### Installation

You can install the Venice AI client library from PyPI:

```bash
pip install venice-ai
```

Alternatively, to install the latest development version from source (recommended if you want to contribute or need the absolute latest changes not yet released on PyPI):

```bash
git clone https://github.com/sethbang/venice-ai.git # Or your fork
cd venice-ai-python
poetry install
```

Note: `poetry install` installs main dependencies. For development or running all tests, install with dev dependencies: `poetry install --with dev`.

### API Key Setup

To use the Venice AI API, you need an API key. You can obtain your API key from your Venice AI dashboard.

The client library expects the API key to be available as an environment variable (recommended):

```bash
export VENICE_API_KEY="your_api_key_here"
```

Alternatively, you can pass the API key directly when initializing the client:
`client = VeniceClient(api_key="your_api_key_here")`

## Quick Start

Get up and running in seconds!

**Synchronous Chat Completion:**

```python
from venice_ai import VeniceClient

# Assumes VENICE_API_KEY is set in your environment
# Recommended: use as a context manager
with VeniceClient() as client:
    try:
        response = client.chat.completions.create(
            model="llama-3.2-3b", # Or your preferred model
            messages=[{"role": "user", "content": "Hello, Venice AI!"}]
        )
        print(response.choices[0].message.content)
    except Exception as e:
        print(f"An error occurred: {e}")
```

**Asynchronous Chat Completion:**

```python
import asyncio
from venice_ai import AsyncVeniceClient

async def main():
    # Assumes VENICE_API_KEY is set in your environment
    # Recommended: use as an async context manager
    async with AsyncVeniceClient() as async_client:
        try:
            response = await async_client.chat.completions.create(
                model="llama-3.2-3b", # Or your preferred model
                messages=[{"role": "user", "content": "Hello asynchronously, Venice AI!"}]
            )
            print(response.choices[0].message.content)
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Usage

### Client Initialization

**Synchronous Client:**

```python
from venice_ai import VeniceClient

# Option 1: API key from environment variable VENICE_API_KEY
client = VeniceClient()

# Option 2: Pass API key directly
# client = VeniceClient(api_key="your_api_key_here")

# Using as a context manager (recommended for proper resource cleanup):
with VeniceClient(api_key="your_api_key_here") as client:
    # Use the client for API calls
    models_list = client.models.list()
    print(f"Found {len(models_list.data)} models.")

# If not using a context manager, remember to close the client:
# client.close()
```

**Asynchronous Client:**

```python
import asyncio
from venice_ai import AsyncVeniceClient

async def main():
    # Option 1: API key from environment variable VENICE_API_KEY
    async_client = AsyncVeniceClient()

    # Option 2: Pass API key directly
    # async_client = AsyncVeniceClient(api_key="your_api_key_here")

    # Using as an async context manager (recommended):
    async with AsyncVeniceClient(api_key="your_api_key_here") as async_client:
        # Use the client for API calls
        models_list = await async_client.models.list()
        print(f"Found {len(models_list.data)} models.")

    # If not using an async context manager, remember to close the client:
    # await async_client.close()

if __name__ == "__main__":
    asyncio.run(main())
```

It's important to `close()` (for `VeniceClient`) or `await async_client.close()` (for `AsyncVeniceClient`) when you're finished, if not using context managers. This ensures that underlying HTTP resources and connections are properly released.

### Chat Completions

The response objects (`response`, `chunk`) are `TypedDict`s. You can explore their structure for more details (see `src/venice_ai/types/chat.py` or the Sphinx-generated API documentation).

**Non-streaming example:**

```python
from venice_ai import VeniceClient

with VeniceClient() as client:
    response = client.chat.completions.create(
        model="llama-3.2-3b", # Or your preferred model
        messages=[
            {"role": "user", "content": "Hello, how are you?"}
        ]
    )
    print(response.choices[0].message.content)
```

**Streaming example:**

```python
from venice_ai import VeniceClient

with VeniceClient() as client:
    stream = client.chat.completions.create(
        model="llama-3.2-3b",
        messages=[
            {"role": "user", "content": "Tell me a short story."}
        ],
        stream=True
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="")
    print()
```

**Tool Calling Example:**
(Ensure the selected model supports tool calling)

```python
from venice_ai import VeniceClient

with VeniceClient() as client:
    response = client.chat.completions.create(
        model="llama-3.2-3b", # Choose a model that supports tool calls
        messages=[{"role": "user", "content": "What's the weather in London?"}],
        tools=[{
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "The city and state, e.g., San Francisco, CA"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                    },
                    "required": ["location"]
                }
            }
        }],
        tool_choice="auto" # Can be "auto", "none", or a specific tool
    )
    if response.choices[0].message.tool_calls:
        tool_call = response.choices[0].message.tool_calls[0]
        print(f"Tool call requested: {tool_call.function.name}")
        print(f"Arguments: {tool_call.function.arguments}")
    else:
        print(response.choices[0].message.content)
```

### Image Generation

**Basic Generation:**

```python
from venice_ai import VeniceClient
import base64
from PIL import Image
import io

with VeniceClient() as client:
    response = client.image.generate(
        model="venice-sd35", # Or your preferred image model
        prompt="A futuristic cityscape at sunset",
        width=1024,
        height=1024,
        steps=25, # Example of another parameter
        # negative_prompt="blurry, low quality", # Example
        # style_preset="cinematic" # Example
    )
    if response.images:
        img_b64 = response.images[0]
        img_bytes = base64.b64decode(img_b64)
        # pil_image = Image.open(io.BytesIO(img_bytes))
        # pil_image.show()
        # pil_image.save("generated_image.png")
        print("Image generated successfully (first image data received).")
```

**Simple Generation (OpenAI-compatible):**

```python
# with VeniceClient() as client:
#     response = client.image.simple_generate(
#         model="venice-sd35",
#         prompt="A cute cat wearing a small hat",
#         size="512x512"
#     )
#     # Process response.data[0].b64_json or response.data[0].url
```

**Image Upscaling:**

```python
# with VeniceClient() as client:
#     with open("path/to/your/image.png", "rb") as img_file:
#         upscaled_image_bytes = client.image.upscale(
#             image=img_file, # Can be path, bytes, or file-like object
#             scale=2.0 # Example: 2x upscale
#         )
#     with open("upscaled_image.png", "wb") as f:
#         f.write(upscaled_image_bytes)
```

### Audio Synthesis (Text-to-Speech)

Generate speech from text.

**Synchronous Example:**

```python
from venice_ai import VeniceClient
from venice_ai.types.audio import Voice

with VeniceClient() as client:
    audio_bytes = client.audio.create_speech(
        model="tts-kokoro", # Or your preferred TTS model
        input="Hello from Venice AI audio!",
        voice=Voice.KOKORO_DEFAULT # Or other voice options like "kokoro-style-energetic"
    )
    with open("output.mp3", "wb") as f:
        f.write(audio_bytes)
    print("Audio saved to output.mp3")
```

**Streaming Example (Conceptual):**

```python
# with VeniceClient() as client: # or AsyncVeniceClient
#     audio_stream = client.audio.create_speech(
#         model="tts-kokoro",
#         input="This is a streamed audio example.",
#         voice=Voice.KOKORO_DEFAULT,
#         stream=True
#     )
#     with open("streamed_output.mp3", "wb") as f:
#         for chunk in audio_stream:
#             f.write(chunk)
#     print("Streamed audio saved.")
```

### Embeddings Creation

```python
from venice_ai import VeniceClient

with VeniceClient() as client:
    response = client.embeddings.create(
        model="text-embedding-bge-m3", # Or your preferred embeddings model
        input="The Venice AI Python client makes API interaction seamless."
        # input=["Batch sentence 1", "Batch sentence 2"] # For batching
    )
    if response.data and response.data[0].embedding:
        first_embedding_vector = response.data[0].embedding
        print(f"Generated embedding vector (first 5 dimensions): {first_embedding_vector[:5]}")
        print(f"Total dimensions of the vector: {len(first_embedding_vector)}")
    else:
        print("No embedding data received.")
```

### Model Management

```python
from venice_ai import VeniceClient

with VeniceClient() as client:
    # List all models
    models_list = client.models.list()
    print(f"Total models available: {len(models_list.data)}")
    if models_list.data:
        print(f"First model: {models_list.data[0].id}")

    # List text models
    text_models = client.models.list(type="text")
    print(f"Text models: {[m.id for m in text_models.data]}")
```

### API Key Management

```python
from venice_ai import VeniceClient

# Ensure your client is initialized with an ADMIN API key for these operations
# with VeniceClient(api_key="YOUR_ADMIN_API_KEY") as client:
    # List API Keys
    # api_keys_list = client.api_keys.list()
    # print(f"Found {len(api_keys_list.data)} API keys.")

    # Create an API Key (handle the returned key securely - it's shown only once!)
    # new_key_response = client.api_keys.create(
    #     description="My new inference key",
    #     apiKeyType="INFERENCE"
    # )
    # print(f"Created new API key (prefix): {new_key_response.data.last6Chars}")
    # print(f"IMPORTANT: Full API Key: {new_key_response.data.apiKey} - Store it securely NOW!")
```

**Warning:** API Key management operations typically require admin privileges. The API key used to initialize the client must have sufficient permissions.

### Billing Information

```python
from venice_ai import VeniceClient

# with VeniceClient() as client:
    # usage_data = client.billing.get_usage()
    # print(f"Total VCU used: {usage_data.total_vcu_consumed}")
    # print(f"Total USD used: {usage_data.total_usd_consumed}")
```

For more detailed examples of other functionalities (Characters, specific parameters for each endpoint), please refer to the [Showcase Application](#showcase-application) and the official [API Documentation](#documentation).

## Error Handling

The Venice AI client library uses a custom hierarchy of exceptions to help you handle API errors gracefully. All library-specific errors inherit from `venice_ai.exceptions.VeniceError`.

Common exceptions include:

- `venice_ai.exceptions.APIError`: Base class for errors returned by the API (e.g., HTTP 4xx, 5xx).
- `venice_ai.exceptions.AuthenticationError`: For API key issues (HTTP 401).
- `venice_ai.exceptions.InvalidRequestError`: For bad request parameters (HTTP 400).
- `venice_ai.exceptions.RateLimitError`: When API rate limits are exceeded (HTTP 429).
- `venice_ai.exceptions.NotFoundError`: For non-existent resources (HTTP 404).
- `venice_ai.exceptions.APIConnectionError`: For network connectivity issues.
- `venice_ai.exceptions.APITimeoutError`: For request timeouts.

Example:

```python
from venice_ai import VeniceClient, exceptions

try:
    with VeniceClient() as client:
        response = client.chat.completions.create(
            model="non_existent_model",
            messages=[{"role": "user", "content": "Test"}]
        )
except exceptions.NotFoundError as e:
    print(f"Model not found: {e}")
except exceptions.AuthenticationError as e:
    print(f"Authentication failed. Check your API key: {e}")
except exceptions.APIError as e:
    print(f"An API error occurred: {e.status_code} - {e}")
except exceptions.VeniceError as e:
    print(f"A Venice AI client error occurred: {e}")
```

Always check the specific exception type and its attributes (like `e.status_code`, `e.request`, `e.response`) for more details.

## Advanced Usage

### Using a Custom `httpx` Client

You can provide your own `httpx.Client` or `httpx.AsyncClient` instance during initialization if you need custom configurations (e.g., proxies, custom SSL settings, transport options):

```python
import httpx
from venice_ai import VeniceClient, AsyncVeniceClient

# Synchronous
custom_sync_transport = httpx.HTTPTransport(retries=1)
# Note: The custom httpx.Client should ideally be managed as a context manager itself.
# If passing an externally managed httpx.Client, ensure its lifecycle is handled.
# For simplicity, this example shows direct instantiation.
http_client_instance = httpx.Client(transport=custom_sync_transport, timeout=30.0)
try:
    client = VeniceClient(api_key="YOUR_API_KEY", http_client=http_client_instance)
    # Use client...
    models = client.models.list()
    print(f"Sync models: {len(models.data)}")
finally:
    http_client_instance.close() # Close the httpx client if managed externally

# Asynchronous
async def use_custom_async_client():
    custom_async_transport = httpx.AsyncHTTPTransport(retries=1)
    async_http_client_instance = httpx.AsyncClient(transport=custom_async_transport, timeout=30.0)
    try:
        async_client = AsyncVeniceClient(api_key="YOUR_API_KEY", http_client=async_http_client_instance)
        # Use async_client...
        models = await async_client.models.list()
        print(f"Async models: {len(models.data)}")
    finally:
        await async_http_client_instance.aclose() # Close the httpx client if managed externally

# asyncio.run(use_custom_async_client())
```

### Understanding Streaming

When `stream=True` is used (e.g., in `chat.completions.create`), the method returns an iterator (`Stream` or `AsyncStream` object from `venice_ai.streaming`). These objects wrap the raw data chunks from the API.

```python
from venice_ai import VeniceClient
from venice_ai.streaming import Stream # For type hinting or direct use if needed

with VeniceClient() as client:
    stream_response: Stream = client.chat.completions.create( # type: ignore
        model="llama-3.2-3b",
        messages=[{"role": "user", "content": "Tell me a very long story about a brave knight."}],
        stream=True,
        max_completion_tokens=50 # Example: limit stream length
    )
    full_story = []
    for chunk in stream_response:
        # chunk is a dict representing ChatCompletionChunk
        if chunk.choices and chunk.choices[0].delta.content:
            # print(chunk.choices[0].delta.content, end="") # For live printing
            full_story.append(chunk.choices[0].delta.content)
    # print("\n--- End of Story ---")
    # final_text = "".join(full_story)
    # print(f"\nFull story assembled: {final_text[:100]}...")
```

The `Stream` and `AsyncStream` wrappers handle resource management and provide a consistent iteration interface.

### Token Estimation

The library includes a utility for estimating token counts:

```python
from venice_ai.utils import estimate_token_count

text = "This is some sample text to estimate tokens for."
# By default, uses cl100k_base encoding (common for many OpenAI models)
# and falls back to a heuristic if tiktoken is not installed.
count = estimate_token_count(text)
print(f"Estimated tokens: {count}")
```

## Showcase Application

This project includes a Streamlit application ([`app.py`](app.py)) that demonstrates various features of the `venice-ai` library, providing an interactive UI for chat, image generation, audio synthesis, model listing, and more.

To run the showcase application:

```bash
# Ensure you have installed dev dependencies (including Streamlit):
poetry install --with dev
poetry run streamlit run app.py
```

## Testing

The library includes a comprehensive test suite using `pytest`.

To run all tests (unit, E2E, benchmarks) and generate a coverage report:

```bash
# Ensure dev dependencies are installed:
poetry install --with dev
poetry run python test_runner.py --group all --coverage --html
```

The test runner ([`test_runner.py`](test_runner.py)) also supports an interactive mode and options to run specific test groups or files. Run `poetry run python test_runner.py --help` for more options.

## Documentation

This project, venice-ai, has docs @ [venice-ai docs](https://venice-ai.readthedocs.io/)

This Python client library is generated using Sphinx from the docstrings within the codebase. It can be built locally from the `docs/` directory.

Detailed API documentation for the Venice.ai API itself is available @ [https://docs.venice.ai/api-reference](https://docs.venice.ai/api-reference)

## Contributing

Contributions are welcome! Please feel free to open issues for bugs or feature requests. If you'd like to contribute code, please see our (upcoming) `CONTRIBUTING.md` for guidelines on setting up your development environment, running tests, and submitting pull requests.

## License

This project is licensed under the MIT License.

---
