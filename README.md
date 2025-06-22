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
  - [Advanced HTTP Client Configuration](#advanced-http-client-configuration)
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
- Automatic request retries for transient errors and configurable HTTP status codes.
- Comprehensive testing suite with `test_runner.py` for easy execution.
- Detailed API documentation generated with Sphinx.
- Support for `logprobs` and `top_logprobs` in chat completions to retrieve token likelihoods.

## Getting Started

### Prerequisites

Python 3.11 or higher.

### Installation

You can install the Venice AI client library from PyPI:

```bash
pip install venice-ai
```

To include optional dependencies for token estimation:

```bash
pip install venice-ai[tokenizers]
```

Alternatively, to install the latest development version from source (recommended if you want to contribute or need the absolute latest changes not yet released on PyPI):

```bash
git clone https://github.com/sethbang/venice-ai.git # Or your fork
cd venice-ai
poetry install
```

Note: `poetry install` installs main dependencies. For development or running all tests, install with dev dependencies: `poetry install --with dev`. To include optional tokenizers support: `poetry install --extras "tokenizers"`.

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
with VeniceClient(default_timeout=60.0) as client: # Added default_timeout
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
    async with AsyncVeniceClient(default_timeout=60.0) as async_client: # Added default_timeout
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

# Option 1: API key from environment variable VENICE_API_KEY, with default timeout
client = VeniceClient(default_timeout=30.0) # Added default_timeout

# Option 2: Pass API key directly and custom timeout
# client = VeniceClient(api_key="your_api_key_here", default_timeout=45.0)

# Using as a context manager (recommended for proper resource cleanup):
with VeniceClient(api_key="your_api_key_here", default_timeout=30.0) as client: # Added default_timeout
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
    # Option 1: API key from environment variable VENICE_API_KEY, with default timeout
    async_client = AsyncVeniceClient(default_timeout=30.0) # Added default_timeout

    # Option 2: Pass API key directly and custom timeout
    # async_client = AsyncVeniceClient(api_key="your_api_key_here", default_timeout=45.0)

    # Using as an async context manager (recommended):
    async with AsyncVeniceClient(api_key="your_api_key_here", default_timeout=30.0) as async_client: # Added default_timeout
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

The response objects (`response`, `chunk`) are Pydantic models. You can explore their structure for more details (see `src/venice_ai/types/chat.py` or the Sphinx-generated API documentation).

**Model Context Windows:**

Different models support different context window sizes. For example, the "Venice Large" model supports up to 128k tokens, allowing for extensive conversations or document processing. Use the `max_completion_tokens` parameter to control response length within the model's context limits.

**Parameters:**

:param logit_bias: Modify the likelihood of specified tokens appearing in the completion. Accepts a JSON object that maps tokens (specified by their token ID in the tokenizer) to an associated bias value from -100 to 100.
:type logit_bias: Optional[Dict[str, int]]

:param logprobs: Whether to return log probabilities of the output tokens. If `True`, the `logprobs` field will be populated in the `choices` of the response. Defaults to `False`.
:type logprobs: Optional[bool]

:param parallel_tool_calls: Whether to enable parallel function calling during tool use.
:type parallel_tool_calls: Optional[bool]

:param top_logprobs: An integer between 0 and 5 specifying the number of most likely tokens to return at each token position, each with an associated log probability. Requires `logprobs` to be `True`.
:type top_logprobs: Optional[int]

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

**Example with Venice Large (128k context window):**

```python
from venice_ai import VeniceClient

with VeniceClient() as client:
    # Venice Large supports up to 128k tokens, ideal for long documents or conversations
    response = client.chat.completions.create(
        model="venice-large",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that can analyze long documents."},
            {"role": "user", "content": "Please analyze this extensive document..."}  # Can include very long content
        ],
        max_completion_tokens=4000  # Can use higher values with Venice Large's 128k context
    )
    print(response.choices[0].message.content)
```

**Example with `logprobs` and `top_logprobs`:**

```python
from venice_ai import VeniceClient

with VeniceClient() as client:
    response = client.chat.completions.create(
        model="llama-3.2-3b", # Or your preferred model
        messages=[
            {"role": "user", "content": "What is the color of the sky?"}
        ],
        logprobs=True,
        top_logprobs=2 # Request the top 2 most likely tokens at each position
    )

    # Example of accessing logprobs data
    if response.choices and response.choices[0].logprobs:
        print("Logprobs received.")
        first_choice_logprobs = response.choices[0].logprobs
        if first_choice_logprobs.content:
            for i, token_logprob in enumerate(first_choice_logprobs.content[:2]): # Display for first 2 generated tokens
                print(f"Token {i+1}: '{token_logprob.token}' (logprob: {token_logprob.logprob:.4f})")
                if token_logprob.top_logprobs:
                    print(f"  Top {len(token_logprob.top_logprobs)} alternative tokens:")
                    for alt_token_logprob in token_logprob.top_logprobs:
                        print(f"    - '{alt_token_logprob.token}' (logprob: {alt_token_logprob.logprob:.4f})")
    else:
        print("No logprobs data in response or choices.")

    print(f"\nMain response content: {response.choices[0].message.content}")
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
with VeniceClient() as client:
    response = client.image.simple_generate(
        model="venice-sd35",
        prompt="A cute cat wearing a small hat",
        size="512x512"
    )
    # Process response.data[0].b64_json or response.data[0].url
```

**Image Upscaling:**

```python
with VeniceClient() as client:
    with open("path/to/your/image.png", "rb") as img_file:
        upscaled_image_bytes = client.image.upscale(
            image=img_file, # Can be path, bytes, or file-like object
            scale=2.0 # Example: 2x upscale
        )
    with open("upscaled_image.png", "wb") as f:
        f.write(upscaled_image_bytes)
```

**Listing Image Styles:**

```python
with VeniceClient() as client:
    styles_response = client.image.list_styles()
    for style in styles_response.data:
        print(f"Available style: {style}")
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

**Note:** Different models have varying capabilities and context window sizes. For example, "Venice Large" supports up to 128k tokens, making it ideal for processing extensive documents or maintaining long conversations. Refer to the official Venice AI documentation for detailed specifications of each model.

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

### Advanced HTTP Client Configuration

The SDK allows for advanced configuration of the underlying `httpx` client, enabling scenarios like custom mTLS, specific proxy setups, or detailed transport logging. There are now three main ways to achieve this:

1.  **Passing a Pre-configured `httpx.Client` / `httpx.AsyncClient`**:
    You can provide your own `httpx.Client` or `httpx.AsyncClient` instance. The SDK will use it directly but will still manage `base_url`, `timeout` (if not more specific on your client), and authentication. **You are responsible for closing your client instance.**

    ```python
    import httpx
    from venice_ai import VeniceClient # or AsyncVeniceClient

    # Synchronous example
    my_custom_httpx_client = httpx.Client(proxies={"all://": "http://localhost:8080"}, timeout=60.0)
    try:
        client = VeniceClient(api_key="YOUR_API_KEY", http_client=my_custom_httpx_client)
        # Use the client...
        models = client.models.list()
        print(f"Found {len(models.data)} models.")
    finally:
        # Important: Close your custom client if it's not managed elsewhere (e.g., as a context manager)
        if not my_custom_httpx_client.is_closed:
            my_custom_httpx_client.close()

    # For AsyncVeniceClient, use httpx.AsyncClient and await its aclose() method.
    ```

2.  **Passing Common `httpx` Settings Directly**:
    If you don't provide an `http_client` instance, you can pass `httpx.Client` / `httpx.AsyncClient` constructor arguments (e.g., `proxy`, `transport`, `limits`, `verify`) directly to the SDK client. The SDK will create and manage its internal `httpx` client with these settings.

    ```python
    from venice_ai import VeniceClient # or AsyncVeniceClient
    import httpx # For httpx.Limits, httpx.HTTPTransport

    # Synchronous example
    client = VeniceClient(
        api_key="YOUR_API_KEY",
        proxies={"all://": "http://localhost:8080"}, # Example proxy
        transport=httpx.HTTPTransport(retries=3),    # Example custom transport
        limits=httpx.Limits(max_connections=50),     # Example connection limits
        verify=False,                                # Example: disable SSL verification (use with caution)
        default_timeout=30.0                         # Global default timeout for requests
    )
    with client: # SDK manages the internal httpx client's lifecycle
        models = client.models.list()
        print(f"Found {len(models.data)} models.")

    # AsyncVeniceClient works similarly with corresponding async httpx types.
    ```

3.  **Configuring Automatic Retries**:
    The SDK automatically retries requests on transient network errors and specific HTTP status codes (e.g., 429, 500, 502, 503, 504) by leveraging `httpx-retries`. This behavior is configurable through the following parameters in the `VeniceClient` and `AsyncVeniceClient` constructors:

    - `max_retries` (int, default: 2): Maximum number of retries.
    - `retry_backoff_factor` (float, default: 0.1): Backoff factor for calculating delay between retries.
    - `retry_status_forcelist` (list[int], default: `[429, 500, 502, 503, 504]`): HTTP status codes to retry on.
    - `retry_respect_retry_after_header` (bool, default: True): Whether to respect `Retry-After` headers.

    ```python
    from venice_ai import VeniceClient # or AsyncVeniceClient

    # Example: Customize retry behavior
    client = VeniceClient(
        api_key="YOUR_API_KEY",
        max_retries=5,
        retry_backoff_factor=0.5,
        retry_status_forcelist=[429, 500, 502, 503, 504, 520], # Adding 520 to the list
        retry_respect_retry_after_header=True, # Explicitly setting
        default_timeout=30.0 # Global default timeout for requests
    )
    with client:
        # Use the client...
        try:
            models = client.models.list()
            print(f"Found {len(models.data)} models with custom retry settings.")
        except Exception as e:
            print(f"An error occurred: {e}")

    # AsyncVeniceClient works similarly.
    ```

For a detailed explanation of all supported parameters and more advanced use cases, please refer to the "Advanced HTTP Client Configuration" section in our [API Reference documentation](docs/api.rst).

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

**Note:** For accurate token counting, install the optional `tiktoken` dependency:

```bash
pip install venice-ai[tokenizers]
# or with poetry:
poetry install --extras "tokenizers"
```

Without `tiktoken`, the library will use a simple heuristic estimation method.

## Showcase Application

This project focuses on the core Venice AI Python SDK. For an interactive demonstration of the library's capabilities, check out our separate [Venice AI Streamlit Demo](https://github.com/venice-ai/streamlit-demo) repository, which provides a comprehensive UI for chat, image generation, audio synthesis, model listing, and more.

The demo is now available as a separate repository for easier deployment and to keep this package's dependencies minimal.

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

Contributions are welcome! Please feel free to open issues for bugs or feature requests. If you'd like to contribute code, please see our [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines on reporting issues.

## License

This project is licensed under the MIT License.

---
