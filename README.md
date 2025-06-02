# Venice AI Python Client

[![PyPI version](https://img.shields.io/pypi/v/venice-ai.svg)](https://pypi.org/project/venice-ai/)
[![CI Status](https://github.com/venice-ai/venice-ai-python/actions/workflows/python-package.yml/badge.svg)](https://github.com/venice-ai/venice-ai-python/actions/workflows/python-package.yml)
[![Coverage Status](https://img.shields.io/codecov/c/github/venice-ai/venice-ai-python.svg)](https://codecov.io/gh/venice-ai/venice-ai-python)
[![Downloads](https://static.pepy.tech/badge/venice-ai)](https://pepy.tech/project/venice-ai)
[![Downloads](https://static.pepy.tech/badge/venice-ai/month)](https://pepy.tech/project/venice-ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Versions](https://img.shields.io/pypi/pyversions/venice-ai.svg)](https://pypi.org/project/venice-ai/)

Developed to benchmark and explore the full capabilities of the Venice.ai API, the venice-ai Python package has evolved into a comprehensive client library for developers. This library provides convenient access to Venice.ai's powerful features, including chat completions, image generation, audio synthesis, embeddings, and more, with support for both synchronous and asynchronous operations.

## Table of Contents

- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [API Key Setup](#api-key-setup)
- [Usage](#usage)
  - [Client Initialization](#client-initialization)
  - [Chat Completions](#chat-completions)
  - [Image Generation](#image-generation)
- [Showcase Application](#showcase-application)
- [Testing](#testing)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## Features

- Intuitive Pythonic interface for all Venice.ai API endpoints.
- Support for both synchronous and asynchronous operations.
- Comprehensive model support for Chat, Image Generation, Embeddings, etc.
- Streaming capabilities for chat completions and audio.
- Built-in utilities for tasks like token estimation.
- Robust error handling and type-hinted for a better developer experience.
- Detailed API documentation generated with Sphinx.

## Getting Started

### Prerequisites

Python 3.11 or higher.

### Installation

You can install the Venice AI client library from PyPI:

```bash
pip install venice-ai
```

Alternatively, to install the latest development version from source:

```bash
git clone https://github.com/venice-ai/venice-ai-python.git
cd venice-ai-python
poetry install
```

### API Key Setup

To use the Venice AI API, you need an API key.

The client library expects the API key to be available as an environment variable:

```bash
export VENICE_API_KEY="your_api_key_here"
```

Alternatively, you can pass the API key directly when initializing the client, though using environment variables is recommended for security.

## Usage

### Client Initialization

Synchronous Client:

```python
from venice_ai import VeniceClient

client = VeniceClient()
# If API key is not set as an environment variable:
# client = VeniceClient(api_key="your_api_key_here")
```

Asynchronous Client:

```python
import asyncio
from venice_ai import AsyncVeniceClient

async def main():
    async_client = AsyncVeniceClient()
    # If API key is not set as an environment variable:
    # async_client = AsyncVeniceClient(api_key="your_api_key_here")

    # Example: Asynchronous Chat Completion
    try:
        print("Attempting asynchronous chat completion...")
        response = await async_client.chat.completions.create(
            model="llama-3.2-3b", # Or your preferred model
            messages=[{"role": "user", "content": "Hello asynchronously from Venice AI!"}]
        )
        if response.choices:
            print("Async Chat Response:", response.choices[0].message.content)
        else:
            print("No response choices received.")
    except Exception as e: # Catching a general exception for API or client errors
        print(f"An unexpected error occurred during async chat: {e}")
    finally:
        print("Closing async client...")
        await async_client.close()
        print("Async client closed.")

if __name__ == "__main__":
    asyncio.run(main())
```

It's important to `await async_client.close()` when you're finished using the asynchronous client. This ensures that underlying HTTP resources and connections are properly released, preventing potential resource leaks in your application.

### Chat Completions

Non-streaming example:

```python
from venice_ai import VeniceClient

# Ensure VENICE_API_KEY is set in your environment,
# or initialize the client with client = VeniceClient(api_key="your_api_key_here")
client = VeniceClient()
response = client.chat.completions.create(
    model="llama-3.2-3b", # Or your preferred model
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ]
)
print(response.choices[0].message.content)
```

Streaming example:

```python
from venice_ai import VeniceClient

client = VeniceClient()
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

### Image Generation

Example:

```python
from venice_ai import VeniceClient
import base64
from PIL import Image
import io

client = VeniceClient()
response = client.image.generate(
    model="venice-sd35", # Or your preferred image model
    prompt="A futuristic cityscape at sunset",
    width=1024,
    height=1024
)
# Assuming response.images[0] contains base64 encoded image data
if response.images:
    img_b64 = response.images[0]
    img_bytes = base64.b64decode(img_b64)
    # To display or save the image (e.g., using Pillow)
    # pil_image = Image.open(io.BytesIO(img_bytes))
    # pil_image.show()
    # pil_image.save("generated_image.png")
    print("Image generated successfully (first image data received).")
```

<!--
### Audio Transcription

Example of transcribing an audio file using the `whisper-1` model:

```python
from venice_ai import VeniceClient
# Make sure you have an audio file (e.g., "my_audio.mp3") in the specified path.

client = VeniceClient()
try:
    # Replace "path/to/your/my_audio.mp3" with the actual path to your audio file.
    with open("path/to/your/my_audio.mp3", "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1", # Or your preferred transcription model
            file=audio_file
        )
    print("Transcription:")
    print(transcription.text)
except FileNotFoundError:
    print("Error: Audio file not found. Please update the path to your audio file.")
except Exception as e: # Catching a general exception for other potential API or client errors
    print(f"An error occurred during transcription: {e}")
```
-->

### Embeddings Creation

Example of creating embeddings for a piece of text:

```python
from venice_ai import VeniceClient

client = VeniceClient()
try:
    response = client.embeddings.create(
        model="text-embedding-ada-002", # Or your preferred embeddings model
        input="The Venice AI Python client makes API interaction seamless."
    )
    # The response.data contains a list of embedding objects.
    if response.data and response.data[0].embedding:
        first_embedding_vector = response.data[0].embedding
        print(f"Generated embedding vector (first 5 dimensions): {first_embedding_vector[:5]}")
        print(f"Total dimensions of the vector: {len(first_embedding_vector)}")
    else:
        print("No embedding data received.")
except Exception as e: # Catching a general exception for API or client errors
    print(f"An error occurred during embedding creation: {e}")
```

For more detailed examples of other functionalities (Audio, Embeddings, Billing, API Keys, Characters), please refer to the [Showcase Application](#showcase-application) and the official [API Documentation](#documentation).

## Showcase Application

This project includes a Streamlit application ([`app.py`](app.py)) that demonstrates various features of the `venice-ai` library.

To run the showcase application:

```bash
# Ensure you have installed dev dependencies (including Streamlit):
# poetry install --with dev
poetry run streamlit run app.py
```

## Testing

The library includes a comprehensive test suite using `pytest`.

To run all tests (unit, E2E, benchmarks) and generate a coverage report:

```bash
# Ensure dev dependencies are installed: poetry install --with dev
poetry run python test_runner.py --group all --coverage --html
```

The test runner ([`test_runner.py`](test_runner.py)) also supports an interactive mode and options to run specific test groups or files. Run `poetry run python test_runner.py --help` for more options.

## Documentation

Detailed API documentation for the Venice.ai API is available at: https://docs.venice.ai/api-reference

The documentation is generated using Sphinx from the docstrings within the codebase.

## Contributing

Please feel free to open issues.

## License

This project is licensed under the MIT License.
