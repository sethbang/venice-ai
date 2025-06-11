E2E Test Utilities
==================

This document describes the utility functions available for end-to-end testing of the Venice AI client library.

Overview
--------

The Venice AI E2E test utilities provide helper functions that simplify writing tests for the Venice AI client library. These utilities are available in the ``e2e_tests.utils.helpers`` module and are designed to be used with pytest.

Test Utility Functions
----------------------

Model Selection for Tests
~~~~~~~~~~~~~~~~~~~~~~~~~

get_test_model_id
^^^^^^^^^^^^^^^^^

.. code-block:: python

    async def get_test_model_id(
        client: Union[VeniceClient, AsyncVeniceClient], 
        model_type: ModelType, 
        required_capabilities: Optional[List[str]] = None, 
        preferred_models: Optional[List[str]] = None
    ) -> str

Selects a suitable model ID for tests using a prioritized selection strategy.

**Parameters:**

- ``client``: A ``~venice_ai.VeniceClient`` or ``~venice_ai.AsyncVeniceClient`` instance.
- ``model_type``: The type of model to select (e.g., "text", "image").
- ``required_capabilities``: Optional list of capabilities the model must have (e.g., ["supportsFunctionCalling"]).
- ``preferred_models``: Optional list of preferred model IDs to check first.

**Selection Strategy:**

1. First checks environment variables (e.g., ``E2E_TEXT_MODEL_TOOL_CALLS``) that match the requested model type
2. Then checks the preferred models list if provided
3. Falls back to dynamic selection using ``get_filtered_models``
4. Skips the test if no suitable model is found

**Returns:**

- A model ID string suitable for the test.

**Example:**

.. code-block:: python

    from e2e_tests.utils.helpers import get_test_model_id

    def test_function_calling(venice_client):
        model_id = get_test_model_id(
            venice_client,
            model_type="text",
            required_capabilities=["supportsFunctionCalling"],
            preferred_models=["llama-3.2-3b", "qwen-2.5-qwq-32b"]
        )
        
        # Use the selected model in the test
        response = venice_client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "What's the weather like?"}],
            tools=[...]
        )
        # rest of test...

get_model_capabilities_for_test
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    async def get_model_capabilities_for_test(
        client: Union[VeniceClient, AsyncVeniceClient], 
        model_id: str
    ) -> Optional[ModelCapabilities]

Test-specific wrapper for retrieving model capabilities with test-appropriate error handling.

**Parameters:**

- ``client``: A ``~venice_ai.VeniceClient`` or ``~venice_ai.AsyncVeniceClient`` instance.
- ``model_id``: The ID of the model to get capabilities for.

**Returns:**

- A ``ModelCapabilities`` object if found, or ``None``.
- May cause test failure via ``pytest.fail()`` if capabilities are unexpectedly None.

**Example:**

.. code-block:: python

    from e2e_tests.utils.helpers import get_model_capabilities_for_test
    import pytest

    @pytest.mark.asyncio
    async def test_model_has_required_capability(venice_client):
        capabilities = await get_model_capabilities_for_test(venice_client, "llama-3.2-3b")
        assert capabilities.get("supportsFunctionCalling") is True

get_filtered_models (Test Helper)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    async def get_filtered_models(
        client: Union[VeniceClient, AsyncVeniceClient],
        model_type: ModelType,
        required_capabilities: Optional[List[str]] = None,
        filter_func: Callable[[Model], Awaitable[bool]] | Callable[[Model], bool] | None = None,
    ) -> List[Model]:

Test-specific wrapper for retrieving a list of models, filtered by type, capabilities, and an optional custom filter function. This helper internally calls the main ``venice_ai.utils.get_filtered_models`` and provides additional handling for applying a custom ``filter_func``.

**Parameters:**

- ``client``: A ``~venice_ai.VeniceClient`` or ``~venice_ai.AsyncVeniceClient`` instance.
- ``model_type``: The type of model to filter by (e.g., "text", "image").
- ``required_capabilities``: Optional list of capabilities the model must have.
- ``filter_func``: Optional custom filter function that can be synchronous or asynchronous. Takes a ``Model`` object and returns a boolean indicating whether the model should be included.

**Returns:**

- ``List[Model]``: A list of ``Model`` objects that match the specified filters.

**Example:**

.. code-block:: python

    from e2e_tests.utils.helpers import get_filtered_models

    @pytest.mark.asyncio
    async def test_custom_model_filtering(venice_client):
        # Define a custom filter function
        def custom_filter(model):
            return "gpt" in model.get("id", "").lower()
        
        # Get filtered models with custom logic
        models = await get_filtered_models(
            venice_client,
            model_type="text",
            required_capabilities=["supportsFunctionCalling"],
            filter_func=custom_filter
        )
        
        # All returned models should match our criteria
        for model in models:
            assert "gpt" in model.get("id", "").lower()

Chat Message Generation
~~~~~~~~~~~~~~~~~~~~~~~

generate_sample_messages
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def generate_sample_messages(
        count: int = 1, 
        include_system_prompt: bool = False, 
        content_prefix: str = "Test message"
    ) -> List[Dict[str, Any]]

Creates sample message lists for tests with configurable content.

**Parameters:**

- ``count``: Number of user messages to generate.
- ``include_system_prompt``: Whether to include a system message at the beginning.
- ``content_prefix``: Prefix for user message content.

**Returns:**

- A list of message dictionaries suitable for the chat completions API.

**Example:**

.. code-block:: python

    from e2e_tests.utils.helpers import generate_sample_messages

    def test_chat_completion(venice_client):
        # Generate 3 messages with a system prompt
        messages = generate_sample_messages(
            count=3, 
            include_system_prompt=True,
            content_prefix="Answer briefly"
        )
        
        response = venice_client.chat.completions.create(
            model="llama-3.2-3b",
            messages=messages
        )
        # rest of test...

Response Validation
~~~~~~~~~~~~~~~~~~~

assert_chat_completion_structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def assert_chat_completion_structure(
        response_data: Dict[str, Any],
        is_streaming_chunk: bool = False,
        is_final_usage_chunk: bool = False
    )

Validates the structure of chat completion API responses.

**Parameters:**

- ``response_data``: The API response dictionary to validate.
- ``is_streaming_chunk``: Whether this is a streaming chunk (vs. full response).
- ``is_final_usage_chunk``: Whether this is a final streaming chunk with only usage.

**Assertions:**

- For full responses: Checks ``id``, ``object``, ``created``, ``model``, ``choices`` (with ``message``, ``finish_reason``), and optional ``usage``.
- For streaming chunks: Checks appropriate ``delta`` structure and handles empty ``delta`` with ``finish_reason``.
- For final usage chunks: Validates empty ``choices`` and presence of ``usage`` data.

**Example:**

.. code-block:: python

    from e2e_tests.utils.helpers import assert_chat_completion_structure

    def test_chat_completion(venice_client):
        response = venice_client.chat.completions.create(
            model="llama-3.2-3b",
            messages=[{"role": "user", "content": "Hello!"}]
        )
        
        # Validate response structure
        assert_chat_completion_structure(response)
        
        # Access content knowing structure is valid
        content = response["choices"][0]["message"]["content"]
        assert isinstance(content, str)

assert_tool_call_structure
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def assert_tool_call_structure(
        tool_call_data: Dict[str, Any],
        is_delta: bool = False
    )

Validates the structure of tool call objects within responses.

**Parameters:**

- ``tool_call_data``: The tool call dictionary to validate.
- ``is_delta``: Whether this is part of a streaming delta (vs. full object).

**Assertions:**

- For full tool calls: Checks ``id``, ``type``, ``function`` (with ``name``, ``arguments``).
- For deltas: Expects ``index``, handles optional presence of ``id``, ``type``, ``function``.

**Example:**

.. code-block:: python

    from e2e_tests.utils.helpers import assert_tool_call_structure

    def test_function_calling(venice_client):
        response = venice_client.chat.completions.create(
            model="llama-3.2-3b",
            messages=[{"role": "user", "content": "What's the weather like?"}],
            tools=[...]
        )
        
        # Check if response has tool calls
        if "tool_calls" in response["choices"][0]["message"]:
            tool_call = response["choices"][0]["message"]["tool_calls"][0]
            assert_tool_call_structure(tool_call)

Test Data Management
~~~~~~~~~~~~~~~~~~~~

load_test_data
^^^^^^^^^^^^^^

.. code-block:: python

    def load_test_data(
        filename: str,
        data_dir: str = "e2e_tests/data",
        mode: str = "rb"
    ) -> Union[str, bytes]

Loads test data files from the data directory.

**Parameters:**

- ``filename``: Name of the file to load.
- ``data_dir``: Directory path containing test data (default: "e2e_tests/data").
- ``mode``: File opening mode (default: "rb" for binary).

**Returns:**

- Content of the file as string or bytes, depending on the mode.

**Example:**

.. code-block:: python

    from e2e_tests.utils.helpers import load_test_data

    def test_image_upload(venice_client):
        # Load a test image
        image_data = load_test_data("sample_image.png")
        
        # Use the loaded image in a test
        result = venice_client.image.upscale(image=image_data)
        assert result["success"] is True

create_temp_test_file
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def create_temp_test_file(
        tmp_path_fixture: pathlib.Path,
        filename: str,
        content: Union[str, bytes],
        encoding: Optional[str] = 'utf-8'
    ) -> pathlib.Path

Creates a temporary file with specified content using pytest's ``tmp_path`` fixture.

**Parameters:**

- ``tmp_path_fixture``: The pytest ``tmp_path`` fixture.
- ``filename``: Name for the temporary file.
- ``content``: Content to write to the file (string or bytes).
- ``encoding``: Character encoding for string content (default: 'utf-8').

**Returns:**

- A ``pathlib.Path`` object pointing to the created file.

**Example:**

.. code-block:: python

    def test_file_operations(tmp_path):
        from e2e_tests.utils.helpers import create_temp_test_file
        
        # Create a temporary file with content
        temp_file = create_temp_test_file(
            tmp_path,
            "test_config.json",
            '{"api_key": "test_key", "model": "test-model"}'
        )
        
        # Use the file in the test
        assert temp_file.exists()
        assert temp_file.read_text() == '{"api_key": "test_key", "model": "test-model"}'

Best Practices
--------------

- **Model Selection**: Use ``get_test_model_id`` to ensure tests run with suitable models, making tests more robust across different environments.
- **Validation**: Use the assertion helpers to validate API responses, reducing test boilerplate and ensuring consistent validation.
- **Test Data**: Use ``load_test_data`` and ``create_temp_test_file`` to manage test data files properly.
- **Message Generation**: Use ``generate_sample_messages`` to quickly create valid message arrays for chat completion tests.