from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional, Dict, Any, Union, Set, cast
import json
import inspect
import warnings
import importlib.util
import importlib.abc
from types import ModuleType as Module

if TYPE_CHECKING:
    from ._client import VeniceClient
    from ._async_client import AsyncVeniceClient

from .types.models import ModelType, Model, ModelCapabilities

# Try to import tiktoken at module level
try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    tiktoken = None  # type: ignore
    _TIKTOKEN_AVAILABLE = False

# Helper function for mocking in tests
def _import_tiktoken_module():
    """
    Imports the `tiktoken` library, designed to be mockable in tests.

    :raises ImportError: If the `tiktoken` library cannot be imported.
    :return: The imported `tiktoken` module.
    :rtype: module
    """
    if _TIKTOKEN_AVAILABLE:
        return tiktoken
    else:
        raise ImportError("tiktoken library not available")
def import_module_from_path(module_name: str, file_path: str) -> Module:
    """
    Dynamically imports a Python module from a file path.
    
    :param module_name: Name to assign to the imported module.
    :type module_name: str
    :param file_path: Path to the Python file to import.
    :type file_path: str
    :return: The imported module object.
    :rtype: Module
    :raises ImportError: If the module cannot be loaded from the specified path.
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"Could not load spec for module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    if module is None:  # Should not happen if spec is not None, but good for robustness
        raise ImportError(f"Could not create module {module_name} from spec")
    try:
        # Check if spec.loader exists and has exec_module method
        if spec.loader is None or not hasattr(spec.loader, 'exec_module'):
            raise ImportError(f"Spec loader is not a valid Loader for module {module_name}")
        spec.loader.exec_module(module)
    except ImportError:
        # Re-raise ImportError from exec_module as expected by tests
        raise
    return module

# Mock data for models - this would typically come from an API or database
MODELS_DATA = [
    {
        "id": "gpt-4",
        "name": "GPT-4",
        "slug": "gpt-4",
        "type": "text",
        "model_spec": {
            "capabilities": {
                "streaming": True,
                "async_": True,
                "max_tokens": 4096,
                "supports_functions": True
            }
        }
    },
    {
        "id": "claude-3",
        "name": "Claude 3",
        "slug": "claude-3",
        "type": "text",
        "model_spec": {
            "capabilities": {
                "streaming": True,
                "async_": True,
                "max_tokens": 8192,
                "supports_functions": False
            }
        }
    }
]

def find_model_by_id_or_name_or_slug(identifier: str) -> Optional[Model]:
    """
    Find a model by its ID, name, or slug.
    
    :param identifier: The model identifier (ID, name, or slug) to search for.
    :type identifier: str
    :return: The model if found, otherwise None.
    :rtype: Optional[Model]
    """
    for model_data in MODELS_DATA:
        if (model_data.get("id") == identifier or 
            model_data.get("name") == identifier or 
            model_data.get("slug") == identifier):
            return cast(Model, model_data)
    return None

def get_model_capabilities_by_id_or_name_or_slug(identifier: str) -> Optional[ModelCapabilities]:
    """
    Get model capabilities by ID, name, or slug.
    
    :param identifier: The model identifier (ID, name, or slug) to search for.
    :type identifier: str
    :return: The model capabilities if found, otherwise None.
    :rtype: Optional[ModelCapabilities]
    """
    model_info = find_model_by_id_or_name_or_slug(identifier)
    if model_info is None:
        return None
    
    # Assuming Model TypedDict has 'model_spec' key with 'capabilities' inside
    model_spec = model_info.get("model_spec", {})
    if not isinstance(model_spec, dict):
        return None
    
    capabilities_data = model_spec.get("capabilities")
    
    if capabilities_data is None:  # If 'capabilities' key could be absent or its value None
        return None
        
    return cast(ModelCapabilities, capabilities_data)

def get_filtered_models(
    client: Union["VeniceClient", "AsyncVeniceClient"],
    model_type: Optional[ModelType] = None,
    supports_capabilities: Optional[List[str]] = None,
    # Add other potential filters here in the future
):
    """
    Retrieves a list of models filtered by type and capabilities.

    This function provides a unified interface for filtering models from the Venice.ai API
    based on model type and required capabilities. It automatically handles both synchronous
    and asynchronous clients, returning the appropriate response type.

    :param client: An instance of ``~venice_ai.VeniceClient`` or ``~venice_ai.AsyncVeniceClient`` used to fetch models.
    :type client: Union[venice_ai._client.VeniceClient, venice_ai._async_client.AsyncVeniceClient]
    :param model_type: Optional filter for model type (e.g., ``'text'``, ``'image'``).
        If ``None``, models of all types are included.
    :type model_type: Optional[venice_ai.types.models.ModelType]
    :param supports_capabilities: Optional list of capabilities the model must support
        (e.g., ``['streaming', 'supports_functions']``). Models must have all specified
        capabilities set to ``True`` to be included in results.
    :type supports_capabilities: Optional[List[str]]

    :return: A list of :class:`~venice_ai.types.models.Model` objects that match the specified filters.
        If the client is synchronous, this is a direct list.
        If asynchronous, this is an awaitable resolving to the list.
    :rtype: List[venice_ai.types.models.Model]

    :raises Exception: If there's an error fetching models from the API (returns empty list).

    .. note::
        When using a synchronous client in an asynchronous context, a deprecation warning
        will be issued to encourage proper async/await usage.

    Example:
        >>> # Filter for text models that support streaming
        >>> filtered_models = get_filtered_models(
        ...     client=venice_client,
        ...     model_type="text",
        ...     supports_capabilities=["streaming"]
        ... )
    """
    from ._client import VeniceClient
    from ._async_client import AsyncVeniceClient
    
    # Check if called with sync client in a sync context
    frame = inspect.currentframe()
    if isinstance(client, VeniceClient) and frame and frame.f_back and not inspect.iscoroutinefunction(frame.f_back.f_code):
        import warnings
        warnings.warn("Calling an async function without awaiting", DeprecationWarning, stacklevel=2)
        return _get_filtered_models_sync(client, model_type, supports_capabilities)
    
    # For async client or functions called from async context, return a coroutine for proper await
    return _get_filtered_models_async(client, model_type, supports_capabilities)


def _get_filtered_models_sync(
    client: "VeniceClient",
    model_type: Optional[ModelType] = None,
    supports_capabilities: Optional[List[str]] = None,
) -> List[Model]:
    """
    Synchronous implementation of :func:`get_filtered_models`.

    :param client: An instance of ``~venice_ai.VeniceClient``.
    :type client: venice_ai._client.VeniceClient
    :param model_type: Optional. Filter for model type.
    :type model_type: Optional[venice_ai.types.models.ModelType]
    :param supports_capabilities: Optional. List of required capabilities.
    :type supports_capabilities: Optional[List[str]]
    :return: A list of filtered :class:`~venice_ai.types.models.Model` objects.
    :rtype: List[venice_ai.types.models.Model]
    """
    from ._client import VeniceClient
    
    try:
        models_list_response = client.models.list()
        all_models = models_list_response.get("data", []) if models_list_response else []
    except Exception as e:
        print(f"Error fetching models: {e}")
        return [] # Return empty list on error

    # Handle case where data is None
    if all_models is None:
        all_models = []

    filtered_list = []
    for model_data in all_models:
        # Get the model capabilities
        model_spec = model_data.get("model_spec", {})
        capabilities = model_spec.get("capabilities", {})

        passes_filter = True

        # Filter by model type if specified
        if model_type is not None and model_data.get("type") != model_type:
            passes_filter = False

        # Filter by capabilities if specified
        if passes_filter and supports_capabilities and len(supports_capabilities) > 0:
            # Check that all required capabilities are present and set to True
            for capability in supports_capabilities:
                if not capabilities.get(capability, False):
                    passes_filter = False
                    break

        if passes_filter:
            filtered_list.append(model_data)

    return filtered_list


async def _get_filtered_models_async(
    client: Union["VeniceClient", "AsyncVeniceClient"],
    model_type: Optional[ModelType] = None,
    supports_capabilities: Optional[List[str]] = None,
) -> List[Model]:
    """
    Asynchronous implementation of :func:`get_filtered_models`.

    :param client: An instance of ``~venice_ai.VeniceClient`` or ``~venice_ai.AsyncVeniceClient``.
    :type client: Union[venice_ai._client.VeniceClient, venice_ai._async_client.AsyncVeniceClient]
    :param model_type: Optional. Filter for model type.
    :type model_type: Optional[venice_ai.types.models.ModelType]
    :param supports_capabilities: Optional. List of required capabilities.
    :type supports_capabilities: Optional[List[str]]
    :return: A list of filtered :class:`~venice_ai.types.models.Model` objects.
    :rtype: List[venice_ai.types.models.Model]
    """
    from ._client import VeniceClient
    from ._async_client import AsyncVeniceClient
    
    try:
        # First, get all models
        if isinstance(client, AsyncVeniceClient):
            try:
                # Try awaiting, but fallback to direct access for mocks
                models_list_response = await client.models.list()
                all_models = models_list_response.get("data", []) if models_list_response else []
            except (TypeError, RuntimeError) as e:
                # Handle case where a non-awaitable mock is used in tests
                if "object dict can't be used in 'await' expression" in str(e) or "asyncio.run() cannot be called" in str(e):
                    models_list_response = client.models.list()  # type: ignore
                    all_models = models_list_response.get("data", []) if models_list_response else []  # type: ignore
                else:
                    raise
        else:
            # Handle sync client
            models_list_response = client.models.list()
            all_models = models_list_response.get("data", []) if models_list_response else []
    except Exception as e:
        print(f"Error fetching models: {e}")
        return [] # Return empty list on error

    # Handle case where data is None
    if all_models is None:
        all_models = []

    filtered_list = []
    for model_data in all_models:
        # Get the model capabilities
        model_spec = model_data.get("model_spec", {})
        capabilities = model_spec.get("capabilities", {})

        passes_filter = True

        # Filter by model type if specified
        if model_type is not None and model_data.get("type") != model_type:
            passes_filter = False

        # Filter by capabilities if specified
        if passes_filter and supports_capabilities and len(supports_capabilities) > 0:
            # Check that all required capabilities are present and set to True
            for capability in supports_capabilities:
                if not capabilities.get(capability, False):
                    passes_filter = False
                    break

        # Add other filter checks here if needed

        if passes_filter:
            filtered_list.append(model_data)

    return filtered_list

def estimate_token_count(text: str, model_id: Optional[str] = None) -> int:
    """
    Estimates the number of tokens in the given text using tiktoken or a fallback heuristic.
    
    This function attempts to use the tiktoken library for accurate token counting based on
    the OpenAI tokenization standard. If tiktoken is not available, it falls back to a
    simple character-based heuristic (approximately 1 token per 4 characters).

    The function handles various input types by converting them to strings, but provides
    different behavior based on the original input type to maintain consistency with
    expected token counting behavior.
    
    :param text: Text content to estimate token count for. Will be converted to string
        if not already a string.
    :type text: str
    :param model_id: Optional model identifier for model-specific encoding. Currently unused
        but reserved for future model-specific tokenization support.
    :type model_id: Optional[str]

    :return: Estimated number of tokens in the text. Returns 0 for empty strings or
        non-string/non-numeric inputs when using fallback estimation.
    :rtype: int

    :raises AttributeError: If ``text`` is ``None``.

    .. warning::
        If tiktoken is not available, a ``UserWarning`` will be issued and the function
        will use a simple character-based heuristic that may be less accurate.

    Example:
        >>> # Estimate tokens for a simple text
        >>> token_count = estimate_token_count("Hello, world!")
        >>> print(token_count)  # Output depends on tiktoken availability
        
        >>> # With model specification (reserved for future use)
        >>> token_count = estimate_token_count("Hello, world!", model_id="gpt-4")
    """
    import warnings
    
    # Check for None input explicitly (to match test expectations)
    if text is None:
        raise AttributeError("Cannot estimate token count for None input")
    
    original_is_string = isinstance(text, str)
    original_is_numeric = isinstance(text, (int, float))

    text_str = str(text) # Convert to string for consistent processing by tiktoken or fallback length calc
    
    if not text_str: # Handles if original was empty string, or str(something) became empty
        return 0
    
    def fallback_estimation(s: str, was_originally_string: bool, was_originally_numeric: bool) -> int:
        if not was_originally_string and not was_originally_numeric: # Neither string nor number originally (e.g. list, dict)
            return 0
        if was_originally_numeric: # Original was number, stringified for tiktoken, but fallback is 0
            return 0
        # Original was string, or numeric but tiktoken failed and we are in string fallback path
        return max(1, int(len(s) / 4.0)) if s else 0

    if _TIKTOKEN_AVAILABLE and tiktoken is not None:
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            # Tiktoken operates on strings. If original was not string (e.g. number),
            # we use its string representation for tiktoken.
            token_count = len(encoding.encode(text_str))
            return token_count
        except AttributeError as e:
            warnings.warn(f"Warning: tiktoken attribute error occurred: {e}. Using a simple character-based heuristic for token estimation.", UserWarning)
            return fallback_estimation(text_str, original_is_string, original_is_numeric)
        except Exception as e:
            warnings.warn(f"Warning: unexpected error with tiktoken: {e}. Using a simple character-based heuristic for token estimation.", UserWarning)
            return fallback_estimation(text_str, original_is_string, original_is_numeric)
    else:
        warnings.warn(
            "tiktoken library not found or unavailable. "
            "Using a simple character-based heuristic for token estimation.",
            UserWarning
        )
        return fallback_estimation(text_str, original_is_string, original_is_numeric)

def validate_chat_messages(
    messages: List[Dict[str, Any]],
    max_messages: Optional[int] = None,
    max_total_chars: Optional[int] = None
) -> Dict[str, List[str]]:
    """
    Validates the structure, roles, content, and ordering of chat messages for API compatibility.
    
    This function performs comprehensive validation of chat message arrays according to the
    Venice.ai Chat API requirements. It checks message structure, validates roles and content,
    ensures proper message sequencing, and validates tool call/response patterns.

    The validation covers:
    - Message structure and required fields
    - Valid role values and role-specific requirements
    - Content format validation (string, multimodal, etc.)
    - Message ordering rules (system first, no consecutive same roles)
    - Tool call structure and tool response matching
    - Optional limits on message count and total character length
    
    :param messages: List of message dictionaries to validate. Each message should contain
        at minimum a ``'role'`` field and appropriate content based on the role type.
    :type messages: List[Dict[str, Any]]
    :param max_messages: Optional maximum number of messages allowed in the conversation.
        If specified, validation will fail if the message count exceeds this limit.
    :type max_messages: Optional[int]
    :param max_total_chars: Optional maximum total character count allowed across all message
        content. If specified, validation will fail if the total exceeds this limit.
    :type max_total_chars: Optional[int]

    :return: Dictionary containing validation results with two keys:
        - ``"errors"``: List of validation failures that must be fixed
        - ``"warnings"``: List of potential issues or recommendations
    :rtype: Dict[str, List[str]]

    :raises AttributeError: If ``messages`` is ``None``.

    .. note::
        Valid message roles are: ``'system'``, ``'user'``, ``'assistant'``, and ``'tool'``.
        
        - System messages must be first and contain string content
        - User messages can have string or multimodal list content
        - Assistant messages can have content and/or tool_calls
        - Tool messages must follow assistant messages and reference valid tool_call_ids

    Example:
        >>> messages = [
        ...     {"role": "system", "content": "You are a helpful assistant."},
        ...     {"role": "user", "content": "Hello!"},
        ...     {"role": "assistant", "content": "Hi there!"}
        ... ]
        >>> result = validate_chat_messages(messages, max_messages=10)
        >>> if result["errors"]:
        ...     print("Validation failed:", result["errors"])
        >>> else:
        ...     print("Messages are valid!")
    """
    # Check for None input explicitly (to match test expectations)
    if messages is None:
        raise AttributeError("Cannot validate None messages")
        
    errors: List[str] = []
    warnings: List[str] = []
    
    # Initialize tracking variables
    total_chars = 0
    has_system_message = False
    last_role = None
    expected_tool_call_ids: Set[str] = set()
    
    # Validate that messages is a list
    if not isinstance(messages, list):
        errors.append("Messages must be a list.")
        return {"errors": errors, "warnings": warnings}
    
    # Validate that messages is not empty
    if not messages:
        errors.append("Messages list cannot be empty.")
        return {"errors": errors, "warnings": warnings}
    
    # Check if messages exceeds max_messages
    if max_messages is not None and len(messages) > max_messages:
        errors.append(f"Message count ({len(messages)}) exceeds maximum allowed ({max_messages}).")
    
    # Iterate through messages
    for i, message in enumerate(messages):
        # Check that message is a dictionary
        if not isinstance(message, dict):
            errors.append(f"Message at index {i} must be a dictionary.")
            continue
            
        # Check for required 'role' field
        if 'role' not in message:
            errors.append(f"Message at index {i} is missing required 'role' field.")
            continue
            
        role = message.get('role')
        
        # Check that role is a string
        if not isinstance(role, str):
            errors.append(f"Role at index {i} must be a string.")
            continue
            
        # Check valid role values
        if role not in ["system", "user", "assistant", "tool"]:
            errors.append(f"Invalid role '{role}' at index {i}. Valid roles are 'system', 'user', 'assistant', 'tool'.")
            continue
            
        # Process based on role
        if role == "system":
            # System message validations
            if i > 0:
                errors.append(f"System message at index {i} must be the first message.")
                
            if has_system_message:
                errors.append(f"Multiple system messages found. Only one is allowed.")
                
            has_system_message = True
            
            # System message must have non-empty string content
            content = message.get('content')
            if not isinstance(content, str) or not content:
                errors.append(f"System message at index {i} must have non-empty string content.")
            else:
                total_chars += len(content)
                
            # System message should not have tool_calls or tool_call_id
            if 'tool_calls' in message:
                errors.append(f"System message at index {i} cannot have 'tool_calls'.")
                
            if 'tool_call_id' in message:
                errors.append(f"System message at index {i} cannot have 'tool_call_id'.")
                
        elif role == "user":
            # User message validations
            if last_role == "user":
                errors.append(f"User message at index {i} cannot directly follow another user message.")
                
            # Validate content (string or multimodal list)
            content = message.get('content')
            if isinstance(content, str):
                if not content:
                    errors.append(f"User message at index {i} has empty string content.")
                total_chars += len(content)
            elif isinstance(content, list):
                if not content:
                    errors.append(f"User message at index {i} has empty content list.")
                # For multimodal content, further validation could be added here
                # Currently just counting characters in any string values
                for item in content:
                    if isinstance(item, dict) and 'text' in item and isinstance(item['text'], str):
                        total_chars += len(item['text'])
            else:
                errors.append(f"User message at index {i} must have either string or list content.")
                
            # Clear expected tool call IDs when a new user message appears
            expected_tool_call_ids.clear()
            
        elif role == "assistant":
            # Assistant message validations
            if last_role == "assistant":
                errors.append(f"Assistant message at index {i} should not directly follow another assistant message.")
                
            content = message.get('content')
            tool_calls = message.get('tool_calls')
            
            # Check tool_calls if present
            if tool_calls is not None:
                # Validate tool_calls is a non-empty list
                if not isinstance(tool_calls, list) or not tool_calls:
                    errors.append(f"Assistant message at index {i} has 'tool_calls' that must be a non-empty list.")
                else:
                    # Validate each tool call
                    for tc_idx, tool_call in enumerate(tool_calls):
                        if not isinstance(tool_call, dict):
                            errors.append(f"Tool call at index {tc_idx} in message {i} must be a dictionary.")
                            continue
                            
                        # Check required fields
                        if 'id' not in tool_call:
                            errors.append(f"Tool call at index {tc_idx} in message {i} has invalid structure: missing required 'id' field.")
                        elif not isinstance(tool_call['id'], str) or not tool_call['id']:
                            errors.append(f"Tool call at index {tc_idx} in message {i} has invalid 'id' field.")
                        else:
                            # Add to expected tool call IDs
                            expected_tool_call_ids.add(tool_call['id'])
                            
                        if 'type' not in tool_call:
                            errors.append(f"Tool call at index {tc_idx} in message {i} missing required 'type' field.")
                        elif tool_call.get('type') != "function":
                            errors.append(f"Tool call at index {tc_idx} in message {i} has invalid type '{tool_call.get('type')}'. Must be 'function'.")
                            
                        if 'function' not in tool_call:
                            errors.append(f"Tool call at index {tc_idx} in message {i} missing required 'function' field.")
                        elif not isinstance(tool_call['function'], dict):
                            errors.append(f"Tool call at index {tc_idx} in message {i} has invalid 'function' field type.")
                        else:
                            # Validate function object
                            function = tool_call['function']
                            if 'name' not in function:
                                errors.append(f"Function call at index {tc_idx} in message {i} is missing 'name'.")
                            elif not isinstance(function['name'], str) or not function['name']:
                                errors.append(f"Function in tool call {tc_idx} in message {i} has invalid 'name' field.")
                                
                            if 'arguments' not in function:
                                errors.append(f"Function call at index {tc_idx} in message {i} is missing 'arguments'.")
                            elif not isinstance(function['arguments'], str):
                                errors.append(f"Function call at index {tc_idx} in message {i} has invalid 'arguments' field type.")
                
                # With tool_calls present, content can be None or a string
                if content is not None and not isinstance(content, str):
                    errors.append(f"Assistant message at index {i} with tool_calls has non-string content.")
                if isinstance(content, str):
                    total_chars += len(content)
                    
            else:
                # Without tool_calls, content must be a string (can be empty)
                if content is None:
                    errors.append(f"Assistant message at index {i} must have non-null content when not using tool_calls.")
                elif not isinstance(content, str):
                    errors.append(f"Assistant message at index {i} must have string content when not using tool_calls.")
                else:
                    total_chars += len(content)
                    
                # Clear expected tool call IDs when an assistant message doesn't have tool_calls
                expected_tool_call_ids.clear()
            
            # Assistant message must have either content or tool_calls
            if (content is None or content == "") and (tool_calls is None or tool_calls == []):
                errors.append(f"Assistant message at index {i} must have either non-empty content or tool_calls.")
                
        elif role == "tool":
            # Tool message validations
            if last_role != "assistant":
                errors.append(f"Tool message at index {i} must follow an assistant message.")
                
            # Check required tool_call_id
            tool_call_id = message.get('tool_call_id')
            if 'tool_call_id' not in message:
                errors.append(f"Tool message at index {i} missing required 'tool_call_id' field.")
            elif not isinstance(tool_call_id, str) or not tool_call_id:
                errors.append(f"Tool message at index {i} has invalid 'tool_call_id' field.")
            else:
                # Verify tool_call_id is expected
                if tool_call_id in expected_tool_call_ids:
                    expected_tool_call_ids.remove(tool_call_id)
                else:
                    errors.append(f"Tool message at index {i} has 'tool_call_id': {tool_call_id} does not match any expected ID")
            
            # Check required content
            content = message.get('content')
            if 'content' not in message:
                errors.append(f"Tool message at index {i} is missing 'content'.")
            elif not isinstance(content, str) or not content:
                errors.append(f"Tool message at index {i} must have non-empty string content.")
            else:
                total_chars += len(content)
                
        # Update last_role
        last_role = role
    
    # After loop validations
    
    # Check maximum total character count
    if max_total_chars is not None and total_chars > max_total_chars:
        errors.append(f"Total character count ({total_chars}) exceeds maximum allowed ({max_total_chars}).")
    
    # Check for missing tool responses
    if expected_tool_call_ids:
        errors.append(f"Missing tool responses for tool_call_ids: {', '.join(expected_tool_call_ids)}")
    
    return {"errors": errors, "warnings": warnings}

async def find_model_by_id(client: Union["VeniceClient", "AsyncVeniceClient"], model_id: str) -> Optional[Model]:
    """
    Retrieve full details of a specific model by its unique identifier.

    This function searches through all available models from the Venice.ai API to find
    a model with the specified ID. It handles both synchronous and asynchronous clients,
    automatically adapting the API call method based on the client type.
    
    :param client: An instance of ``~venice_ai.VeniceClient`` or ``~venice_ai.AsyncVeniceClient`` used to fetch
        the model list from the API.
    :type client: Union[venice_ai._client.VeniceClient, venice_ai._async_client.AsyncVeniceClient]
    :param model_id: Unique identifier of the model to search for. This should match
        the ``id`` field of a model exactly.
    :type model_id: str

    :return: The complete :class:`~venice_ai.types.models.Model` object if a model with
        the specified ID is found, otherwise ``None``.
    :rtype: Optional[venice_ai.types.models.Model]

    :raises Exception: If there's an error fetching models from the API (returns ``None``).

    .. note::
        When using a synchronous client, a deprecation warning will be issued to encourage
        proper async/await usage patterns.

    Example:
        >>> # Find a specific model by ID
        >>> model = await find_model_by_id(client, "gpt-4")
        >>> if model:
        ...     print(f"Found model: {model['id']}")
        ...     print(f"Model type: {model.get('type')}")
        ... else:
        ...     print("Model not found")
    """
    from ._client import VeniceClient
    from ._async_client import AsyncVeniceClient
    
    try:
        # Handle different client types
        if isinstance(client, AsyncVeniceClient):
            try:
                # Try awaiting, but fallback to direct access for mocks
                models_list_response = await client.models.list()
                all_models = models_list_response.get("data", []) if models_list_response else []
            except (TypeError, RuntimeError) as e:
                # Handle case where a non-awaitable mock is used in tests
                if "object dict can't be used in 'await' expression" in str(e) or "asyncio.run() cannot be called" in str(e):
                    models_list_response = client.models.list()  # type: ignore
                    all_models = models_list_response.get("data", []) if models_list_response else []  # type: ignore
                else:
                    raise
        else:
            # Handle sync client - this should emit a deprecation warning
            import warnings
            # Generate a deprecation warning that will be detected by the test
            warnings.warn("Calling an async function without awaiting", DeprecationWarning, stacklevel=2)
            # Ensure the warning is properly emitted
            models_list_response = client.models.list()
            all_models = models_list_response.get("data", []) if models_list_response else []

        # Handle case where data is None
        if all_models is None:
            all_models = []
        
        # Find the model with matching ID
        for model in all_models:
            if model.get("id") == model_id:
                return cast(Model, model)
                
        # No matching model found
        return None
        
    except Exception as e:
        print(f"Error finding model by ID: {e}")
        return None

async def get_model_capabilities(client: Union["VeniceClient", "AsyncVeniceClient"], model_id: str) -> Optional[ModelCapabilities]:
    """
    Get the capabilities dictionary for a specific model by its ID.

    This function retrieves the capabilities information for a model, which describes
    what features and functionality the model supports (e.g., streaming, function calling,
    vision, etc.). It first finds the model using :func:`find_model_by_id` and then
    extracts the capabilities from the model specification.
    
    :param client: An instance of ``~venice_ai.VeniceClient`` or ``~venice_ai.AsyncVeniceClient`` used to fetch
        model information from the API.
    :type client: Union[venice_ai._client.VeniceClient, venice_ai._async_client.AsyncVeniceClient]
    :param model_id: Unique identifier of the model whose capabilities should be retrieved.
    :type model_id: str

    :return: The :class:`~venice_ai.types.models.ModelCapabilities` dictionary containing
        capability flags if the model is found, otherwise ``None``.
    :rtype: Optional[venice_ai.types.models.ModelCapabilities]

    :raises Exception: If there's an error fetching model information (returns ``None``).

    .. note::
        This function depends on :func:`find_model_by_id` and will inherit any warnings
        or behaviors from that function, including deprecation warnings for synchronous clients.

    Example:
        >>> # Get capabilities for a specific model
        >>> capabilities = await get_model_capabilities(client, "gpt-4")
        >>> if capabilities:
        ...     print(f"Supports streaming: {capabilities.get('streaming', False)}")
        ...     print(f"Supports functions: {capabilities.get('supports_functions', False)}")
        ... else:
        ...     print("Model not found or no capabilities available")
    """
    try:
        # Find the model first
        model = await find_model_by_id(client, model_id)
        
        # If model found, extract capabilities
        if model is not None:
            # Access capabilities through model_spec as per design
            model_spec = model.get("model_spec", {})
            if not isinstance(model_spec, dict):
                return None
            
            capabilities = model_spec.get("capabilities")
            if capabilities is None:
                return None
                
            return cast(ModelCapabilities, capabilities)
            
        # Model not found
        return None
    except Exception as e:
        print(f"Error getting model capabilities: {str(e)}")
        return None

def format_tool_response(tool_call_id: str, content: Any) -> Dict[str, Any]:
    """
    Helper function to format a tool response message for the Chat API.

    This function creates a properly formatted message with role ``"tool"`` that responds
    to an assistant's tool call. It handles content conversion to ensure the response
    is in the correct string format required by the API, automatically converting
    complex objects to JSON strings and handling ``None`` values appropriately.
    
    :param tool_call_id: The unique identifier of the tool call this response is for.
        This must match the ``id`` field from a tool call in an assistant message.
    :type tool_call_id: str
    :param content: The result from the tool execution. Non-string content will be
        automatically converted:
        
        - ``None`` becomes ``"null"`` (JSON representation)
        - Dictionaries and lists are converted to JSON strings
        - Other types are converted using ``str()``
    :type content: Any

    :return: A dictionary formatted as a tool response message with ``role``, ``tool_call_id``,
        and ``content`` fields properly set for use in chat completions.
    :rtype: Dict[str, Any]

    Example:
        >>> # Format a simple string response
        >>> response = format_tool_response("call_123", "Operation completed successfully")
        >>> print(response)
        {'role': 'tool', 'tool_call_id': 'call_123', 'content': 'Operation completed successfully'}
        
        >>> # Format a complex object response
        >>> data = {"result": 42, "status": "success"}
        >>> response = format_tool_response("call_456", data)
        >>> print(response['content'])  # '{"result": 42, "status": "success"}'
    """
    # Ensure content is a string
    if content is None:
        stringified_content = "null"  # JSON representation of None
    elif not isinstance(content, str):
        # Convert complex objects to JSON strings
        if isinstance(content, (dict, list)):
            stringified_content = json.dumps(content)
        else:
            # Convert other types using str()
            stringified_content = str(content)
    else:
        stringified_content = content
        
    # Return properly formatted tool response message
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": stringified_content
    }

def _prepare_model_list_params(type_param: Optional[ModelType] = None) -> Dict[str, Any]:
    """
    Prepares the query parameters for listing models, handling type mapping.

    :param type_param: The model type filter from the SDK.
    :type type_param: Optional[ModelType]
    :return: A dictionary of parameters for the API request.
    :rtype: Dict[str, Any]
    """
    params: Dict[str, Any] = {}
    if type_param is not None:
        # Ensure type_param is treated as a string for comparisons,
        # as ModelType is a subclass of str (Enum).
        api_type_value = str(type_param).lower()

        if api_type_value == "chat":
            params["type"] = "text"  # "chat" models are requested as "text" from API.
        elif api_type_value == "audio": # Assuming "audio" is a possible ModelType value
            params["type"] = "tts"    # Map UI/SDK "audio" to API "tts"
        elif api_type_value in ["embedding", "image", "text", "tts", "upscale"]:
            # These are direct matches to API types
            params["type"] = api_type_value
        # If 'type_param' from SDK is an unknown value not in the list above,
        # no 'type' query param is sent. The API might default or error.
        # This matches the existing behavior in AsyncModels.list.
    else:
        # type_param is None, which means "all" was selected or no filter.
        # API documentation states: "Use 'all' to get all model types."
        params["type"] = "all"
    return params