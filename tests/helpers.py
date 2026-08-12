"""Test helper functions moved from src/venice_ai/utils.py.

These functions have zero production callers and exist solely for test support.
"""

from __future__ import annotations

import importlib.util
from types import ModuleType as Module
from typing import Any

from venice_ai.types.api.models import ModelResponse as ModelDetails


def truncate_string(s: str | None, max_len: int) -> str | None:
    """
    Truncate a string to a maximum length, appending ellipsis if needed.

    This utility function safely truncates strings that exceed a specified
    maximum length, appending '...' to indicate truncation. It handles None
    values gracefully and ensures the total length never exceeds max_len.

    Args:
        s: The string to truncate, or None
        max_len: Maximum allowed length including ellipsis

    Returns:
        Truncated string with '...' if truncation occurred, original string
        if within limit, or None if input was None

    Example:
        >>> truncate_string("This is a long string", 10)
        'This is...'
        >>> truncate_string("Short", 10)
        'Short'
        >>> truncate_string(None, 10)
        None
    """
    if s is None:
        return None
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


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
        if spec.loader is None or not hasattr(spec.loader, "exec_module"):
            raise ImportError(f"Spec loader is not a valid Loader for module {module_name}")
        spec.loader.exec_module(module)
    except ImportError:
        # Re-raise ImportError from exec_module as expected by tests
        raise
    return module


def get_models_by_capability(models: list[ModelDetails], capability: str) -> list[ModelDetails]:
    """
    Filters a list of models by a specific capability.

    Uses camelCase capability names (snake_case not supported).
    For example, use "supportsFunctionCalling", "supportsReasoning", etc.

    :param models: A list of model objects to filter.
    :type models: List[Model]
    :param capability: The capability to filter by (e.g., "supportsReasoning").
    :type capability: str
    :return: A new list of models that have the specified capability.
    :rtype: List[Model]
    """
    filtered_models = []
    for model in models:
        # Access model_spec (same pattern as _apply_model_filters)
        model_spec = model.model_spec if hasattr(model, "model_spec") else None

        if model_spec is None:
            continue

        # Handle both Pydantic model and dict formats
        if hasattr(model_spec, "model_dump"):
            model_spec_dict = model_spec.model_dump()
        elif isinstance(model_spec, dict):
            model_spec_dict = model_spec
        else:
            continue

        # Get capabilities from model_spec (camelCase only)
        capabilities_obj = model_spec_dict.get("capabilities", {})
        if isinstance(capabilities_obj, dict):
            capabilities = capabilities_obj
        elif hasattr(capabilities_obj, "model_dump"):
            capabilities = capabilities_obj.model_dump()
        else:
            capabilities = {}

        if not isinstance(capabilities, dict):
            continue

        # Check for camelCase capability only
        if capabilities.get(capability):
            filtered_models.append(model)

    return filtered_models


def _prepare_model_list_params(type_param: str | None = None) -> dict[str, Any]:
    """
    Prepares the query parameters for listing models, handling type mapping.

    :param type_param: The model type filter from the SDK.
    :type type_param: Optional[ModelType]
    :return: A dictionary of parameters for the API request.
    :rtype: Dict[str, Any]
    """
    params: dict[str, Any] = {}
    if type_param is not None:
        # Ensure type_param is treated as a string for comparisons,
        # as ModelType is a subclass of str (Enum).
        api_type_value = str(type_param).lower()

        if api_type_value == "chat":
            params["type"] = "text"  # "chat" models are requested as "text" from API.
        elif api_type_value == "audio":  # Assuming "audio" is a possible ModelType value
            params["type"] = "tts"  # Map UI/SDK "audio" to API "tts"
        elif api_type_value in ["embedding", "image", "text", "tts", "upscale"]:
            # These are direct matches to API types
            params["type"] = api_type_value
        # If 'type_param' from SDK is an unknown value not in the list above,
        # no 'type' query param is sent. The API might default or error.
        # This matches the existing behavior in Models.list.
    else:
        # type_param is None, which means "all" was selected or no filter.
        # API documentation states: "Use 'all' to get all model types."
        params["type"] = "all"
    return params
