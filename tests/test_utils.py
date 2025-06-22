import pytest
import warnings
from unittest.mock import patch, MagicMock, AsyncMock
from venice_ai.utils import (
    truncate_string,
    NotGiven,
    NOT_GIVEN,
    _import_tiktoken_module,
    import_module_from_path,
    find_model_by_id_or_name_or_slug,
    get_model_capabilities_by_id_or_name_or_slug,
    get_models_by_capability,
    get_filtered_models,
    estimate_token_count,
    validate_chat_messages,
    find_model_by_id,
    get_model_capabilities,
    format_tool_response,
    _prepare_model_list_params,
)
from venice_ai.types.models import Model, ModelType
from typing import cast, List

def test_truncate_string():
    """Test truncate_string function."""
    assert truncate_string("hello", 10) == "hello"
    assert truncate_string("hello world", 10) == "hello w..."
    assert truncate_string("hello world", 5) == "he..."
    assert truncate_string(None, 10) is None

def test_not_given():
    """Test NotGiven sentinel type."""
    assert str(NOT_GIVEN) == "NOT_GIVEN"
    assert repr(NOT_GIVEN) == "NOT_GIVEN"

@patch("venice_ai.utils._TIKTOKEN_AVAILABLE", True)
def test_import_tiktoken_module_available():
    """Test _import_tiktoken_module when tiktoken is available."""
    with patch("venice_ai.utils.tiktoken") as mock_tiktoken:
        assert _import_tiktoken_module() == mock_tiktoken

@patch("venice_ai.utils._TIKTOKEN_AVAILABLE", False)
def test_import_tiktoken_module_not_available():
    """Test _import_tiktoken_module when tiktoken is not available."""
    with pytest.raises(ImportError):
        _import_tiktoken_module()

def test_import_module_from_path():
    """Test import_module_from_path function."""
    with patch("importlib.util.spec_from_file_location") as mock_spec:
        mock_spec.return_value = None
        with pytest.raises(ImportError):
            import_module_from_path("test_module", "test_path")

def test_find_model_by_id_or_name_or_slug():
    """Test find_model_by_id_or_name_or_slug function."""
    assert find_model_by_id_or_name_or_slug("gpt-4") is not None
    assert find_model_by_id_or_name_or_slug("Claude 3") is not None
    assert find_model_by_id_or_name_or_slug("claude-3") is not None
    assert find_model_by_id_or_name_or_slug("unknown") is None

def test_get_model_capabilities_by_id_or_name_or_slug():
    """Test get_model_capabilities_by_id_or_name_or_slug function."""
    assert get_model_capabilities_by_id_or_name_or_slug("gpt-4") is not None
    assert get_model_capabilities_by_id_or_name_or_slug("unknown") is None

def test_get_models_by_capability():
    """Test get_models_by_capability function."""
    models = [
        {"model_spec": {"capabilities": {"streaming": True}}},
        {"model_spec": {"capabilities": {"streaming": False}}},
    ]
    assert len(get_models_by_capability(cast(List[Model], models), "streaming")) == 1

@pytest.mark.asyncio
async def test_get_filtered_models():
    """Test get_filtered_models function."""
    mock_client = AsyncMock()
    # Make the mock return a dict directly (not awaitable) to trigger the fallback in utils.py
    mock_response = {
        "data": [
            {"type": "text", "model_spec": {"capabilities": {"streaming": True}}},
            {"type": "image", "model_spec": {"capabilities": {"streaming": False}}},
        ]
    }
    # Use MagicMock instead of AsyncMock for models.list to return dict directly
    mock_client.models.list = MagicMock(return_value=mock_response)
    assert len(await get_filtered_models(mock_client, model_type="text")) == 1  # type: ignore
    assert len(await get_filtered_models(mock_client, supports_capabilities=["streaming"])) == 1  # type: ignore

def test_estimate_token_count():
    """Test estimate_token_count function."""
    with patch("venice_ai.utils._TIKTOKEN_AVAILABLE", False):
        # Suppress the expected UserWarning about tiktoken not being available
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            # "hello world" has 11 characters, 11/4 = 2.75, which rounds down to 2
            assert estimate_token_count("hello world") == 2
    with pytest.raises(AttributeError):
        estimate_token_count(None)  # type: ignore

def test_validate_chat_messages():
    """Test validate_chat_messages function."""
    with pytest.raises(AttributeError):
        validate_chat_messages(None)  # type: ignore
    # The actual error message is "Messages list cannot be empty."
    assert "cannot be empty" in validate_chat_messages([])["errors"][0]

@pytest.mark.asyncio
async def test_find_model_by_id():
    """Test find_model_by_id function."""
    mock_client = AsyncMock()
    # Make the mock return a dict directly (not awaitable) to trigger the fallback in utils.py
    mock_response = {"data": [{"id": "gpt-4"}]}
    # Use MagicMock instead of AsyncMock for models.list to return dict directly
    mock_client.models.list = MagicMock(return_value=mock_response)
    
    # Suppress the expected DeprecationWarning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        assert await find_model_by_id(mock_client, "gpt-4") is not None

@pytest.mark.asyncio
async def test_get_model_capabilities():
    """Test get_model_capabilities function."""
    mock_client = AsyncMock()
    # Make the mock return a dict directly (not awaitable) to trigger the fallback in utils.py
    mock_response = {
        "data": [{"id": "gpt-4", "model_spec": {"capabilities": {"streaming": True}}}]
    }
    # Use MagicMock instead of AsyncMock for models.list to return dict directly
    mock_client.models.list = MagicMock(return_value=mock_response)
    
    # Suppress the expected DeprecationWarning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        assert await get_model_capabilities(mock_client, "gpt-4") is not None

def test_format_tool_response():
    """Test format_tool_response function."""
    assert format_tool_response("123", "test")["content"] == "test"
    assert format_tool_response("123", None)["content"] == "null"
    assert format_tool_response("123", {"a": 1})["content"] == '{"a": 1}'

def test_prepare_model_list_params():
    """Test _prepare_model_list_params function."""
    assert _prepare_model_list_params(cast(ModelType, "text"))["type"] == "text"
    assert _prepare_model_list_params(cast(ModelType, "tts"))["type"] == "tts"
    assert _prepare_model_list_params(cast(ModelType, "image"))["type"] == "image"
    assert _prepare_model_list_params(None)["type"] == "all"