"""
Comprehensive unit tests for venice_ai.utils module.

This test suite aims for >90% line coverage and >80% branch coverage by testing:
- All utility functions with various input types and edge cases
- Error conditions and exception handling
- Both successful and failure paths
- Boundary conditions and edge cases
"""

from typing import Any
from unittest.mock import Mock, patch

import pytest

from tests.helpers import (
    _prepare_model_list_params,
    get_models_by_capability,
    import_module_from_path,
    truncate_string,
)
from venice_ai.types.api.models import ModelResponse

# Import the functions we're testing
from venice_ai.utils import (
    NOT_GIVEN,
    NotGivenType,
    _apply_model_filters,
    get_filtered_models,
)


class TestTruncateString:
    """Test string truncation utility."""

    def test_truncate_string_none_input(self):
        """Test truncation with None input."""
        assert truncate_string(None, 10) is None

    def test_truncate_string_short_string(self):
        """Test truncation with string shorter than max length."""
        result = truncate_string("hello", 10)
        assert result == "hello"

    def test_truncate_string_exact_length(self):
        """Test truncation with string exactly at max length."""
        result = truncate_string("hello", 5)
        assert result == "hello"

    def test_truncate_string_long_string(self):
        """Test truncation with string longer than max length."""
        result = truncate_string("hello world", 8)
        assert result == "hello..."

    def test_truncate_string_empty_string(self):
        """Test truncation with empty string."""
        result = truncate_string("", 10)
        assert result == ""

    def test_truncate_string_max_len_zero(self):
        """Test truncation with max_len of 0."""
        result = truncate_string("hello", 0)
        assert result == "he..."

    def test_truncate_string_max_len_negative(self):
        """Test truncation with negative max_len."""
        result = truncate_string("hello", -1)
        assert result == "h..."


class TestNotGivenType:
    """Test the NotGiven sentinel type."""

    def test_not_given_repr(self):
        """Test the string representation of NOT_GIVEN."""
        assert repr(NOT_GIVEN) == "NOT_GIVEN"

    def test_not_given_is_instance(self):
        """Test that NOT_GIVEN is instance of NotGivenType."""
        assert isinstance(NOT_GIVEN, NotGivenType)

    def test_not_given_singleton(self):
        """Test that NOT_GIVEN behaves as singleton."""
        other_instance = NotGivenType()
        assert repr(other_instance) == "NOT_GIVEN"


class TestImportModuleFromPath:
    """Test dynamic module import utility."""

    def test_import_module_from_path_success(self, tmp_path):
        """Test successful module import."""
        # Create a temporary Python file
        module_file = tmp_path / "test_module.py"
        module_file.write_text("def test_function():\n    return 'success'")

        # Import the module
        module = import_module_from_path("test_module", str(module_file))

        # Test that the module was imported correctly
        assert hasattr(module, "test_function")
        assert module.test_function() == "success"

    def test_import_module_from_path_file_not_found(self):
        """Test import with non-existent file."""
        with pytest.raises(FileNotFoundError):
            import_module_from_path("nonexistent", "/nonexistent/path.py")

    def test_import_module_from_path_invalid_python(self, tmp_path):
        """Test import with invalid Python syntax."""
        # Create a file with invalid Python syntax
        module_file = tmp_path / "invalid_module.py"
        module_file.write_text("def invalid_syntax(:\n    pass")

        with pytest.raises(SyntaxError):
            import_module_from_path("invalid_module", str(module_file))

    @patch("importlib.util.spec_from_file_location")
    def test_import_module_from_path_spec_none(self, mock_spec_from_file):
        """Test when spec_from_file_location returns None."""
        mock_spec_from_file.return_value = None

        with pytest.raises(ImportError, match="Could not load spec for module"):
            import_module_from_path("test", "test.py")

    @patch("importlib.util.module_from_spec")
    def test_import_module_from_path_module_none(self, mock_module_from_spec):
        """Test when module_from_spec returns None."""
        mock_module_from_spec.return_value = None

        with pytest.raises(ImportError, match="Could not create module"):
            import_module_from_path("test", "test.py")

    def test_import_module_from_path_no_loader(self):
        """Test when spec has no loader."""
        with (
            patch("importlib.util.spec_from_file_location") as mock_spec,
            patch("importlib.util.module_from_spec") as mock_module_from_spec,
        ):
            mock_spec_obj = Mock()
            mock_spec_obj.loader = None
            mock_spec_obj.name = "test"  # Must be string, not Mock
            mock_spec.return_value = mock_spec_obj
            mock_module_from_spec.return_value = Mock()  # Create a mock module

            with pytest.raises(ImportError, match="Spec loader is not a valid Loader"):
                import_module_from_path("test", "test.py")

    def test_import_module_from_path_loader_no_exec_module(self):
        """Test when loader doesn't have exec_module method."""
        with patch("importlib.util.spec_from_file_location") as mock_spec:
            mock_loader = Mock()
            del mock_loader.exec_module  # Remove the exec_module attribute
            mock_spec.return_value = Mock(loader=mock_loader)

            with pytest.raises(ImportError, match="Spec loader is not a valid Loader"):
                import_module_from_path("test", "test.py")


class TestGetModelsByCapability:
    """Test filtering models by capability."""

    def create_test_model(self, model_id: str, capabilities: dict[str, Any]) -> Mock:
        """Helper to create test model."""
        model = Mock(spec=ModelResponse)
        model.model_spec = Mock()
        # model_dump() needs to return dict with capabilities dict
        model.model_spec.model_dump = Mock(return_value={"capabilities": capabilities})
        return model

    def test_get_models_by_capability_success(self):
        """Test filtering by existing capability."""
        models = [
            self.create_test_model("model1", {"streaming": True}),
            self.create_test_model("model2", {"streaming": False}),
            self.create_test_model("model3", {"streaming": True}),
        ]

        result = get_models_by_capability(models, "streaming")  # type: ignore
        assert len(result) == 2

    def test_get_models_by_capability_supports_functions(self):
        """Test filtering by supportsFunctionCalling capability (v2.0.0 camelCase only)."""
        models = [
            self.create_test_model("model1", {"supportsFunctionCalling": True}),
            self.create_test_model("model2", {"supportsFunctionCalling": True}),
            self.create_test_model("model3", {"streaming": True}),
        ]

        result = get_models_by_capability(models, "supportsFunctionCalling")  # type: ignore
        assert len(result) == 2

    def test_get_models_by_capability_pydantic_model(self):
        """Test with Pydantic model with model_spec."""
        mock_model = Mock()
        mock_model.model_spec = Mock()
        # model_dump() needs to return dict with capabilities dict
        mock_model.model_spec.model_dump = Mock(return_value={"capabilities": {"streaming": True}})

        result = get_models_by_capability([mock_model], "streaming")  # type: ignore
        assert len(result) == 1

    def test_get_models_by_capability_no_model_spec(self):
        """Test with model that has no model_spec."""
        mock_model = Mock()
        mock_model.model_spec = None

        result = get_models_by_capability([mock_model], "streaming")  # type: ignore
        assert len(result) == 0

    def test_get_models_by_capability_model_spec_no_capabilities(self):
        """Test with model_spec that has no capabilities."""
        mock_model = Mock()
        mock_model.model_spec = Mock()
        mock_model.model_spec.capabilities = None

        result = get_models_by_capability([mock_model], "streaming")  # type: ignore
        assert len(result) == 0


class TestPrepareModelListParams:
    """Test model list parameter preparation."""

    def test_prepare_model_list_params_none(self):
        """Test with None type parameter."""
        result = _prepare_model_list_params(None)
        assert result == {"type": "all"}

    def test_prepare_model_list_params_chat(self):
        """Test with chat type."""
        result = _prepare_model_list_params("chat")
        assert result == {"type": "text"}

    def test_prepare_model_list_params_audio(self):
        """Test with audio type."""
        result = _prepare_model_list_params("audio")
        assert result == {"type": "tts"}

    def test_prepare_model_list_params_direct_match(self):
        """Test with direct match types."""
        for model_type in ["embedding", "image", "text", "tts", "upscale"]:
            result = _prepare_model_list_params(model_type)
            assert result == {"type": model_type}

    def test_prepare_model_list_params_unknown(self):
        """Test with unknown type."""
        result = _prepare_model_list_params("unknown")
        assert result == {}


class TestGetFilteredModels:
    """Test filtered model retrieval."""

    def create_test_model_response(self, model_id: str, model_spec_data: dict[str, Any]) -> Mock:
        """Helper to create test ModelResponse that mimics Pydantic model behavior."""
        model = Mock()
        model.model_spec = Mock()

        # Make model_spec.model_dump() return a proper dict
        model.model_spec.model_dump = Mock(return_value=model_spec_data)

        # Also set attributes directly for backward compatibility
        for key, value in model_spec_data.items():
            setattr(model.model_spec, key, value)

        return model

    def test_get_filtered_models_by_type(self):
        """Test filtering by model type."""
        # Create test models with proper structure - capabilities must be dict, not Mock
        # Note: type attribute is on the model itself, not model_spec
        model1 = Mock()
        model1.type = "text"  # type is on model, not model_spec
        model1.model_spec = Mock()
        model1.model_spec.model_dump = Mock(return_value={"type": "text", "capabilities": {}})

        model2 = Mock()
        model2.type = "image"  # type is on model, not model_spec
        model2.model_spec = Mock()
        model2.model_spec.model_dump = Mock(return_value={"type": "image", "capabilities": {}})

        model3 = Mock()
        model3.type = "text"  # type is on model, not model_spec
        model3.model_spec = Mock()
        model3.model_spec.model_dump = Mock(return_value={"type": "text", "capabilities": {}})

        models = [model1, model2, model3]

        result = get_filtered_models(models, model_type="text")
        assert len(result) == 2

    def test_get_filtered_models_by_vision(self):
        """Test filtering by vision support."""
        models = [
            self.create_test_model_response("model1", {"capabilities": {"supportsVision": True}}),
            self.create_test_model_response("model2", {"capabilities": {"supportsVision": False}}),
            self.create_test_model_response("model3", {"capabilities": {"supportsVision": False}}),
        ]

        result = get_filtered_models(models, supports_vision=True)
        assert len(result) == 1

    def test_get_filtered_models_by_function_calling(self):
        """Test filtering by function calling support (v2.0.0 camelCase only)."""
        models = [
            self.create_test_model_response(
                "model1", {"capabilities": {"supportsFunctionCalling": True}}
            ),
            self.create_test_model_response(
                "model2", {"capabilities": {"supportsFunctionCalling": True}}
            ),
            self.create_test_model_response(
                "model3", {"capabilities": {"supportsFunctionCalling": False}}
            ),
        ]

        result = get_filtered_models(models, supports_function_calling=True)
        assert len(result) == 2

    def test_get_filtered_models_multiple_filters(self):
        """Test filtering with multiple criteria."""
        # Note: type must be on model itself, not in model_spec
        model1 = Mock()
        model1.type = "text"  # type is on model, not model_spec
        model1.model_spec = Mock()
        model1.model_spec.model_dump = Mock(
            return_value={"capabilities": {"supportsVision": True, "supportsReasoning": True}}
        )

        model2 = Mock()
        model2.type = "text"  # type is on model, not model_spec
        model2.model_spec = Mock()
        model2.model_spec.model_dump = Mock(
            return_value={"capabilities": {"supportsVision": False, "supportsReasoning": True}}
        )

        model3 = Mock()
        model3.type = "image"  # type is on model, not model_spec
        model3.model_spec = Mock()
        model3.model_spec.model_dump = Mock(
            return_value={"capabilities": {"supportsVision": True, "supportsReasoning": False}}
        )

        models = [model1, model2, model3]

        result = get_filtered_models(
            models, model_type="text", supports_vision=True, supports_reasoning=True
        )
        assert len(result) == 1

    def test_get_filtered_models_beta_filter(self):
        """Test filtering by beta status."""
        models = [
            self.create_test_model_response("model1", {"beta": True, "capabilities": {}}),
            self.create_test_model_response("model2", {"beta": False, "capabilities": {}}),
            self.create_test_model_response("model3", {"capabilities": {}}),  # No beta field
        ]

        result = get_filtered_models(models, is_beta=True)
        assert len(result) == 1

        result = get_filtered_models(models, is_beta=False)
        assert len(result) == 2  # model2 and model3 (False is default)

    def test_get_filtered_models_traits_filter(self):
        """Test filtering by traits."""
        models = [
            self.create_test_model_response(
                "model1", {"traits": ["fast", "accurate"], "capabilities": {}}
            ),
            self.create_test_model_response(
                "model2", {"traits": ["slow", "accurate"], "capabilities": {}}
            ),
            self.create_test_model_response("model3", {"traits": [], "capabilities": {}}),
        ]

        result = get_filtered_models(models, has_trait="accurate")
        assert len(result) == 2

        result = get_filtered_models(models, has_trait="nonexistent")
        assert len(result) == 0


# =============================================================================
# Extended coverage tests (merged from test_utils_coverage_expansion.py)
# =============================================================================


class TestGetModelsByCapabilityExtended:
    """Extended tests for get_models_by_capability dict/non-dict paths."""

    def test_model_spec_as_dict(self):
        """Test when model_spec is a dict rather than Pydantic model."""
        mock_model = Mock()
        del mock_model.model_spec
        mock_model.model_spec = {"capabilities": {"supportsReasoning": True}}

        result = get_models_by_capability([mock_model], "supportsReasoning")
        assert len(result) == 1

    def test_model_spec_neither_pydantic_nor_dict(self):
        """Test when model_spec is neither Pydantic model nor dict."""
        mock_model = Mock()
        mock_model.model_spec = "invalid_type"

        result = get_models_by_capability([mock_model], "supportsReasoning")
        assert len(result) == 0

    def test_capabilities_neither_dict_nor_pydantic(self):
        """Test when capabilities is neither dict nor has model_dump."""
        mock_model = Mock()
        mock_model.model_spec = Mock()
        mock_model.model_spec.model_dump = Mock(
            return_value={"capabilities": "not_a_dict_or_pydantic"}
        )

        result = get_models_by_capability([mock_model], "supportsReasoning")
        assert len(result) == 0

    def test_capabilities_as_pydantic_model(self):
        """Test when capabilities itself is a Pydantic model with model_dump."""
        mock_model = Mock()
        mock_model.model_spec = Mock()

        mock_capabilities = Mock()
        mock_capabilities.model_dump = Mock(return_value={"supportsReasoning": True})

        mock_model.model_spec.model_dump = Mock(return_value={"capabilities": mock_capabilities})

        result = get_models_by_capability([mock_model], "supportsReasoning")
        assert len(result) == 1


class TestApplyModelFiltersExtended:
    """Extended tests for _apply_model_filters dict/non-dict paths."""

    def test_model_spec_as_dict_in_filters(self):
        """Test when model_spec is a dict."""
        mock_model = Mock()
        mock_model.type = "text"
        mock_model.model_spec = {
            "capabilities": {"supportsVision": True},
            "beta": False,
        }

        result = _apply_model_filters([mock_model], supports_vision=True)
        assert len(result) == 1

    def test_model_spec_neither_type_in_filters(self):
        """Test when model_spec is neither Pydantic nor dict."""
        mock_model = Mock()
        mock_model.type = "text"
        mock_model.model_spec = "invalid_string"

        result = _apply_model_filters([mock_model])
        assert len(result) == 0

    def test_capabilities_as_pydantic_in_filters(self):
        """Test when capabilities has model_dump method."""
        mock_model = Mock()
        mock_model.type = "text"
        mock_model.model_spec = Mock()

        mock_capabilities = Mock()
        mock_capabilities.model_dump = Mock(return_value={"supportsVision": True})

        mock_model.model_spec.model_dump = Mock(return_value={"capabilities": mock_capabilities})

        result = _apply_model_filters([mock_model], supports_vision=True)
        assert len(result) == 1

    def test_capabilities_neither_type_in_filters(self):
        """Test when capabilities is neither dict nor has model_dump."""
        mock_model = Mock()
        mock_model.type = "text"
        mock_model.model_spec = Mock()
        mock_model.model_spec.model_dump = Mock(return_value={"capabilities": 12345})

        result = _apply_model_filters([mock_model], supports_vision=True)
        assert len(result) == 0

    def test_capabilities_not_dict_after_processing(self):
        """Test when capabilities result is not a dict after model_dump."""
        mock_model = Mock()
        mock_model.type = "text"
        mock_model.model_spec = Mock()

        mock_capabilities = Mock()
        mock_capabilities.model_dump = Mock(return_value="not_a_dict")

        mock_model.model_spec.model_dump = Mock(return_value={"capabilities": mock_capabilities})

        result = _apply_model_filters([mock_model])
        assert len(result) == 0
