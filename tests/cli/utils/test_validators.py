"""
Comprehensive tests for ParameterValidator in cli/utils/validators.py.

Coverage target: 80%+ line and branch coverage.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from venice_ai.cli.utils.validators import ParameterValidator
from venice_ai.types.api.models import ImageModelConstraints, StepsConstraint

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_client():
    """Create a mock VeniceClient."""
    return MagicMock()


@pytest.fixture
def validator(mock_client):
    """Create a ParameterValidator with mocked client."""
    return ParameterValidator(mock_client)


@pytest.fixture
def image_model_constraints():
    """Create ImageModelConstraints for testing."""
    return ImageModelConstraints(
        promptCharacterLimit=1000,
        steps=StepsConstraint(default=25, max=50),
        widthHeightDivisor=64,
    )


@pytest.fixture
def mock_model_spec(image_model_constraints):
    """Create a mock model spec with constraints."""
    return SimpleNamespace(constraints=image_model_constraints)


@pytest.fixture
def mock_model_response(mock_model_spec):
    """Create a mock model response."""
    return SimpleNamespace(id="test-model", model_spec=mock_model_spec)


@pytest.fixture
def mock_models_response(mock_model_response):
    """Create a mock models list response."""
    return SimpleNamespace(data=[mock_model_response])


# ============================================================================
# Test Class: ParameterValidator Initialization
# ============================================================================


class TestParameterValidatorInit:
    """Tests for ParameterValidator.__init__ (lines 37-38)."""

    def test_init_stores_client(self, mock_client):
        """Test that __init__ stores the client reference."""
        # Covers line 37
        validator = ParameterValidator(mock_client)
        assert validator.client is mock_client

    def test_init_creates_empty_cache(self, mock_client):
        """Test that __init__ creates an empty model specs cache."""
        # Covers line 38
        validator = ParameterValidator(mock_client)
        assert validator._model_specs_cache == {}
        assert isinstance(validator._model_specs_cache, dict)


# ============================================================================
# Test Class: _get_model_spec
# ============================================================================


class TestGetModelSpec:
    """Tests for ParameterValidator._get_model_spec (lines 51-69)."""

    @pytest.mark.asyncio
    async def test_returns_cached_spec(self, validator, mock_model_spec):
        """Test that cached model specs are returned without API call."""
        # Covers line 51-52 (cache hit branch)
        validator._model_specs_cache["cached-model"] = mock_model_spec

        result = await validator._get_model_spec("cached-model")

        assert result is mock_model_spec
        # Client.models.list should not be called
        validator.client.models.list.assert_not_called()

    @pytest.mark.asyncio
    async def test_fetches_and_caches_spec(self, validator, mock_models_response, mock_model_spec):
        """Test that model specs are fetched and cached."""
        # Covers lines 54, 56, 58-59, 61-62
        validator.client.models.list = AsyncMock(return_value=mock_models_response)

        result = await validator._get_model_spec("test-model")

        assert result is mock_model_spec
        assert "test-model" in validator._model_specs_cache
        assert validator._model_specs_cache["test-model"] is mock_model_spec
        validator.client.models.list.assert_called_once_with(type="image")

    @pytest.mark.asyncio
    async def test_returns_none_when_model_not_found(self, validator):
        """Test that None is returned when model is not in the list."""
        # Covers lines 64-65 (model not found branch)
        other_model = SimpleNamespace(id="other-model", model_spec=SimpleNamespace())
        mock_response = SimpleNamespace(data=[other_model])
        validator.client.models.list = AsyncMock(return_value=mock_response)

        result = await validator._get_model_spec("nonexistent-model")

        assert result is None
        # Cache should not have the model
        assert "nonexistent-model" not in validator._model_specs_cache

    @pytest.mark.asyncio
    async def test_returns_none_on_exception(self, validator):
        """Test that None is returned when API call fails."""
        # Covers lines 67-69 (exception branch)
        validator.client.models.list = AsyncMock(side_effect=Exception("API Error"))

        result = await validator._get_model_spec("any-model")

        assert result is None


# ============================================================================
# Test Class: validate_image_parameters
# ============================================================================


class TestValidateImageParameters:
    """Tests for ParameterValidator.validate_image_parameters (lines 106-187)."""

    @pytest.mark.asyncio
    async def test_valid_params_with_constraints(
        self, validator, mock_model_spec, mock_models_response
    ):
        """Test validation passes with valid params and model constraints."""
        # Covers lines 106, 109-112, 115-116, 135-136, 187
        validator.client.models.list = AsyncMock(return_value=mock_models_response)

        is_valid, error = await validator.validate_image_parameters(
            model="test-model",
            width=1024,  # Divisible by 64
            height=768,  # Divisible by 64
            prompt="A valid test prompt",
        )

        assert is_valid is True
        assert error is None

    @pytest.mark.asyncio
    async def test_width_not_divisible_by_constraint_divisor(self, validator, mock_models_response):
        """Test validation fails when width not divisible by constraint divisor."""
        # Covers lines 117-118
        validator.client.models.list = AsyncMock(return_value=mock_models_response)

        is_valid, error = await validator.validate_image_parameters(
            model="test-model",
            width=1025,  # Not divisible by 64
            height=768,
            prompt="Test prompt",
        )

        assert is_valid is False
        assert "Width must be divisible by 64" in error
        assert "1025" in error

    @pytest.mark.asyncio
    async def test_height_not_divisible_by_constraint_divisor(
        self, validator, mock_models_response
    ):
        """Test validation fails when height not divisible by constraint divisor."""
        # Covers lines 119-120
        validator.client.models.list = AsyncMock(return_value=mock_models_response)

        is_valid, error = await validator.validate_image_parameters(
            model="test-model",
            width=1024,
            height=769,  # Not divisible by 64
            prompt="Test prompt",
        )

        assert is_valid is False
        assert "Height must be divisible by 64" in error
        assert "769" in error

    @pytest.mark.asyncio
    async def test_width_not_divisible_by_8_without_constraints(self, validator):
        """Test validation fails when width not divisible by 8 (no constraints)."""
        # Covers lines 122-124 (no constraints, default validation)
        validator.client.models.list = AsyncMock(return_value=SimpleNamespace(data=[]))

        is_valid, error = await validator.validate_image_parameters(
            model="unknown-model",
            width=1025,  # Not divisible by 8
            height=1024,
            prompt="Test prompt",
        )

        assert is_valid is False
        assert "Width must be divisible by 8" in error

    @pytest.mark.asyncio
    async def test_height_not_divisible_by_8_without_constraints(self, validator):
        """Test validation fails when height not divisible by 8 (no constraints)."""
        # Covers lines 125-126
        validator.client.models.list = AsyncMock(return_value=SimpleNamespace(data=[]))

        is_valid, error = await validator.validate_image_parameters(
            model="unknown-model",
            width=1024,
            height=1025,  # Not divisible by 8
            prompt="Test prompt",
        )

        assert is_valid is False
        assert "Height must be divisible by 8" in error

    @pytest.mark.asyncio
    async def test_non_positive_dimensions(self, validator):
        """Test validation fails for non-positive dimensions."""
        # Covers lines 129-130
        validator.client.models.list = AsyncMock(return_value=SimpleNamespace(data=[]))

        is_valid, error = await validator.validate_image_parameters(
            model="model",
            width=0,
            height=512,
            prompt="Test prompt",
        )

        assert is_valid is False
        assert "Width and height must be positive" in error

    @pytest.mark.asyncio
    async def test_negative_dimensions(self, validator):
        """Test validation fails for negative dimensions."""
        # Covers line 129-130
        validator.client.models.list = AsyncMock(return_value=SimpleNamespace(data=[]))

        is_valid, error = await validator.validate_image_parameters(
            model="model",
            width=-512,
            height=512,
            prompt="Test prompt",
        )

        assert is_valid is False
        assert "Width and height must be positive" in error

    @pytest.mark.asyncio
    async def test_dimensions_exceed_max(self, validator):
        """Test validation fails when dimensions exceed 4096."""
        # Covers lines 131-132
        validator.client.models.list = AsyncMock(return_value=SimpleNamespace(data=[]))

        is_valid, error = await validator.validate_image_parameters(
            model="model",
            width=4104,  # > 4096 and divisible by 8
            height=512,
            prompt="Test prompt",
        )

        assert is_valid is False
        assert "must not exceed 4096 pixels" in error

    @pytest.mark.asyncio
    async def test_height_exceeds_max(self, validator):
        """Test validation fails when height exceeds 4096."""
        # Covers lines 131-132
        validator.client.models.list = AsyncMock(return_value=SimpleNamespace(data=[]))

        is_valid, error = await validator.validate_image_parameters(
            model="model",
            width=512,
            height=4104,  # > 4096 and divisible by 8
            prompt="Test prompt",
        )

        assert is_valid is False
        assert "must not exceed 4096 pixels" in error

    @pytest.mark.asyncio
    async def test_prompt_exceeds_model_limit(self, validator, mock_models_response):
        """Test validation fails when prompt exceeds model character limit."""
        # Covers lines 135-141
        validator.client.models.list = AsyncMock(return_value=mock_models_response)

        long_prompt = "x" * 1001  # Exceeds 1000 char limit

        is_valid, error = await validator.validate_image_parameters(
            model="test-model",
            width=1024,
            height=768,
            prompt=long_prompt,
        )

        assert is_valid is False
        assert "Prompt exceeds maximum length" in error
        assert "1000" in error

    @pytest.mark.asyncio
    async def test_empty_prompt(self, validator):
        """Test validation fails for empty prompt."""
        # Covers lines 143-144
        validator.client.models.list = AsyncMock(return_value=SimpleNamespace(data=[]))

        is_valid, error = await validator.validate_image_parameters(
            model="model",
            width=512,
            height=512,
            prompt="",
        )

        assert is_valid is False
        assert "Prompt cannot be empty" in error

    @pytest.mark.asyncio
    async def test_whitespace_only_prompt(self, validator):
        """Test validation fails for whitespace-only prompt."""
        # Covers lines 143-144
        validator.client.models.list = AsyncMock(return_value=SimpleNamespace(data=[]))

        is_valid, error = await validator.validate_image_parameters(
            model="model",
            width=512,
            height=512,
            prompt="   \n\t   ",
        )

        assert is_valid is False
        assert "Prompt cannot be empty" in error

    @pytest.mark.asyncio
    async def test_steps_exceed_model_max(self, validator, mock_models_response):
        """Test validation fails when steps exceed model maximum."""
        # Covers lines 147-154
        validator.client.models.list = AsyncMock(return_value=mock_models_response)

        is_valid, error = await validator.validate_image_parameters(
            model="test-model",
            width=1024,
            height=768,
            prompt="Valid prompt",
            steps=60,  # Exceeds max of 50
        )

        assert is_valid is False
        assert "Steps exceed maximum of 50" in error
        assert "test-model" in error

    @pytest.mark.asyncio
    async def test_steps_less_than_one(self, validator):
        """Test validation fails when steps < 1."""
        # Covers lines 156-157
        validator.client.models.list = AsyncMock(return_value=SimpleNamespace(data=[]))

        is_valid, error = await validator.validate_image_parameters(
            model="model",
            width=512,
            height=512,
            prompt="Valid prompt",
            steps=0,
        )

        assert is_valid is False
        assert "Steps must be at least 1" in error

    @pytest.mark.asyncio
    async def test_steps_exceed_general_limit(self, validator):
        """Test validation fails when steps exceed 150."""
        # Covers lines 158-159
        validator.client.models.list = AsyncMock(return_value=SimpleNamespace(data=[]))

        is_valid, error = await validator.validate_image_parameters(
            model="model",
            width=512,
            height=512,
            prompt="Valid prompt",
            steps=151,
        )

        assert is_valid is False
        assert "Steps must not exceed 150" in error

    @pytest.mark.asyncio
    async def test_negative_cfg_scale(self, validator):
        """Test validation fails for negative CFG scale."""
        # Covers lines 162-164
        validator.client.models.list = AsyncMock(return_value=SimpleNamespace(data=[]))

        is_valid, error = await validator.validate_image_parameters(
            model="model",
            width=512,
            height=512,
            prompt="Valid prompt",
            cfg_scale=-1.0,
        )

        assert is_valid is False
        assert "CFG scale must be non-negative" in error

    @pytest.mark.asyncio
    async def test_cfg_scale_exceeds_max(self, validator):
        """Test validation fails when CFG scale exceeds 20."""
        # Covers lines 165-168
        validator.client.models.list = AsyncMock(return_value=SimpleNamespace(data=[]))

        is_valid, error = await validator.validate_image_parameters(
            model="model",
            width=512,
            height=512,
            prompt="Valid prompt",
            cfg_scale=25.5,
        )

        assert is_valid is False
        assert "CFG scale must not exceed 20" in error

    @pytest.mark.asyncio
    async def test_seed_out_of_range_negative(self, validator):
        """Test validation fails when seed is below minimum."""
        # Covers lines 171-173
        validator.client.models.list = AsyncMock(return_value=SimpleNamespace(data=[]))

        is_valid, error = await validator.validate_image_parameters(
            model="model",
            width=512,
            height=512,
            prompt="Valid prompt",
            seed=-1000000000,
        )

        assert is_valid is False
        assert "Seed must be between -999999999 and 999999999" in error

    @pytest.mark.asyncio
    async def test_seed_out_of_range_positive(self, validator):
        """Test validation fails when seed is above maximum."""
        # Covers lines 171-173
        validator.client.models.list = AsyncMock(return_value=SimpleNamespace(data=[]))

        is_valid, error = await validator.validate_image_parameters(
            model="model",
            width=512,
            height=512,
            prompt="Valid prompt",
            seed=1000000000,
        )

        assert is_valid is False
        assert "Seed must be between -999999999 and 999999999" in error

    @pytest.mark.asyncio
    async def test_lora_strength_below_zero(self, validator):
        """Test validation fails when LoRA strength is below 0."""
        # Covers lines 176-178
        validator.client.models.list = AsyncMock(return_value=SimpleNamespace(data=[]))

        is_valid, error = await validator.validate_image_parameters(
            model="model",
            width=512,
            height=512,
            prompt="Valid prompt",
            lora_strength=-1,
        )

        assert is_valid is False
        assert "LoRA strength must be between 0 and 100" in error

    @pytest.mark.asyncio
    async def test_lora_strength_above_100(self, validator):
        """Test validation fails when LoRA strength is above 100."""
        # Covers lines 176-178
        validator.client.models.list = AsyncMock(return_value=SimpleNamespace(data=[]))

        is_valid, error = await validator.validate_image_parameters(
            model="model",
            width=512,
            height=512,
            prompt="Valid prompt",
            lora_strength=101,
        )

        assert is_valid is False
        assert "LoRA strength must be between 0 and 100" in error

    @pytest.mark.asyncio
    async def test_num_images_less_than_one(self, validator):
        """Test validation fails when num_images < 1."""
        # Covers lines 181-182
        validator.client.models.list = AsyncMock(return_value=SimpleNamespace(data=[]))

        is_valid, error = await validator.validate_image_parameters(
            model="model",
            width=512,
            height=512,
            prompt="Valid prompt",
            num_images=0,
        )

        assert is_valid is False
        assert "Number of images must be at least 1" in error

    @pytest.mark.asyncio
    async def test_num_images_exceeds_max(self, validator):
        """Test validation fails when num_images > 4."""
        # Covers lines 183-184
        validator.client.models.list = AsyncMock(return_value=SimpleNamespace(data=[]))

        is_valid, error = await validator.validate_image_parameters(
            model="model",
            width=512,
            height=512,
            prompt="Valid prompt",
            num_images=5,
        )

        assert is_valid is False
        assert "Number of images must not exceed 4" in error

    @pytest.mark.asyncio
    async def test_all_optional_params_valid(self, validator):
        """Test validation passes with all optional params set to valid values."""
        # Covers line 187 with all params
        validator.client.models.list = AsyncMock(return_value=SimpleNamespace(data=[]))

        is_valid, error = await validator.validate_image_parameters(
            model="model",
            width=512,
            height=512,
            prompt="Valid prompt",
            steps=50,
            cfg_scale=7.5,
            seed=12345,
            lora_strength=50,
            num_images=2,
        )

        assert is_valid is True
        assert error is None


# ============================================================================
# Test Class: validate_dimensions
# ============================================================================


class TestValidateDimensions:
    """Tests for ParameterValidator.validate_dimensions (lines 205-214)."""

    def test_valid_dimensions(self, validator):
        """Test validation passes for valid dimensions."""
        # Covers line 214
        is_valid, error = validator.validate_dimensions(1024, 768)
        assert is_valid is True
        assert error is None

    def test_width_not_divisible_by_divisor(self, validator):
        """Test validation fails when width not divisible by divisor."""
        # Covers lines 205-206
        is_valid, error = validator.validate_dimensions(1025, 768, divisor=8)
        assert is_valid is False
        assert "Width must be divisible by 8" in error
        assert "1025" in error

    def test_height_not_divisible_by_divisor(self, validator):
        """Test validation fails when height not divisible by divisor."""
        # Covers lines 207-208
        is_valid, error = validator.validate_dimensions(1024, 769, divisor=8)
        assert is_valid is False
        assert "Height must be divisible by 8" in error
        assert "769" in error

    def test_non_positive_width(self, validator):
        """Test validation fails for non-positive width."""
        # Covers lines 209-210
        is_valid, error = validator.validate_dimensions(0, 512)
        assert is_valid is False
        assert "Width and height must be positive" in error

    def test_negative_height(self, validator):
        """Test validation fails for negative height."""
        # Covers lines 209-210
        is_valid, error = validator.validate_dimensions(512, -8)
        assert is_valid is False
        assert "Width and height must be positive" in error

    def test_width_exceeds_max(self, validator):
        """Test validation fails when width exceeds 4096."""
        # Covers lines 211-212
        is_valid, error = validator.validate_dimensions(4104, 512)  # 4104 % 8 == 0
        assert is_valid is False
        assert "must not exceed 4096 pixels" in error

    def test_height_exceeds_max(self, validator):
        """Test validation fails when height exceeds 4096."""
        # Covers lines 211-212
        is_valid, error = validator.validate_dimensions(512, 4104)
        assert is_valid is False
        assert "must not exceed 4096 pixels" in error

    def test_custom_divisor(self, validator):
        """Test validation with custom divisor."""
        # Covers custom divisor parameter
        is_valid, error = validator.validate_dimensions(1024, 768, divisor=64)
        assert is_valid is True

        is_valid, error = validator.validate_dimensions(1000, 768, divisor=64)
        assert is_valid is False
        assert "divisible by 64" in error


# ============================================================================
# Test Class: validate_prompt
# ============================================================================


class TestValidatePrompt:
    """Tests for ParameterValidator.validate_prompt (lines 229-238)."""

    def test_valid_prompt(self, validator):
        """Test validation passes for valid prompt."""
        # Covers line 238
        is_valid, error = validator.validate_prompt("A beautiful landscape")
        assert is_valid is True
        assert error is None

    def test_empty_prompt(self, validator):
        """Test validation fails for empty prompt."""
        # Covers lines 229-230
        is_valid, error = validator.validate_prompt("")
        assert is_valid is False
        assert "Prompt cannot be empty" in error

    def test_none_prompt(self, validator):
        """Test validation fails for None prompt (if passed)."""
        # Covers line 229 (falsy check)
        is_valid, error = validator.validate_prompt(None)
        assert is_valid is False
        assert "Prompt cannot be empty" in error

    def test_whitespace_prompt(self, validator):
        """Test validation fails for whitespace-only prompt."""
        # Covers lines 229-230 (strip check)
        is_valid, error = validator.validate_prompt("   \t\n   ")
        assert is_valid is False
        assert "Prompt cannot be empty" in error

    def test_prompt_exceeds_max_length(self, validator):
        """Test validation fails when prompt exceeds max length."""
        # Covers lines 232-236
        long_prompt = "x" * 101
        is_valid, error = validator.validate_prompt(long_prompt, max_length=100)
        assert is_valid is False
        assert "Prompt exceeds maximum length" in error
        assert "100" in error
        assert "101" in error

    def test_prompt_at_max_length(self, validator):
        """Test validation passes when prompt is exactly at max length."""
        # Covers line 232 (boundary)
        prompt = "x" * 100
        is_valid, error = validator.validate_prompt(prompt, max_length=100)
        assert is_valid is True
        assert error is None

    def test_prompt_no_max_length(self, validator):
        """Test validation passes for long prompt when no max specified."""
        # Covers line 232 (max_length None branch)
        long_prompt = "x" * 10000
        is_valid, error = validator.validate_prompt(long_prompt)
        assert is_valid is True
        assert error is None


# ============================================================================
# Test Class: validate_steps
# ============================================================================


class TestValidateSteps:
    """Tests for ParameterValidator.validate_steps (lines 253-262)."""

    def test_valid_steps(self, validator):
        """Test validation passes for valid steps."""
        # Covers line 262
        is_valid, error = validator.validate_steps(25)
        assert is_valid is True
        assert error is None

    def test_steps_less_than_one(self, validator):
        """Test validation fails when steps < 1."""
        # Covers lines 253-254
        is_valid, error = validator.validate_steps(0)
        assert is_valid is False
        assert "Steps must be at least 1" in error

    def test_steps_negative(self, validator):
        """Test validation fails for negative steps."""
        # Covers lines 253-254
        is_valid, error = validator.validate_steps(-5)
        assert is_valid is False
        assert "Steps must be at least 1" in error

    def test_steps_exceed_model_max(self, validator):
        """Test validation fails when steps exceed model max."""
        # Covers lines 256-257
        is_valid, error = validator.validate_steps(60, max_steps=50)
        assert is_valid is False
        assert "Steps exceed maximum of 50" in error
        assert "60" in error

    def test_steps_exceed_general_limit(self, validator):
        """Test validation fails when steps exceed 150."""
        # Covers lines 259-260
        is_valid, error = validator.validate_steps(151)
        assert is_valid is False
        assert "Steps must not exceed 150" in error

    def test_steps_at_general_limit(self, validator):
        """Test validation passes when steps are exactly 150."""
        # Covers line 262 (boundary)
        is_valid, error = validator.validate_steps(150)
        assert is_valid is True
        assert error is None

    def test_steps_at_one(self, validator):
        """Test validation passes when steps are exactly 1."""
        # Covers line 262 (min boundary)
        is_valid, error = validator.validate_steps(1)
        assert is_valid is True
        assert error is None

    def test_steps_valid_with_model_max(self, validator):
        """Test validation passes when steps are within model max."""
        # Covers line 256 (max_steps check passes)
        is_valid, error = validator.validate_steps(40, max_steps=50)
        assert is_valid is True
        assert error is None


# ============================================================================
# Test Class: validate_cfg_scale
# ============================================================================


class TestValidateCfgScale:
    """Tests for ParameterValidator.validate_cfg_scale (lines 274-279)."""

    def test_valid_cfg_scale(self, validator):
        """Test validation passes for valid CFG scale."""
        # Covers line 279
        is_valid, error = validator.validate_cfg_scale(7.5)
        assert is_valid is True
        assert error is None

    def test_negative_cfg_scale(self, validator):
        """Test validation fails for negative CFG scale."""
        # Covers lines 274-275
        is_valid, error = validator.validate_cfg_scale(-0.1)
        assert is_valid is False
        assert "CFG scale must be non-negative" in error

    def test_cfg_scale_exceeds_max(self, validator):
        """Test validation fails when CFG scale exceeds 20."""
        # Covers lines 276-277
        is_valid, error = validator.validate_cfg_scale(20.1)
        assert is_valid is False
        assert "CFG scale must not exceed 20" in error
        assert "20.1" in error

    def test_cfg_scale_at_zero(self, validator):
        """Test validation passes when CFG scale is 0."""
        # Covers line 279 (boundary)
        is_valid, error = validator.validate_cfg_scale(0.0)
        assert is_valid is True
        assert error is None

    def test_cfg_scale_at_max(self, validator):
        """Test validation passes when CFG scale is exactly 20."""
        # Covers line 279 (max boundary)
        is_valid, error = validator.validate_cfg_scale(20.0)
        assert is_valid is True
        assert error is None


# ============================================================================
# Test Class: validate_seed
# ============================================================================


class TestValidateSeed:
    """Tests for ParameterValidator.validate_seed (lines 291-294)."""

    def test_valid_seed(self, validator):
        """Test validation passes for valid seed."""
        # Covers line 294
        is_valid, error = validator.validate_seed(12345)
        assert is_valid is True
        assert error is None

    def test_seed_below_minimum(self, validator):
        """Test validation fails when seed is below minimum."""
        # Covers lines 291-292
        is_valid, error = validator.validate_seed(-1000000000)
        assert is_valid is False
        assert "Seed must be between -999999999 and 999999999" in error

    def test_seed_above_maximum(self, validator):
        """Test validation fails when seed is above maximum."""
        # Covers lines 291-292
        is_valid, error = validator.validate_seed(1000000000)
        assert is_valid is False
        assert "Seed must be between -999999999 and 999999999" in error

    def test_seed_at_minimum(self, validator):
        """Test validation passes when seed is exactly at minimum."""
        # Covers line 294 (min boundary)
        is_valid, error = validator.validate_seed(-999999999)
        assert is_valid is True
        assert error is None

    def test_seed_at_maximum(self, validator):
        """Test validation passes when seed is exactly at maximum."""
        # Covers line 294 (max boundary)
        is_valid, error = validator.validate_seed(999999999)
        assert is_valid is True
        assert error is None

    def test_seed_zero(self, validator):
        """Test validation passes when seed is 0."""
        # Covers line 294
        is_valid, error = validator.validate_seed(0)
        assert is_valid is True
        assert error is None

    def test_seed_negative(self, validator):
        """Test validation passes for valid negative seed."""
        # Covers line 294
        is_valid, error = validator.validate_seed(-12345)
        assert is_valid is True
        assert error is None


# ============================================================================
# Test Class: validate_lora_strength
# ============================================================================


class TestValidateLoraStrength:
    """Tests for ParameterValidator.validate_lora_strength (lines 306-309)."""

    def test_valid_lora_strength(self, validator):
        """Test validation passes for valid LoRA strength."""
        # Covers line 309
        is_valid, error = validator.validate_lora_strength(50)
        assert is_valid is True
        assert error is None

    def test_lora_strength_below_zero(self, validator):
        """Test validation fails when LoRA strength is below 0."""
        # Covers lines 306-307
        is_valid, error = validator.validate_lora_strength(-1)
        assert is_valid is False
        assert "LoRA strength must be between 0 and 100" in error

    def test_lora_strength_above_100(self, validator):
        """Test validation fails when LoRA strength is above 100."""
        # Covers lines 306-307
        is_valid, error = validator.validate_lora_strength(101)
        assert is_valid is False
        assert "LoRA strength must be between 0 and 100" in error

    def test_lora_strength_at_zero(self, validator):
        """Test validation passes when LoRA strength is 0."""
        # Covers line 309 (min boundary)
        is_valid, error = validator.validate_lora_strength(0)
        assert is_valid is True
        assert error is None

    def test_lora_strength_at_100(self, validator):
        """Test validation passes when LoRA strength is 100."""
        # Covers line 309 (max boundary)
        is_valid, error = validator.validate_lora_strength(100)
        assert is_valid is True
        assert error is None


# ============================================================================
# Test Class: validate_num_images
# ============================================================================


class TestValidateNumImages:
    """Tests for ParameterValidator.validate_num_images (lines 321-326)."""

    def test_valid_num_images(self, validator):
        """Test validation passes for valid number of images."""
        # Covers line 326
        is_valid, error = validator.validate_num_images(2)
        assert is_valid is True
        assert error is None

    def test_num_images_less_than_one(self, validator):
        """Test validation fails when num_images < 1."""
        # Covers lines 321-322
        is_valid, error = validator.validate_num_images(0)
        assert is_valid is False
        assert "Number of images must be at least 1" in error

    def test_num_images_negative(self, validator):
        """Test validation fails for negative num_images."""
        # Covers lines 321-322
        is_valid, error = validator.validate_num_images(-1)
        assert is_valid is False
        assert "Number of images must be at least 1" in error

    def test_num_images_exceeds_max(self, validator):
        """Test validation fails when num_images > 4."""
        # Covers lines 323-324
        is_valid, error = validator.validate_num_images(5)
        assert is_valid is False
        assert "Number of images must not exceed 4" in error

    def test_num_images_at_one(self, validator):
        """Test validation passes when num_images is 1."""
        # Covers line 326 (min boundary)
        is_valid, error = validator.validate_num_images(1)
        assert is_valid is True
        assert error is None

    def test_num_images_at_four(self, validator):
        """Test validation passes when num_images is 4."""
        # Covers line 326 (max boundary)
        is_valid, error = validator.validate_num_images(4)
        assert is_valid is True
        assert error is None


# ============================================================================
# Test Class: Edge Cases and Branch Coverage
# ============================================================================


class TestEdgeCasesAndBranchCoverage:
    """Additional tests for edge cases and branch coverage."""

    @pytest.mark.asyncio
    async def test_model_spec_without_constraints_attribute(self, validator):
        """Test handling model spec without constraints attribute."""
        # Covers line 110 (hasattr check fails)
        model_spec = SimpleNamespace(name="test")  # No constraints
        model = SimpleNamespace(id="test-model", model_spec=model_spec)
        mock_response = SimpleNamespace(data=[model])
        validator.client.models.list = AsyncMock(return_value=mock_response)

        is_valid, error = await validator.validate_image_parameters(
            model="test-model",
            width=512,
            height=512,
            prompt="Valid prompt",
        )

        # Should use default validation (divisible by 8)
        assert is_valid is True

    @pytest.mark.asyncio
    async def test_model_spec_with_non_image_constraints(self, validator):
        """Test handling model spec with non-ImageModelConstraints."""
        # Covers line 111 (isinstance check fails)
        model_spec = SimpleNamespace(
            constraints=SimpleNamespace(temperature=SimpleNamespace(default=0.7))
        )
        model = SimpleNamespace(id="test-model", model_spec=model_spec)
        mock_response = SimpleNamespace(data=[model])
        validator.client.models.list = AsyncMock(return_value=mock_response)

        is_valid, error = await validator.validate_image_parameters(
            model="test-model",
            width=512,
            height=512,
            prompt="Valid prompt",
        )

        # Should use default validation (divisible by 8)
        assert is_valid is True

    @pytest.mark.asyncio
    async def test_steps_with_constraints_no_max_attribute(self, validator):
        """Test steps validation when constraints exist but steps has no max."""
        # Covers line 148 (hasattr check for steps.max fails)
        ImageModelConstraints(
            promptCharacterLimit=1000,
            steps=StepsConstraint(default=25, max=50),
            widthHeightDivisor=8,
        )
        # Create a modified constraints where steps doesn't have max
        modified_constraints = SimpleNamespace(
            promptCharacterLimit=1000,
            steps=SimpleNamespace(default=25),  # No max attribute
            widthHeightDivisor=8,
        )
        model_spec = SimpleNamespace(constraints=modified_constraints)
        model = SimpleNamespace(id="test-model", model_spec=model_spec)
        mock_response = SimpleNamespace(data=[model])
        validator.client.models.list = AsyncMock(return_value=mock_response)

        # Note: This test covers the branch where constraints exist but
        # the isinstance check for ImageModelConstraints fails
        is_valid, error = await validator.validate_image_parameters(
            model="test-model",
            width=512,
            height=512,
            prompt="Valid prompt",
            steps=100,
        )

        # Should still pass general validation (100 < 150)
        assert is_valid is True

    @pytest.mark.asyncio
    async def test_validate_image_parameters_caches_model_spec(
        self, validator, mock_models_response
    ):
        """Test that model spec is cached after first fetch."""
        validator.client.models.list = AsyncMock(return_value=mock_models_response)

        # First call
        await validator.validate_image_parameters(
            model="test-model",
            width=1024,
            height=768,
            prompt="First call",
        )

        # Second call - should use cache
        await validator.validate_image_parameters(
            model="test-model",
            width=1024,
            height=768,
            prompt="Second call",
        )

        # models.list should only be called once
        assert validator.client.models.list.call_count == 1
