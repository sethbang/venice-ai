"""
Parameter validation for Venice AI CLI image generation.

This module provides comprehensive parameter validation against model-specific
constraints to ensure parameters are valid before making API calls.
"""

from typing import TYPE_CHECKING, Any

from venice_ai import VeniceClient
from venice_ai.types.api.models import ImageModelConstraints

if TYPE_CHECKING:
    pass


class ParameterValidator:
    """
    Validates image generation parameters against model-specific constraints.

    This validator fetches model specifications from the Venice API and caches
    them to avoid repeated API calls. It validates all image generation parameters
    including dimensions, steps, prompts, and other settings against the constraints
    defined for each specific model.

    Attributes:
        client: Venice AI client instance for fetching model specifications
        _model_specs_cache: Cached model specifications to avoid repeated API calls
    """

    def __init__(self, client: VeniceClient):
        """
        Initialize the parameter validator.

        Args:
            client: Venice AI client instance
        """
        self.client = client
        self._model_specs_cache: dict[str, Any] = {}

    async def _get_model_spec(self, model_id: str) -> Any | None:
        """
        Fetch and cache model specification.

        Args:
            model_id: The model identifier

        Returns:
            Model specification object or None if not found
        """
        # Return cached spec if available
        if model_id in self._model_specs_cache:
            return self._model_specs_cache[model_id]

        try:
            # Fetch all models and find the requested one
            models_response = await self.client.models.list(type="image")

            for model in models_response.data:
                if model.id == model_id:
                    # Cache the model spec
                    self._model_specs_cache[model_id] = model.model_spec
                    return model.model_spec

            # Model not found
            return None

        except Exception:
            # If we can't fetch model specs, return None to skip validation
            return None

    async def validate_image_parameters(
        self,
        model: str,
        width: int,
        height: int,
        prompt: str,
        steps: int | None = None,
        cfg_scale: float | None = None,
        seed: int | None = None,
        lora_strength: int | None = None,
        num_images: int = 1,
    ) -> tuple[bool, str | None]:
        """
        Validate all image generation parameters against model constraints.

        This method performs comprehensive validation of image generation parameters,
        checking against model-specific constraints when available and general limits
        for all parameters.

        Args:
            model: Model identifier
            width: Image width in pixels
            height: Image height in pixels
            prompt: Text prompt for generation
            steps: Optional number of inference steps
            cfg_scale: Optional CFG scale value
            seed: Optional random seed
            lora_strength: Optional LoRA strength (0-100)
            num_images: Number of images to generate

        Returns:
            Tuple of (is_valid, error_message). If valid, error_message is None.
            Otherwise, error_message contains a helpful description of the issue.
        """
        # Fetch model specification
        model_spec = await self._get_model_spec(model)

        # Extract constraints if available
        constraints: ImageModelConstraints | None = None
        if (
            model_spec
            and hasattr(model_spec, "constraints")
            and isinstance(model_spec.constraints, ImageModelConstraints)
        ):
            constraints = model_spec.constraints

        # Validate dimensions
        if constraints:
            divisor = int(constraints.widthHeightDivisor)
            if width % divisor != 0:
                return False, f"Width must be divisible by {divisor} (got {width})"
            if height % divisor != 0:
                return False, f"Height must be divisible by {divisor} (got {height})"
        else:
            # Default validation: divisible by 8 (common requirement)
            if width % 8 != 0:
                return False, f"Width must be divisible by 8 (got {width})"
            if height % 8 != 0:
                return False, f"Height must be divisible by 8 (got {height})"

        # Validate dimensions are positive and reasonable
        if width <= 0 or height <= 0:
            return False, "Width and height must be positive"
        if width > 4096 or height > 4096:
            return False, "Width and height must not exceed 4096 pixels"

        # Validate prompt length
        if constraints:
            max_prompt_length = int(constraints.promptCharacterLimit)
            if len(prompt) > max_prompt_length:
                return (
                    False,
                    f"Prompt exceeds maximum length of {max_prompt_length} characters (got {len(prompt)})",
                )

        if not prompt or len(prompt.strip()) == 0:
            return False, "Prompt cannot be empty"

        # Validate steps
        if steps is not None:
            if constraints and hasattr(constraints.steps, "max"):
                max_steps = int(constraints.steps.max)
                if steps > max_steps:
                    return (
                        False,
                        f"Steps exceed maximum of {max_steps} for model '{model}' (got {steps})",
                    )

            if steps < 1:
                return False, "Steps must be at least 1"
            if steps > 150:  # General upper limit
                return False, "Steps must not exceed 150"

        # Validate CFG scale
        if cfg_scale is not None:
            if cfg_scale < 0:
                return False, "CFG scale must be non-negative"
            if cfg_scale > 20:
                return False, f"CFG scale must not exceed 20 (got {cfg_scale:.1f})"

        # Validate seed
        if seed is not None and (seed < -999999999 or seed > 999999999):
            return False, "Seed must be between -999999999 and 999999999"

        # Validate LoRA strength
        if lora_strength is not None and (lora_strength < 0 or lora_strength > 100):
            return False, "LoRA strength must be between 0 and 100"

        # Validate number of images
        if num_images < 1:
            return False, "Number of images must be at least 1"
        if num_images > 4:
            return False, "Number of images must not exceed 4"

        # All validations passed
        return True, None

    def validate_dimensions(
        self, width: int, height: int, divisor: int = 8
    ) -> tuple[bool, str | None]:
        """
        Validate image dimensions without model specification.

        Provides basic dimension validation when model specs are unavailable.

        Args:
            width: Image width in pixels
            height: Image height in pixels
            divisor: Required divisor for dimensions (default: 8)

        Returns:
            Tuple of (is_valid, error_message)
        """
        if width % divisor != 0:
            return False, f"Width must be divisible by {divisor} (got {width})"
        if height % divisor != 0:
            return False, f"Height must be divisible by {divisor} (got {height})"
        if width <= 0 or height <= 0:
            return False, "Width and height must be positive"
        if width > 4096 or height > 4096:
            return False, "Width and height must not exceed 4096 pixels"

        return True, None

    def validate_prompt(
        self, prompt: str, max_length: int | None = None
    ) -> tuple[bool, str | None]:
        """
        Validate prompt text.

        Args:
            prompt: The prompt text
            max_length: Optional maximum length constraint

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not prompt or len(prompt.strip()) == 0:
            return False, "Prompt cannot be empty"

        if max_length and len(prompt) > max_length:
            return (
                False,
                f"Prompt exceeds maximum length of {max_length} characters (got {len(prompt)})",
            )

        return True, None

    def validate_steps(self, steps: int, max_steps: int | None = None) -> tuple[bool, str | None]:
        """
        Validate inference steps.

        Args:
            steps: Number of inference steps
            max_steps: Optional model-specific maximum

        Returns:
            Tuple of (is_valid, error_message)
        """
        if steps < 1:
            return False, "Steps must be at least 1"

        if max_steps and steps > max_steps:
            return False, f"Steps exceed maximum of {max_steps} (got {steps})"

        if steps > 150:
            return False, "Steps must not exceed 150"

        return True, None

    def validate_cfg_scale(self, cfg_scale: float) -> tuple[bool, str | None]:
        """
        Validate CFG scale parameter.

        Args:
            cfg_scale: CFG scale value

        Returns:
            Tuple of (is_valid, error_message)
        """
        if cfg_scale < 0:
            return False, "CFG scale must be non-negative"
        if cfg_scale > 20:
            return False, f"CFG scale must not exceed 20 (got {cfg_scale:.1f})"

        return True, None

    def validate_seed(self, seed: int) -> tuple[bool, str | None]:
        """
        Validate random seed.

        Args:
            seed: Random seed value

        Returns:
            Tuple of (is_valid, error_message)
        """
        if seed < -999999999 or seed > 999999999:
            return False, "Seed must be between -999999999 and 999999999"

        return True, None

    def validate_lora_strength(self, lora_strength: int) -> tuple[bool, str | None]:
        """
        Validate LoRA strength parameter.

        Args:
            lora_strength: LoRA strength value (0-100)

        Returns:
            Tuple of (is_valid, error_message)
        """
        if lora_strength < 0 or lora_strength > 100:
            return False, "LoRA strength must be between 0 and 100"

        return True, None

    def validate_num_images(self, num_images: int) -> tuple[bool, str | None]:
        """
        Validate number of images to generate.

        Args:
            num_images: Number of images

        Returns:
            Tuple of (is_valid, error_message)
        """
        if num_images < 1:
            return False, "Number of images must be at least 1"
        if num_images > 4:
            return False, "Number of images must not exceed 4"

        return True, None


__all__ = ["ParameterValidator"]
