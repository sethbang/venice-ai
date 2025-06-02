from typing import Optional, Literal, Dict, Any, TYPE_CHECKING

from .._resource import APIResource, AsyncAPIResource
from ..utils import _prepare_model_list_params
from venice_ai.types.models import ModelList, ModelType, ModelTraitList, ModelCompatibilityList

if TYPE_CHECKING:
    from .._client import VeniceClient
    from .._async_client import AsyncVeniceClient

class Models(APIResource):
    """
    Provides access to model listing and capability information.

    This class manages synchronous operations for retrieving information about
    available AI models, their traits, and compatibility mappings. It provides
    methods to list models, query model traits (semantic shortcuts), and get
    compatibility mappings between external model names and Venice model IDs.

    :param client: The Venice client instance used for API requests.
    :type client: VeniceClient

    Example:
        Basic usage through a Venice client::

            from venice_ai import VeniceClient
            
            client = VeniceClient()
            models = client.models.list()
            for model in models.data:
                print(f"Model: {model.name} (ID: {model.id})")
    """

    def list(
        self,
        *,
        type: Optional[ModelType] = None,
    ) -> ModelList:
        """
        Lists available models.

        Retrieves a list of AI models available through the Venice API. Models can
        optionally be filtered by type to narrow down results to specific categories
        such as text generation, image generation, or embedding models.

        :param type: Optional filter for model type. Valid values include ``"text"``,
            ``"image"``, ``"embedding"``, ``"tts"``, and ``"upscale"``. If not provided,
            all available models are returned.
        :type type: Optional[venice_ai.types.models.ModelType]

        :return: A list of available models with their metadata, capabilities, and pricing information.
        :rtype: venice_ai.types.models.ModelList

        :raises venice_ai.exceptions.APIError: If an API error occurs during the request.

        Example:
            List all available models::

                models = client.models.list()
                for model in models.data:
                    print(f"Model ID: {model.id}, Name: {model.name}")

            Filter models by type::

                chat_models = client.models.list(type="text")
                image_models = client.models.list(type="image")
        """
        params = _prepare_model_list_params(type)
        return self._client.get("models", params=params)
        
    def list_traits(
        self,
        *,
        type: Optional[ModelType] = None,
    ) -> ModelTraitList:
        """
        Lists model traits and their associated model IDs.

        Retrieves a mapping of semantic trait names (e.g., "default", "fastest", "best")
        to their corresponding model IDs. Traits provide convenient shortcuts for
        selecting models based on desired characteristics rather than specific model
        identifiers, making it easier to choose appropriate models without needing
        to know exact model versions or IDs.

        :param type: Optional filter for model type. Only traits for models of the
            specified type will be returned. Valid values include ``"text"``,
            ``"image"``, ``"embedding"``, ``"tts"``, and ``"upscale"``.
        :type type: Optional[venice_ai.types.models.ModelType]

        :return: A mapping of trait names to their corresponding model IDs.
        :rtype: venice_ai.types.models.ModelTraitList

        :raises venice_ai.exceptions.APIError: If an API error occurs during the request.

        Example:
            Get all model traits::

                traits = client.models.list_traits()
                default_model = traits.data.get("default")
                fastest_model = traits.data.get("fastest")

            Get traits for specific model type::

                text_traits = client.models.list_traits(type="text")
                print(f"Default text model: {text_traits.data['default']}")
        """
        params: Dict[str, Any] = {}
        if type is not None:
            params["type"] = type

        return self._client.get("models/traits", params=params)
        
    def list_compatibility(
        self,
        *,
        type: Optional[ModelType] = None,
    ) -> ModelCompatibilityList:
        """
        Lists model compatibility mapping between external model names and Venice model IDs.

        Retrieves a mapping that allows applications to reference external model
        identifiers (e.g., from other AI platforms like OpenAI) and have them
        automatically mapped to equivalent Venice models. This compatibility layer
        facilitates smoother transitions when migrating applications from other
        AI platforms to Venice.

        :param type: Optional filter for model type. Only compatibility mappings for
            models of the specified type will be returned. Valid values include
            ``"text"``, ``"image"``, ``"embedding"``, ``"tts"``, and ``"upscale"``.
        :type type: Optional[venice_ai.types.models.ModelType]

        :return: A mapping of external model names to their equivalent Venice model IDs.
        :rtype: venice_ai.types.models.ModelCompatibilityList

        :raises venice_ai.exceptions.APIError: If an API error occurs during the request.

        Example:
            Get all compatibility mappings::

                compatibility = client.models.list_compatibility()
                venice_model = compatibility.data.get("gpt-4")
                print(f"GPT-4 maps to Venice model: {venice_model}")

            Get compatibility for specific model type::

                text_compat = client.models.list_compatibility(type="text")
                for external_name, venice_id in text_compat.data.items():
                    print(f"{external_name} -> {venice_id}")
        """
        params: Dict[str, Any] = {}
        if type is not None:
            params["type"] = type

        return self._client.get("models/compatibility_mapping", params=params)


class AsyncModels(AsyncAPIResource):
    """
    Provides access to model listing and capability information (asynchronous).

    This class manages asynchronous operations for retrieving information about
    available AI models, their traits, and compatibility mappings. It provides
    async methods to list models, query model traits (semantic shortcuts), and get
    compatibility mappings between external model names and Venice model IDs.

    :param client: The async Venice client instance used for API requests.
    :type client: AsyncVeniceClient

    Example:
        Basic usage through an async Venice client::

            from venice_ai import AsyncVeniceClient
            
            async def main():
                client = AsyncVeniceClient()
                models = await client.models.list()
                for model in models.data:
                    print(f"Model: {model.name} (ID: {model.id})")
    """

    async def list(
        self,
        *,
        type: Optional[ModelType] = None,
    ) -> ModelList:
        """
        Lists available models asynchronously.

        Asynchronously retrieves a list of AI models available through the Venice API.
        Models can optionally be filtered by type to narrow down results to specific
        categories such as text generation, image generation, or embedding models.

        :param type: Optional filter for model type. Valid values include ``"text"``,
            ``"image"``, ``"embedding"``, ``"tts"``, and ``"upscale"``. If not provided,
            all available models are returned.
        :type type: Optional[venice_ai.types.models.ModelType]

        :return: A list of available models with their metadata, capabilities, and pricing information.
        :rtype: venice_ai.types.models.ModelList

        :raises venice_ai.exceptions.APIError: If an API error occurs during the request.

        Example:
            List all available models::

                models = await client.models.list()
                for model in models.data:
                    print(f"Model ID: {model.id}, Name: {model.name}")

            Filter models by type::

                chat_models = await client.models.list(type="text")
                image_models = await client.models.list(type="image")
        """
        params = _prepare_model_list_params(type)
        return await self._client.get("models", params=params)
        
    async def list_traits(
        self,
        *,
        type: Optional[ModelType] = None,
    ) -> ModelTraitList:
        """
        Lists model traits and their associated model IDs asynchronously.

        Asynchronously retrieves a mapping of semantic trait names (e.g., "default",
        "fastest", "best") to their corresponding model IDs. Traits provide convenient
        shortcuts for selecting models based on desired characteristics rather than
        specific model identifiers, making it easier to choose appropriate models
        without needing to know exact model versions or IDs.

        :param type: Optional filter for model type. Only traits for models of the
            specified type will be returned. Valid values include ``"text"``,
            ``"image"``, ``"embedding"``, ``"tts"``, and ``"upscale"``.
        :type type: Optional[venice_ai.types.models.ModelType]

        :return: A mapping of trait names to their corresponding model IDs.
        :rtype: venice_ai.types.models.ModelTraitList

        :raises venice_ai.exceptions.APIError: If an API error occurs during the request.

        Example:
            Get all model traits::

                traits = await client.models.list_traits()
                default_model = traits.data.get("default")
                fastest_model = traits.data.get("fastest")

            Get traits for specific model type::

                text_traits = await client.models.list_traits(type="text")
                print(f"Default text model: {text_traits.data['default']}")
        """
        params: Dict[str, Any] = {}
        if type is not None:
            params["type"] = type

        return await self._client.get("models/traits", params=params)
        
    async def list_compatibility(
        self,
        *,
        type: Optional[ModelType] = None,
    ) -> ModelCompatibilityList:
        """
        Lists model compatibility mapping between external model names and Venice model IDs asynchronously.

        Asynchronously retrieves a mapping that allows applications to reference
        external model identifiers (e.g., from other AI platforms like OpenAI) and
        have them automatically mapped to equivalent Venice models. This compatibility
        layer facilitates smoother transitions when migrating applications from other
        AI platforms to Venice.

        :param type: Optional filter for model type. Only compatibility mappings for
            models of the specified type will be returned. Valid values include
            ``"text"``, ``"image"``, ``"embedding"``, ``"tts"``, and ``"upscale"``.
        :type type: Optional[venice_ai.types.models.ModelType]

        :return: A mapping of external model names to their equivalent Venice model IDs.
        :rtype: venice_ai.types.models.ModelCompatibilityList

        :raises venice_ai.exceptions.APIError: If an API error occurs during the request.

        Example:
            Get all compatibility mappings::

                compatibility = await client.models.list_compatibility()
                venice_model = compatibility.data.get("gpt-4")
                print(f"GPT-4 maps to Venice model: {venice_model}")

            Get compatibility for specific model type::

                text_compat = await client.models.list_compatibility(type="text")
                for external_name, venice_id in text_compat.data.items():
                    print(f"{external_name} -> {venice_id}")
        """
        params: Dict[str, Any] = {}
        if type is not None:
            params["type"] = type

        return await self._client.get("models/compatibility_mapping", params=params)