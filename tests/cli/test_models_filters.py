"""
Tests for model filtering functionality
"""

from types import SimpleNamespace

from venice_ai.cli.commands.models.filters import FilterOptions, ModelFilter


class TestModelFilter:
    """Test ModelFilter class"""

    def test_filter_by_type_single(self, sample_models):
        """Test filtering by single type"""
        result = ModelFilter.filter_by_type(sample_models, ["text"])
        assert len(result) == 1
        assert all(m.type == "text" for m in result)

    def test_filter_by_type_multiple(self, sample_models):
        """Test filtering by multiple types"""
        result = ModelFilter.filter_by_type(sample_models, ["text", "image"])
        assert len(result) == 2
        assert all(m.type in ["text", "image"] for m in result)

    def test_filter_by_type_all(self, sample_models):
        """Test that 'all' returns all models"""
        result = ModelFilter.filter_by_type(sample_models, ["all"])
        assert len(result) == len(sample_models)

    def test_filter_by_type_empty(self, sample_models):
        """Test that None/empty returns all models"""
        result = ModelFilter.filter_by_type(sample_models, None)
        assert len(result) == len(sample_models)

    def test_filter_by_capabilities_function_calling(self, sample_models):
        """Test filtering by function calling capability"""
        result = ModelFilter.filter_by_capabilities(sample_models, ["function-calling"])
        assert len(result) == 1
        assert result[0].type == "text"

    def test_filter_by_capabilities_multiple(self, sample_models):
        """Test filtering by multiple capabilities"""
        result = ModelFilter.filter_by_capabilities(
            sample_models, ["function-calling", "reasoning"]
        )
        assert len(result) == 1
        assert result[0].model_spec.capabilities.supportsFunctionCalling
        assert result[0].model_spec.capabilities.supportsReasoning

    def test_filter_by_capabilities_vision(self, sample_models):
        """Test filtering models with vision"""
        result = ModelFilter.filter_by_capabilities(sample_models, ["vision"])
        # None of our test models have vision
        assert len(result) == 0

    def test_filter_by_price_max_input(self, sample_models):
        """Test filtering by max input price"""
        options = FilterOptions(max_input_price=0.2)
        result = ModelFilter.filter_by_price(sample_models, options)
        # Should filter out models with input > 0.2
        for model in result:
            if (
                hasattr(model.model_spec, "pricing")
                and model.model_spec.pricing
                and hasattr(model.model_spec.pricing, "input")
                and model.model_spec.pricing.input
            ):
                assert model.model_spec.pricing.input.usd <= 0.2

    def test_filter_by_price_max_output(self, sample_models):
        """Test filtering by max output price"""
        options = FilterOptions(max_output_price=1.0)
        result = ModelFilter.filter_by_price(sample_models, options)
        # Should filter out models with output > 1.0
        for model in result:
            if (
                hasattr(model.model_spec, "pricing")
                and model.model_spec.pricing
                and hasattr(model.model_spec.pricing, "output")
                and model.model_spec.pricing.output
            ):
                assert model.model_spec.pricing.output.usd <= 1.0

    def test_filter_by_price_budget(self, sample_models):
        """Test filtering by budget (average price)"""
        options = FilterOptions(budget=0.5)
        result = ModelFilter.filter_by_price(sample_models, options)
        # Should filter models where (input+output)/2 <= 0.5
        for model in result:
            if (
                hasattr(model.model_spec, "pricing")
                and model.model_spec.pricing
                and (
                    hasattr(model.model_spec.pricing, "input")
                    and hasattr(model.model_spec.pricing, "output")
                    and model.model_spec.pricing.input
                    and model.model_spec.pricing.output
                )
            ):
                avg = (model.model_spec.pricing.input.usd + model.model_spec.pricing.output.usd) / 2
                assert avg <= 0.5

    def test_filter_by_traits(self, sample_models):
        """Test filtering by traits"""
        result = ModelFilter.filter_by_traits(sample_models, ["default"])
        assert len(result) == 1
        assert "default" in result[0].model_spec.traits

    def test_filter_by_traits_multiple(self, sample_models):
        """Test filtering by multiple traits (OR logic)"""
        result = ModelFilter.filter_by_traits(sample_models, ["default", "highest_quality"])
        assert len(result) == 2

    def test_filter_by_status_beta(self, sample_models):
        """Test filtering beta models"""
        # Add a beta model
        beta_model = sample_models[0]
        beta_model.model_spec.beta = True

        options = FilterOptions(beta=True)
        result = ModelFilter.filter_by_status(sample_models, options)
        assert all(getattr(m.model_spec, "beta", False) for m in result)

    def test_filter_by_status_no_beta(self, sample_models):
        """Test excluding beta models"""
        options = FilterOptions(beta=False)
        result = ModelFilter.filter_by_status(sample_models, options)
        assert all(not getattr(m.model_spec, "beta", False) for m in result)

    def test_filter_by_status_online(self, sample_models):
        """Test filtering online models"""
        options = FilterOptions(online=True)
        result = ModelFilter.filter_by_status(sample_models, options)
        assert all(not getattr(m.model_spec, "offline", False) for m in result)

    def test_search_models_by_id(self, sample_models):
        """Test searching models by ID"""
        result = ModelFilter.search_models(sample_models, "text")
        assert len(result) >= 1
        assert any("text" in m.id.lower() for m in result)

    def test_search_models_by_name(self, sample_models):
        """Test searching models by name"""
        result = ModelFilter.search_models(sample_models, "Test")
        assert len(result) >= 1

    def test_search_models_case_insensitive(self, sample_models):
        """Test that search is case-insensitive"""
        result1 = ModelFilter.search_models(sample_models, "TEXT")
        result2 = ModelFilter.search_models(sample_models, "text")
        assert len(result1) == len(result2)

    def test_apply_all_filters(self, sample_models):
        """Test applying multiple filters together"""
        options = FilterOptions(
            types=["text"], capabilities=["function-calling"], max_input_price=1.0
        )
        result = ModelFilter.apply_all_filters(sample_models, options)
        assert len(result) == 1
        assert result[0].type == "text"
        assert result[0].model_spec.capabilities.supportsFunctionCalling

    def test_apply_all_filters_no_matches(self, sample_models):
        """Test that incompatible filters return empty list"""
        options = FilterOptions(
            types=["text"],
            capabilities=["vision"],  # None of our text models have vision
        )
        result = ModelFilter.apply_all_filters(sample_models, options)
        assert len(result) == 0

    def test_filter_models_without_pricing(self):
        """Test filtering models that don't have pricing"""
        model_no_price = SimpleNamespace(
            id="no-price",
            type="text",
            model_spec=SimpleNamespace(),
            # No pricing attribute
        )

        options = FilterOptions(max_input_price=1.0)
        result = ModelFilter.filter_by_price([model_no_price], options)
        # Should be excluded when price filters are set
        assert len(result) == 0
