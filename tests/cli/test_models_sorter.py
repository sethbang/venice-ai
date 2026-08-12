"""
Tests for model sorting functionality
"""

from unittest.mock import Mock

from venice_ai.cli.commands.models.sorter import ModelSorter


class TestModelSorter:
    """Test ModelSorter class"""

    def test_sort_by_name(self, sample_models):
        """Test sorting by name"""
        result = ModelSorter.sort_models(sample_models, "name")
        assert len(result) == len(sample_models)
        # Verify sorted order
        names = [
            m.model_spec.name.lower() if hasattr(m.model_spec, "name") else m.id.lower()
            for m in result
        ]
        assert names == sorted(names)

    def test_sort_by_id(self, sample_models):
        """Test sorting by ID"""
        result = ModelSorter.sort_models(sample_models, "id")
        assert len(result) == len(sample_models)
        ids = [m.id.lower() for m in result]
        assert ids == sorted(ids)

    def test_sort_by_price_asc(self):
        """Test sorting by price ascending"""
        # Create models with different prices
        model1 = Mock()
        model1.id = "expensive"
        model1.model_spec = Mock()
        model1.model_spec.pricing = Mock()
        model1.model_spec.pricing.input = Mock()
        model1.model_spec.pricing.input.usd = 1.0

        model2 = Mock()
        model2.id = "cheap"
        model2.model_spec = Mock()
        model2.model_spec.pricing = Mock()
        model2.model_spec.pricing.input = Mock()
        model2.model_spec.pricing.input.usd = 0.1

        models = [model1, model2]
        result = ModelSorter.sort_models(models, "price-asc")
        assert result[0].id == "cheap"
        assert result[1].id == "expensive"

    def test_sort_by_price_desc(self):
        """Test sorting by price descending"""
        model1 = Mock()
        model1.id = "expensive"
        model1.model_spec = Mock()
        model1.model_spec.pricing = Mock()
        model1.model_spec.pricing.input = Mock()
        model1.model_spec.pricing.input.usd = 1.0

        model2 = Mock()
        model2.id = "cheap"
        model2.model_spec = Mock()
        model2.model_spec.pricing = Mock()
        model2.model_spec.pricing.input = Mock()
        model2.model_spec.pricing.input.usd = 0.1

        models = [model2, model1]
        result = ModelSorter.sort_models(models, "price-desc")
        assert result[0].id == "expensive"
        assert result[1].id == "cheap"

    def test_sort_by_context(self):
        """Test sorting by context window"""
        model1 = Mock()
        model1.id = "small-context"
        model1.model_spec = Mock()
        model1.model_spec.availableContextTokens = 4096

        model2 = Mock()
        model2.id = "large-context"
        model2.model_spec = Mock()
        model2.model_spec.availableContextTokens = 131072

        models = [model1, model2]
        result = ModelSorter.sort_models(models, "context")
        assert result[0].id == "large-context"
        assert result[1].id == "small-context"

    def test_sort_by_created(self):
        """Test sorting by creation date"""
        model1 = Mock()
        model1.id = "older"
        model1.created = 1000000
        model1.model_spec = Mock()

        model2 = Mock()
        model2.id = "newer"
        model2.created = 2000000
        model2.model_spec = Mock()

        models = [model1, model2]
        result = ModelSorter.sort_models(models, "created")
        assert result[0].id == "newer"
        assert result[1].id == "older"

    def test_sort_empty_list(self):
        """Test sorting empty list"""
        result = ModelSorter.sort_models([], "name")
        assert result == []

    def test_sort_invalid_criterion(self, sample_models):
        """Test sorting with invalid criterion defaults to ID sort"""
        result = ModelSorter.sort_models(sample_models, "invalid")
        assert len(result) == len(sample_models)
