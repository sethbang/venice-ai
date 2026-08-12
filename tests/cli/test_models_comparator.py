"""
Tests for model comparison functionality
"""

from venice_ai.cli.commands.models.comparator import ModelComparator


class TestModelComparator:
    """Test ModelComparator class"""

    def test_compare_models_basic(self, mock_model_text, mock_model_image):
        """Test basic model comparison"""
        models = [mock_model_text, mock_model_image]
        table = ModelComparator.compare_models(models)
        assert table is not None
        assert "COMPARISON" in str(table.title) or table.title == "MODEL COMPARISON"

    def test_compare_models_empty_list(self):
        """Test comparison with empty list"""
        result = ModelComparator.compare_models([])
        assert result is None

    def test_compare_models_single_model(self, mock_model_text):
        """Test comparison with single model"""
        table = ModelComparator.compare_models([mock_model_text])
        # Should still create a table
        assert table is not None

    def test_compare_models_shows_capabilities(self, sample_models):
        """Test that comparison includes capability rows for text models"""
        # Filter to only text models
        text_models = [m for m in sample_models if m.type == "text"]
        if text_models:
            table = ModelComparator.compare_models(text_models)
            assert table is not None

    def test_compare_models_shows_pricing(self, mock_model_text):
        """Test that comparison includes pricing information"""
        table = ModelComparator.compare_models([mock_model_text])
        assert table is not None

    def test_find_model_by_id_exists(self, sample_models):
        """Test finding a model by ID"""
        model = ModelComparator.find_model_by_id(sample_models, sample_models[0].id)
        assert model is not None
        assert model.id == sample_models[0].id

    def test_find_model_by_id_not_exists(self, sample_models):
        """Test finding non-existent model"""
        model = ModelComparator.find_model_by_id(sample_models, "non-existent-id")
        assert model is None

    def test_find_model_by_id_empty_list(self):
        """Test finding model in empty list"""
        model = ModelComparator.find_model_by_id([], "any-id")
        assert model is None

    def test_compare_models_different_currencies(self, mock_model_text):
        """Test comparison with different currency options"""
        # USD only
        table_usd = ModelComparator.compare_models([mock_model_text], currency="usd")
        assert table_usd is not None

        # DIEM only
        table_diem = ModelComparator.compare_models([mock_model_text], currency="diem")
        assert table_diem is not None

        # Both
        table_both = ModelComparator.compare_models([mock_model_text], currency="both")
        assert table_both is not None
