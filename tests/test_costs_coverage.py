"""
Additional test cases for costs.py to improve coverage.
"""

import pytest
from unittest.mock import Mock
from venice_ai.costs import calculate_completion_cost, calculate_embedding_cost, estimate_completion_cost
from venice_ai.types.models import ModelPricing
from venice_ai.types.chat import ChatCompletion, UsageData
from typing import Any, Dict


class TestCostsCoverage:
    """Additional test cases to improve coverage of costs.py"""
    
    def test_calculate_completion_cost_legacy_with_vcu(self):
        """Test legacy pricing with VCU values."""
        mock_usage = Mock(spec=UsageData)
        mock_usage.prompt_tokens = 100
        mock_usage.completion_tokens = 50
        
        mock_completion = Mock(spec=ChatCompletion)
        mock_completion.usage = mock_usage
        
        # Legacy pricing with VCU
        pricing: Dict[str, Any] = {
            'input_cost_per_mtok': 0.5,
            'output_cost_per_mtok': 2.0,
            'input_cost_per_mtok_vcu': 5.0,
            'output_cost_per_mtok_vcu': 20.0
        }
        
        result = calculate_completion_cost(mock_completion, pricing)  # type: ignore
        
        # USD: (100/1,000,000 * 0.5) + (50/1,000,000 * 2.0) = 0.00005 + 0.0001 = 0.00015
        # VCU: (100/1,000,000 * 5.0) + (50/1,000,000 * 20.0) = 0.0005 + 0.001 = 0.0015
        assert result['usd'] == pytest.approx(0.00015)
        assert result['vcu'] == pytest.approx(0.0015)
    
    def test_calculate_embedding_cost_legacy_with_vcu(self):
        """Test embedding cost with legacy VCU pricing."""
        mock_usage = Mock()
        mock_usage.total_tokens = 500
        
        mock_response = Mock()
        mock_response.usage = mock_usage
        
        pricing: Dict[str, Any] = {
            'input_cost_per_mtok': 0.1,
            'input_cost_per_mtok_vcu': 1.0
        }
        
        result = calculate_embedding_cost(mock_response, pricing)  # type: ignore
        
        # USD: 500 / 1,000,000 * 0.1 = 0.00005
        # VCU: 500 / 1,000,000 * 1.0 = 0.0005
        assert result['usd'] == pytest.approx(0.00005)
        assert result['vcu'] == pytest.approx(0.0005)
    
    def test_estimate_completion_cost_legacy_with_vcu(self):
        """Test estimation with legacy VCU pricing."""
        prompt = "Test prompt with some words"
        
        pricing: Dict[str, Any] = {
            'input_cost_per_mtok': 0.3,
            'output_cost_per_mtok': 1.5,
            'input_cost_per_mtok_vcu': 3.0,
            'output_cost_per_mtok_vcu': 15.0
        }
        
        result = estimate_completion_cost(
            prompt=prompt,
            estimated_completion_tokens=100,
            model_pricing=pricing  # type: ignore
        )
        
        # Should have both USD and VCU costs
        assert result['usd'] > 0
        assert result['vcu'] > 0
    
    def test_calculate_completion_cost_new_structure_zero_values_fallback(self):
        """Test that when new structure has zero values, it falls back to legacy."""
        mock_usage = Mock(spec=UsageData)
        mock_usage.prompt_tokens = 100
        mock_usage.completion_tokens = 50
        
        mock_completion = Mock(spec=ChatCompletion)
        mock_completion.usage = mock_usage
        
        # New structure with zero values, should fallback to legacy
        pricing: Dict[str, Any] = {
            'input': {'usd': 0.0, 'vcu': 0.0},
            'output': {'usd': 0.0, 'vcu': 0.0},
            'input_cost_per_mtok': 0.5,
            'output_cost_per_mtok': 2.0,
            'input_cost_per_mtok_vcu': 5.0,
            'output_cost_per_mtok_vcu': 20.0
        }
        
        result = calculate_completion_cost(mock_completion, pricing)  # type: ignore
        
        # Should use legacy values
        assert result['usd'] == pytest.approx(0.00015)
        assert result['vcu'] == pytest.approx(0.0015)
    
    def test_calculate_embedding_cost_new_structure_zero_values_fallback(self):
        """Test embedding cost fallback when new structure has zero values."""
        mock_usage = Mock()
        mock_usage.total_tokens = 500
        
        mock_response = Mock()
        mock_response.usage = mock_usage
        
        pricing: Dict[str, Any] = {
            'input': {'usd': 0.0, 'vcu': 0.0},
            'input_cost_per_mtok': 0.1,
            'input_cost_per_mtok_vcu': 1.0
        }
        
        result = calculate_embedding_cost(mock_response, pricing)  # type: ignore
        
        # Should use legacy values
        assert result['usd'] == pytest.approx(0.00005)
        assert result['vcu'] == pytest.approx(0.0005)
    
    def test_estimate_completion_cost_new_structure_zero_values_fallback(self):
        """Test estimation fallback when new structure has zero values."""
        prompt = "Test prompt"
        
        pricing: Dict[str, Any] = {
            'input': {'usd': 0.0, 'vcu': 0.0},
            'output': {'usd': 0.0, 'vcu': 0.0},
            'input_cost_per_mtok': 0.3,
            'output_cost_per_mtok': 1.5,
            'input_cost_per_mtok_vcu': 3.0,
            'output_cost_per_mtok_vcu': 15.0
        }
        
        result = estimate_completion_cost(
            prompt=prompt,
            estimated_completion_tokens=100,
            model_pricing=pricing  # type: ignore
        )
        
        # Should use legacy values
        assert result['usd'] > 0
        assert result['vcu'] > 0