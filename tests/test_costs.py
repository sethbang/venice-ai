"""
Test suite for the costs module.
"""

import pytest
from unittest.mock import Mock
from venice_ai.costs import calculate_completion_cost, calculate_embedding_cost, estimate_completion_cost
from venice_ai.types.models import ModelPricing
from venice_ai.types.chat import ChatCompletion, UsageData
from typing import Dict, Any


class TestCalculateCompletionCost:
    """Test cases for calculate_completion_cost function."""
    
    def test_calculate_with_new_pricing_structure(self):
        """Test calculation with new USD and VCU pricing structure."""
        # Create mock completion
        mock_usage = Mock(spec=UsageData)
        mock_usage.prompt_tokens = 100
        mock_usage.completion_tokens = 50
        
        mock_completion = Mock(spec=ChatCompletion)
        mock_completion.usage = mock_usage
        
        # Create pricing with new structure (per million tokens)
        pricing: ModelPricing = {
            'input': {'usd': 500, 'vcu': 5000},
            'output': {'usd': 2000, 'vcu': 20000}
        }
        
        # Calculate costs
        result = calculate_completion_cost(mock_completion, pricing)
        
        # Verify calculations
        # Input: 100 tokens / 1_000_000 * 500 USD = 0.05 USD
        # Input: 100 tokens / 1_000_000 * 5000 VCU = 0.5 VCU
        # Output: 50 tokens / 1_000_000 * 2000 USD = 0.1 USD
        # Output: 50 tokens / 1_000_000 * 20000 VCU = 1.0 VCU
        assert result['usd'] == pytest.approx(0.15)
        assert result['vcu'] == pytest.approx(1.5)
    
    def test_calculate_with_legacy_pricing_structure(self):
        """Test calculation with legacy pricing structure (USD only)."""
        # Create mock completion
        mock_usage = Mock(spec=UsageData)
        mock_usage.prompt_tokens = 200
        mock_usage.completion_tokens = 100
        
        mock_completion = Mock(spec=ChatCompletion)
        mock_completion.usage = mock_usage
        
        # Create pricing with legacy structure
        pricing: ModelPricing = {
            'input': {'usd': 0.0, 'vcu': 0.0},  # Required by ModernPricing
            'output': {'usd': 0.0, 'vcu': 0.0}, # Required by ModernPricing
            'input_cost_per_mtok': 0.3,
            'output_cost_per_mtok': 1.5
        }
        
        # Calculate costs
        result = calculate_completion_cost(mock_completion, pricing)
        
        # Verify calculations
        # Input: 200 tokens / 1,000,000 * 0.3 USD = 0.00006 USD
        # Output: 100 tokens / 1,000,000 * 1.5 USD = 0.00015 USD
        assert result['usd'] == pytest.approx(0.00021)
        assert result['vcu'] == 0.0
    
    def test_calculate_with_missing_usage(self):
        """Test calculation when usage data is missing."""
        mock_completion = Mock(spec=ChatCompletion)
        mock_completion.usage = None
        
        pricing: ModelPricing = {
            'input': {'usd': 0.5, 'vcu': 5},
            'output': {'usd': 2.0, 'vcu': 20}
        }
        
        result = calculate_completion_cost(mock_completion, pricing)
        
        assert result['usd'] == 0.0
        assert result['vcu'] == 0.0
    
    def test_calculate_with_partial_pricing(self):
        """Test calculation when pricing has only USD or only VCU."""
        mock_usage = Mock(spec=UsageData)
        mock_usage.prompt_tokens = 100
        mock_usage.completion_tokens = 50
        
        mock_completion = Mock(spec=ChatCompletion)
        mock_completion.usage = mock_usage
        
        # Pricing with only USD (per million tokens)
        pricing: ModelPricing = {
            'input': {'usd': 500, 'vcu': 0.0},
            'output': {'usd': 2000, 'vcu': 0.0}
        }
        
        result = calculate_completion_cost(mock_completion, pricing)
        
        assert result['usd'] == pytest.approx(0.15)
        assert result['vcu'] == 0.0
    
    def test_calculate_with_empty_pricing(self):
        """Test calculation with empty pricing structure."""
        mock_usage = Mock(spec=UsageData)
        mock_usage.prompt_tokens = 100
        mock_usage.completion_tokens = 50
        
        mock_completion = Mock(spec=ChatCompletion)
        mock_completion.usage = mock_usage
        
        pricing: ModelPricing = {
            'input': {'usd': 0.0, 'vcu': 0.0},  # Required by ModernPricing
            'output': {'usd': 0.0, 'vcu': 0.0} # Required by ModernPricing
        }
        
        result = calculate_completion_cost(mock_completion, pricing)
        
        assert result['usd'] == 0.0
        assert result['vcu'] == 0.0


class TestCalculateEmbeddingCost:
    """Test cases for calculate_embedding_cost function."""
    
    def test_calculate_embedding_with_new_pricing(self):
        """Test embedding cost calculation with new pricing structure."""
        # Create mock embedding response
        mock_usage = Mock()
        mock_usage.total_tokens = 500
        
        mock_response = Mock()
        mock_response.usage = mock_usage
        
        # Create pricing (per million tokens)
        pricing: ModelPricing = {
            'input': {'usd': 100, 'vcu': 1000},
            'output': {'usd': 0.0, 'vcu': 0.0}  # Required by ModernPricing
        }
        
        result = calculate_embedding_cost(mock_response, pricing)
        
        # 500 tokens / 1_000_000 * 100 USD = 0.05 USD
        # 500 tokens / 1_000_000 * 1000 VCU = 0.5 VCU
        assert result['usd'] == pytest.approx(0.05)
        assert result['vcu'] == pytest.approx(0.5)
    
    def test_calculate_embedding_with_legacy_pricing(self):
        """Test embedding cost calculation with legacy pricing."""
        mock_usage = Mock()
        mock_usage.total_tokens = 1000
        
        mock_response = Mock()
        mock_response.usage = mock_usage
        
        pricing: ModelPricing = {
            'input': {'usd': 0.0, 'vcu': 0.0},  # Required by ModernPricing
            'output': {'usd': 0.0, 'vcu': 0.0}, # Required by ModernPricing
            'input_cost_per_mtok': 0.2
        }
        
        result = calculate_embedding_cost(mock_response, pricing)
        
        # 1000 tokens / 1,000,000 * 0.2 USD = 0.0002 USD
        assert result['usd'] == pytest.approx(0.0002)
        assert result['vcu'] == 0.0
    
    def test_calculate_embedding_with_missing_usage(self):
        """Test embedding calculation when usage is missing."""
        mock_response = Mock()
        mock_response.usage = None
        
        pricing: ModelPricing = {
            'input': {'usd': 0.1, 'vcu': 1.0},
            'output': {'usd': 0.0, 'vcu': 0.0}  # Required by ModernPricing
        }
        
        result = calculate_embedding_cost(mock_response, pricing)
        
        assert result['usd'] == 0.0
        assert result['vcu'] == 0.0


class TestEstimateCompletionCost:
    """Test cases for estimate_completion_cost function."""
    
    def test_estimate_with_defaults(self):
        """Test estimation with default parameters."""
        prompt = "This is a test prompt with ten words in it."
        
        pricing: ModelPricing = {
            'input': {'usd': 1.0, 'vcu': 10},
            'output': {'usd': 2.0, 'vcu': 20}
        }
        
        result = estimate_completion_cost(
            prompt=prompt,
            estimated_completion_tokens=100,  # Using default value
            model_pricing=pricing
        )
        
        # 10 words * 1.33 tokens/word = ~13 tokens
        # Default 100 completion tokens
        # Input: 13 / 1000 * 1.0 = 0.013 USD
        # Output: 100 / 1000 * 2.0 = 0.2 USD
        assert result['usd'] > 0
        assert result['vcu'] > 0
    
    def test_estimate_with_custom_parameters(self):
        """Test estimation with custom token parameters."""
        prompt = "Short prompt"  # 2 words
        
        pricing: ModelPricing = {
            'input': {'usd': 500, 'vcu': 5000},
            'output': {'usd': 1000, 'vcu': 10000}
        }
        
        result = estimate_completion_cost(
            prompt=prompt,
            model_pricing=pricing,
            estimated_completion_tokens=50,
            tokens_per_word=2.0
        )
        
        # 2 words * 2.0 tokens/word = 4 tokens
        # Input: 4 / 1_000_000 * 500 = 0.002 USD
        # Output: 50 / 1_000_000 * 1000 = 0.05 USD
        assert result['usd'] == pytest.approx(0.052)
        assert result['vcu'] == pytest.approx(0.52)
    
    def test_estimate_with_legacy_pricing(self):
        """Test estimation with legacy pricing structure."""
        prompt = "Test prompt"
        
        pricing: ModelPricing = {
            'input': {'usd': 0.0, 'vcu': 0.0},  # Required by ModernPricing
            'output': {'usd': 0.0, 'vcu': 0.0}, # Required by ModernPricing
            'input_cost_per_mtok': 0.3,
            'output_cost_per_mtok': 1.5
        }
        
        result = estimate_completion_cost(
            prompt=prompt,
            estimated_completion_tokens=100,  # Using default value
            model_pricing=pricing
        )
        
        assert result['usd'] > 0
        assert result['vcu'] == 0.0
    
    def test_estimate_with_empty_prompt(self):
        """Test estimation with empty prompt."""
        prompt = ""
        
        pricing: ModelPricing = {
            'input': {'usd': 1000, 'vcu': 10000},
            'output': {'usd': 2000, 'vcu': 20000}
        }
        
        result = estimate_completion_cost(
            prompt=prompt,
            estimated_completion_tokens=100,  # Using default value
            model_pricing=pricing
        )
        
        # Should still have output cost
        assert result['usd'] == pytest.approx(0.2)  # 100 tokens * 2000 / 1_000_000
        assert result['vcu'] == pytest.approx(2.0)   # 100 tokens * 20000 / 1_000_000
    
    def test_estimate_with_zero_completion_tokens(self):
        """Test estimation with zero completion tokens."""
        prompt = "Test prompt"
        
        pricing: ModelPricing = {
            'input': {'usd': 1.0, 'vcu': 10},
            'output': {'usd': 2.0, 'vcu': 20}
        }
        
        result = estimate_completion_cost(
            prompt=prompt,
            model_pricing=pricing,
            estimated_completion_tokens=0
        )
        
        # Should only have input cost
        assert result['usd'] > 0
        assert result['vcu'] > 0
        # But much less than with completion tokens
        assert result['usd'] < 0.1
        assert result['vcu'] < 1.0