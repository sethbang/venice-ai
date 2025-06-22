"""
Test suite for the get_model_pricing method in both sync and async clients.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from venice_ai import VeniceClient, AsyncVeniceClient
from venice_ai.types.models import ModelPricing, ModelList


class TestSyncClientGetModelPricing:
    """Test cases for VeniceClient.get_model_pricing method."""
    
    def test_get_model_pricing_success(self):
        """Test successful retrieval of model pricing."""
        # Create mock client
        client = VeniceClient(api_key="test-key")
        
        # Create mock model list response
        mock_model_data = {
            'id': 'llama-3.3-70b',
            'model_spec': {
                'pricing': {
                    'input': {'usd': 0.7, 'vcu': 7},
                    'output': {'usd': 2.8, 'vcu': 28}
                }
            }
        }
        
        mock_models_response = {
            'data': [mock_model_data],
            'object': 'list',
            'type': 'text'
        }
        
        # Mock the models.list() method
        with patch.object(client.models, 'list', return_value=mock_models_response):
            pricing = client.get_model_pricing('llama-3.3-70b')
            
            assert pricing == mock_model_data['model_spec']['pricing']
            assert pricing['input']['usd'] == 0.7
            assert pricing['input']['vcu'] == 7
            assert pricing['output']['usd'] == 2.8
            assert pricing['output']['vcu'] == 28
    
    def test_get_model_pricing_not_found(self):
        """Test error when model is not found."""
        client = VeniceClient(api_key="test-key")
        
        mock_models_response = {
            'data': [
                {
                    'id': 'other-model',
                    'model_spec': {
                        'pricing': {
                            'input': {'usd': 1.0, 'vcu': 10},
                            'output': {'usd': 2.0, 'vcu': 20}
                        }
                    }
                }
            ],
            'object': 'list',
            'type': 'text'
        }
        
        with patch.object(client.models, 'list', return_value=mock_models_response):
            with pytest.raises(ValueError) as exc_info:
                client.get_model_pricing('non-existent-model')
            
            assert "Model 'non-existent-model' not found" in str(exc_info.value)
    
    def test_get_model_pricing_multiple_models(self):
        """Test retrieval when multiple models are present."""
        client = VeniceClient(api_key="test-key")
        
        mock_models_response = {
            'data': [
                {
                    'id': 'model-1',
                    'model_spec': {
                        'pricing': {
                            'input': {'usd': 1.0, 'vcu': 10},
                            'output': {'usd': 2.0, 'vcu': 20}
                        }
                    }
                },
                {
                    'id': 'model-2',
                    'model_spec': {
                        'pricing': {
                            'input': {'usd': 0.5, 'vcu': 5},
                            'output': {'usd': 1.5, 'vcu': 15}
                        }
                    }
                },
                {
                    'id': 'model-3',
                    'model_spec': {
                        'pricing': {
                            'input': {'usd': 2.0, 'vcu': 20},
                            'output': {'usd': 4.0, 'vcu': 40}
                        }
                    }
                }
            ],
            'object': 'list',
            'type': 'text'
        }
        
        with patch.object(client.models, 'list', return_value=mock_models_response):
            # Should find the correct model
            pricing = client.get_model_pricing('model-2')
            assert pricing['input']['usd'] == 0.5
            assert pricing['input']['vcu'] == 5


class TestAsyncClientGetModelPricing:
    """Test cases for AsyncVeniceClient.get_model_pricing method."""
    
    @pytest.mark.asyncio
    async def test_async_get_model_pricing_success(self):
        """Test successful async retrieval of model pricing."""
        client = AsyncVeniceClient(api_key="test-key")
        
        mock_model_data = {
            'id': 'venice-uncensored',
            'model_spec': {
                'pricing': {
                    'input': {'usd': 0.5, 'vcu': 5},
                    'output': {'usd': 2.0, 'vcu': 20}
                }
            }
        }
        
        mock_models_response = {
            'data': [mock_model_data],
            'object': 'list',
            'type': 'text'
        }
        
        # Create async mock
        mock_list = AsyncMock(return_value=mock_models_response)
        
        with patch.object(client.models, 'list', mock_list):
            pricing = await client.get_model_pricing('venice-uncensored')
            
            assert pricing == mock_model_data['model_spec']['pricing']
            assert pricing['input']['usd'] == 0.5
            assert pricing['input']['vcu'] == 5
            assert pricing['output']['usd'] == 2.0
            assert pricing['output']['vcu'] == 20
        
        await client.close()
    
    @pytest.mark.asyncio
    async def test_async_get_model_pricing_not_found(self):
        """Test async error when model is not found."""
        client = AsyncVeniceClient(api_key="test-key")
        
        mock_models_response = {
            'data': [],
            'object': 'list',
            'type': 'text'
        }
        
        mock_list = AsyncMock(return_value=mock_models_response)
        
        with patch.object(client.models, 'list', mock_list):
            with pytest.raises(ValueError) as exc_info:
                await client.get_model_pricing('non-existent-model')
            
            assert "Model 'non-existent-model' not found" in str(exc_info.value)
        
        await client.close()
    
    @pytest.mark.asyncio
    async def test_async_get_model_pricing_with_legacy_pricing(self):
        """Test async retrieval of model with legacy pricing structure."""
        client = AsyncVeniceClient(api_key="test-key")
        
        # Model with legacy pricing structure
        mock_model_data = {
            'id': 'legacy-model',
            'model_spec': {
                'pricing': {
                    'input_cost_per_mtok': 0.3,
                    'output_cost_per_mtok': 1.5
                }
            }
        }
        
        mock_models_response = {
            'data': [mock_model_data],
            'object': 'list',
            'type': 'text'
        }
        
        mock_list = AsyncMock(return_value=mock_models_response)
        
        with patch.object(client.models, 'list', mock_list):
            pricing = await client.get_model_pricing('legacy-model')
            
            # Should return the pricing structure as-is
            assert pricing == mock_model_data['model_spec']['pricing']
            assert 'input_cost_per_mtok' in pricing
            assert 'output_cost_per_mtok' in pricing
        
        await client.close()


class TestIntegrationScenarios:
    """Test integration scenarios combining get_model_pricing with cost calculations."""
    
    def test_pricing_retrieval_and_cost_calculation(self):
        """Test full flow of getting pricing and calculating costs."""
        from venice_ai.costs import calculate_completion_cost
        
        client = VeniceClient(api_key="test-key")
        
        # Mock model pricing (per million tokens)
        mock_pricing = {
            'input': {'usd': 1000, 'vcu': 10000},
            'output': {'usd': 2000, 'vcu': 20000}
        }
        
        mock_models_response = {
            'data': [{
                'id': 'test-model',
                'model_spec': {'pricing': mock_pricing}
            }]
        }
        
        # Mock completion response
        mock_completion = Mock()
        mock_completion.usage = Mock()
        mock_completion.usage.prompt_tokens = 100
        mock_completion.usage.completion_tokens = 50
        
        with patch.object(client.models, 'list', return_value=mock_models_response):
            # Get pricing
            pricing = client.get_model_pricing('test-model')
            
            # Calculate costs
            costs = calculate_completion_cost(mock_completion, pricing)
            
            # Verify calculations
            expected_usd = (100 / 1_000_000 * 1000) + (50 / 1_000_000 * 2000)  # 0.1 + 0.1 = 0.2
            expected_vcu = (100 / 1_000_000 * 10000) + (50 / 1_000_000 * 20000)    # 1.0 + 1.0 = 2.0
            
            assert costs['usd'] == pytest.approx(expected_usd)
            assert costs['vcu'] == pytest.approx(expected_vcu)