"""
Cost calculation utilities for Venice AI API usage.

This module provides functions to calculate the cost of API usage based on
token consumption and model pricing information. Supports both USD and VCU
(Venice Compute Units) calculations.
"""

from typing import Dict, Optional, cast, Any
from .types.chat import ChatCompletion, UsageData
from .types.models import ModelPricing


def calculate_completion_cost(
    completion: ChatCompletion,
    model_pricing: Optional[ModelPricing]
) -> Dict[str, float]:
    """
    Calculate the cost of a chat completion in both USD and VCU.
    
    Takes a ChatCompletion response and the pricing information for the model
    used, then calculates the total cost based on token usage.
    
    :param completion: The chat completion response containing usage data
    :type completion: ChatCompletion
    :param model_pricing: The pricing information for the model used
    :type model_pricing: ModelPricing
    :return: Dictionary containing 'usd' and 'vcu' costs
    :rtype: Dict[str, float]
    :raises ValueError: If the completion has no usage data
    
    Example:
        >>> from venice_ai import VeniceClient
        >>> client = VeniceClient(api_key="your-api-key")
        >>>
        >>> # Get a chat completion
        >>> completion = client.chat.completions.create(
        ...     model="llama-3.3-70b",
        ...     messages=[{"role": "user", "content": "Hello!"}]
        ... )
        >>>
        >>> # Get model pricing
        >>> model_pricing = client.get_model_pricing("llama-3.3-70b")
        >>>
        >>> # Calculate costs
        >>> costs = calculate_completion_cost(completion, model_pricing)
        >>> print(f"Cost: ${costs['usd']:.6f} USD, {costs['vcu']:.2f} VCU")
    """
    if not model_pricing or not hasattr(completion, 'usage') or not completion.usage:
        return {'usd': 0.0, 'vcu': 0.0}
    
    # Extract token counts
    prompt_tokens = completion.usage.prompt_tokens
    completion_tokens = completion.usage.completion_tokens
    
    # Initialize costs
    usd_cost = 0.0
    vcu_cost = 0.0
    
    # Handle new pricing structure first
    if 'input' in model_pricing and 'output' in model_pricing:
        input_pricing = model_pricing['input']
        output_pricing = model_pricing['output']
        
        # Check if new pricing has non-zero values
        has_new_pricing = False
        
        if 'usd' in input_pricing and input_pricing['usd'] > 0:
            # Assuming input_pricing['usd'] is cost PER MILLION TOKENS
            usd_cost += (prompt_tokens / 1_000_000) * input_pricing['usd']
            has_new_pricing = True
        if 'vcu' in input_pricing and input_pricing['vcu'] > 0:
            # Assuming input_pricing['vcu'] is cost PER MILLION TOKENS
            vcu_cost += (prompt_tokens / 1_000_000) * input_pricing['vcu']
            has_new_pricing = True
        
        if 'usd' in output_pricing and output_pricing['usd'] > 0:
            # Assuming output_pricing['usd'] is cost PER MILLION TOKENS
            usd_cost += (completion_tokens / 1_000_000) * output_pricing['usd']
            has_new_pricing = True
        if 'vcu' in output_pricing and output_pricing['vcu'] > 0:
            # Assuming output_pricing['vcu'] is cost PER MILLION TOKENS
            vcu_cost += (completion_tokens / 1_000_000) * output_pricing['vcu']
            has_new_pricing = True
        
        # If new pricing structure has all zero values, fall back to legacy
        if not has_new_pricing:
            # Fallback to legacy pricing fields
            if 'input_cost_per_mtok' in model_pricing:
                usd_cost += (prompt_tokens / 1_000_000) * model_pricing['input_cost_per_mtok']
            if 'output_cost_per_mtok' in model_pricing:
                usd_cost += (completion_tokens / 1_000_000) * model_pricing['output_cost_per_mtok']
            
            if 'input_cost_per_mtok_vcu' in model_pricing:
                vcu_cost += (prompt_tokens / 1_000_000) * model_pricing['input_cost_per_mtok_vcu']
            if 'output_cost_per_mtok_vcu' in model_pricing:
                vcu_cost += (completion_tokens / 1_000_000) * model_pricing['output_cost_per_mtok_vcu']
    else:
        # Fallback to legacy pricing fields
        if 'input_cost_per_mtok' in model_pricing:
            usd_cost += (prompt_tokens / 1_000_000) * model_pricing['input_cost_per_mtok']
        if 'output_cost_per_mtok' in model_pricing:
            usd_cost += (completion_tokens / 1_000_000) * model_pricing['output_cost_per_mtok']
        
        if 'input_cost_per_mtok_vcu' in model_pricing:
            vcu_cost += (prompt_tokens / 1_000_000) * model_pricing['input_cost_per_mtok_vcu']
        if 'output_cost_per_mtok_vcu' in model_pricing:
            vcu_cost += (completion_tokens / 1_000_000) * model_pricing['output_cost_per_mtok_vcu']
            
    return {
        'usd': usd_cost,
        'vcu': vcu_cost
    }


def calculate_embedding_cost(
    embedding_response: Any,
    model_pricing: Optional[ModelPricing]
) -> Dict[str, float]:
    """
    Calculate the cost of an embedding request in both USD and VCU.
    
    :param embedding_response: The embedding response containing usage data
    :type embedding_response: Any
    :param model_pricing: The pricing information for the model used
    :type model_pricing: ModelPricing
    :return: Dictionary containing 'usd' and 'vcu' costs
    :rtype: Dict[str, float]
    
    Example:
        >>> from venice_ai import VeniceClient
        >>> client = VeniceClient(api_key="your-api-key")
        >>>
        >>> # Get embeddings
        >>> response = client.embeddings.create(
        ...     model="text-embedding-3-small",
        ...     input="Hello, world!"
        ... )
        >>>
        >>> # Get model pricing
        >>> model_pricing = client.get_model_pricing("text-embedding-3-small")
        >>>
        >>> # Calculate costs
        >>> costs = calculate_embedding_cost(response, model_pricing)
        >>> print(f"Cost: ${costs['usd']:.6f} USD, {costs['vcu']:.2f} VCU")
    """
    if not model_pricing or not hasattr(embedding_response, 'usage') or not embedding_response.usage:
        return {'usd': 0.0, 'vcu': 0.0}
    
    total_tokens = embedding_response.usage.total_tokens
    usd_cost = 0.0
    vcu_cost = 0.0
    
    # Handle new pricing structure first
    if 'input' in model_pricing:
        input_pricing = model_pricing['input']
        
        # Check if new pricing has non-zero values
        has_new_pricing = False
        
        if 'usd' in input_pricing and input_pricing['usd'] > 0:
            # Assuming input_pricing['usd'] is cost PER MILLION TOKENS
            usd_cost = (total_tokens / 1_000_000) * input_pricing['usd']
            has_new_pricing = True
        if 'vcu' in input_pricing and input_pricing['vcu'] > 0:
            # Assuming input_pricing['vcu'] is cost PER MILLION TOKENS
            vcu_cost = (total_tokens / 1_000_000) * input_pricing['vcu']
            has_new_pricing = True
        
        # If new pricing structure has all zero values, fall back to legacy
        if not has_new_pricing:
            # Fallback to legacy pricing fields
            if 'input_cost_per_mtok' in model_pricing:
                usd_cost = (total_tokens / 1_000_000) * model_pricing['input_cost_per_mtok']
            if 'input_cost_per_mtok_vcu' in model_pricing:
                vcu_cost = (total_tokens / 1_000_000) * model_pricing['input_cost_per_mtok_vcu']
    else:
        # Fallback to legacy pricing fields
        if 'input_cost_per_mtok' in model_pricing:
            usd_cost = (total_tokens / 1_000_000) * model_pricing['input_cost_per_mtok']
        if 'input_cost_per_mtok_vcu' in model_pricing:
            vcu_cost = (total_tokens / 1_000_000) * model_pricing['input_cost_per_mtok_vcu']
            
    return {
        'usd': usd_cost,
        'vcu': vcu_cost
    }


def estimate_completion_cost(
    prompt: str,
    estimated_completion_tokens: int,
    model_pricing: Optional[ModelPricing],
    tokens_per_word: float = 1.3
) -> Dict[str, float]:
    """
    Estimate the cost of a chat completion before making the request.
    
    This function provides a rough estimate based on word count and expected
    completion length. The actual cost may vary depending on tokenization.
    
    :param prompt: The prompt text to estimate tokens for
    :type prompt: str
    :param estimated_completion_tokens: Expected number of completion tokens
    :type estimated_completion_tokens: int
    :param model_pricing: The pricing information for the model
    :type model_pricing: ModelPricing
    :param tokens_per_word: Average tokens per word (default: 1.3)
    :type tokens_per_word: float
    :return: Dictionary containing estimated 'usd' and 'vcu' costs
    :rtype: Dict[str, float]
    
    Example:
        >>> from venice_ai import VeniceClient
        >>> client = VeniceClient(api_key="your-api-key")
        >>> 
        >>> # Get model pricing
        >>> model_pricing = client.get_model_pricing("llama-3.3-70b")
        >>> 
        >>> # Estimate costs
        >>> prompt = "Write a detailed explanation of quantum computing"
        >>> estimated_cost = estimate_completion_cost(
        ...     prompt=prompt,
        ...     estimated_completion_tokens=500,
        ...     model_pricing=model_pricing
        ... )
        >>> print(f"Estimated cost: ${estimated_cost['usd']:.6f} USD")
    """
    if not model_pricing:
        return {'usd': 0.0, 'vcu': 0.0}
        
    # Estimate prompt tokens based on word count
    word_count = len(prompt.split())
    estimated_prompt_tokens = int(word_count * tokens_per_word)
    
    # Initialize costs
    usd_cost = 0.0
    vcu_cost = 0.0
    
    # Handle new pricing structure first
    if 'input' in model_pricing and 'output' in model_pricing:
        input_pricing = model_pricing['input']
        output_pricing = model_pricing['output']
        
        # Check if new pricing has non-zero values
        has_new_pricing = False
        
        if 'usd' in input_pricing and input_pricing['usd'] > 0:
            # Assuming input_pricing['usd'] is cost PER MILLION TOKENS
            usd_cost += (estimated_prompt_tokens / 1_000_000) * input_pricing['usd']
            has_new_pricing = True
        if 'vcu' in input_pricing and input_pricing['vcu'] > 0:
            # Assuming input_pricing['vcu'] is cost PER MILLION TOKENS
            vcu_cost += (estimated_prompt_tokens / 1_000_000) * input_pricing['vcu']
            has_new_pricing = True
        
        if 'usd' in output_pricing and output_pricing['usd'] > 0:
            # Assuming output_pricing['usd'] is cost PER MILLION TOKENS
            usd_cost += (estimated_completion_tokens / 1_000_000) * output_pricing['usd']
            has_new_pricing = True
        if 'vcu' in output_pricing and output_pricing['vcu'] > 0:
            # Assuming output_pricing['vcu'] is cost PER MILLION TOKENS
            vcu_cost += (estimated_completion_tokens / 1_000_000) * output_pricing['vcu']
            has_new_pricing = True
        
        # If new pricing structure has all zero values, fall back to legacy
        if not has_new_pricing:
            # Fallback to legacy pricing fields
            if 'input_cost_per_mtok' in model_pricing:
                usd_cost += (estimated_prompt_tokens / 1_000_000) * model_pricing['input_cost_per_mtok']
            if 'output_cost_per_mtok' in model_pricing:
                usd_cost += (estimated_completion_tokens / 1_000_000) * model_pricing['output_cost_per_mtok']
                
            if 'input_cost_per_mtok_vcu' in model_pricing:
                vcu_cost += (estimated_prompt_tokens / 1_000_000) * model_pricing['input_cost_per_mtok_vcu']
            if 'output_cost_per_mtok_vcu' in model_pricing:
                vcu_cost += (estimated_completion_tokens / 1_000_000) * model_pricing['output_cost_per_mtok_vcu']
    else:
        # Fallback to legacy pricing fields
        if 'input_cost_per_mtok' in model_pricing:
            usd_cost += (estimated_prompt_tokens / 1_000_000) * model_pricing['input_cost_per_mtok']
        if 'output_cost_per_mtok' in model_pricing:
            usd_cost += (estimated_completion_tokens / 1_000_000) * model_pricing['output_cost_per_mtok']
            
        if 'input_cost_per_mtok_vcu' in model_pricing:
            vcu_cost += (estimated_prompt_tokens / 1_000_000) * model_pricing['input_cost_per_mtok_vcu']
        if 'output_cost_per_mtok_vcu' in model_pricing:
            vcu_cost += (estimated_completion_tokens / 1_000_000) * model_pricing['output_cost_per_mtok_vcu']
            
    return {
        'usd': usd_cost,
        'vcu': vcu_cost
    }