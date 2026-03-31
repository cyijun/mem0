"""
Memory categorization utilities for OpenMemory.

This module provides memory categorization functionality using the configured
LLM provider instead of hardcoded OpenAI. Supports all LLM providers supported
by mem0 including OpenAI, Ollama, Anthropic, Groq, etc.
"""

import json
import logging
from typing import List

from app.utils.memory import get_memory_client_safe
from app.utils.prompts import MEMORY_CATEGORIZATION_PROMPT
from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


class MemoryCategories(BaseModel):
    """Schema for memory categorization response."""
    categories: List[str]


def _get_llm():
    """
    Get the configured LLM instance from the memory client.
    
    This respects all environment variables and DB config overrides,
    supporting all 17+ providers via LlmFactory.
    
    Returns:
        LLM instance or None if not available
    """
    memory_client = get_memory_client_safe()
    if not memory_client:
        return None
    return memory_client.llm


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=15),
    retry=retry_if_exception_type((Exception,)),
    reraise=True
)
def get_categories_for_memory(memory: str) -> List[str]:
    """
    Get categories for a memory using the configured LLM.
    
    Args:
        memory: The memory content to categorize
        
    Returns:
        List of category strings
        
    Raises:
        Exception: If categorization fails after retries
    """
    # Get the configured LLM
    llm = _get_llm()
    if not llm:
        logging.warning("[WARN] LLM not available for categorization, returning empty categories")
        return []
    
    try:
        messages = [
            {"role": "system", "content": MEMORY_CATEGORIZATION_PROMPT},
            {"role": "user", "content": memory}
        ]

        # Use the configured LLM with JSON response format
        # This works with all providers that support JSON mode
        response = llm.generate_response(
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0
        )

        # Parse the JSON response
        if not response:
            logging.warning("[WARN] Empty response from LLM for categorization")
            return []
        
        try:
            parsed_data = json.loads(response)
            categories_data = MemoryCategories(**parsed_data)
            return [cat.strip().lower() for cat in categories_data.categories]
        except (json.JSONDecodeError, ValidationError) as e:
            logging.error(f"[ERROR] Failed to parse categorization response: {e}")
            logging.debug(f"[DEBUG] Raw response: {response}")
            # Try to extract categories using fallback method
            return _fallback_extract_categories(response)

    except Exception as e:
        logging.error(f"[ERROR] Failed to get categories: {e}")
        raise


def _fallback_extract_categories(response: str) -> List[str]:
    """
    Fallback method to extract categories from malformed JSON or text response.
    
    Args:
        response: Raw LLM response string
        
    Returns:
        List of category strings
    """
    categories = []
    try:
        # Try to find a JSON-like structure in the response
        import re
        
        # Look for "categories" key with array value
        match = re.search(r'"categories"\s*:\s*(\[[^\]]*\])', response, re.IGNORECASE)
        if match:
            array_str = match.group(1)
            # Extract string values from array
            categories = re.findall(r'"([^"]*)"', array_str)
        else:
            # Try to find any quoted strings that might be categories
            categories = re.findall(r'"([^"]*)"', response)
            # Filter out common non-category words
            categories = [c for c in categories if c.lower() not in 
                         ['categories', 'category', 'json', 'text']]
        
        return [cat.strip().lower() for cat in categories if cat.strip()]
    except Exception as e:
        logging.error(f"[ERROR] Fallback categorization extraction failed: {e}")
        return []
