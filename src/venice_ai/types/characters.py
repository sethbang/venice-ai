from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel


class Character(BaseModel):
    """Represents an AI character definition in the Venice AI system.
    
    This model defines a complete AI character with all its attributes including
    metadata, configuration settings, and behavioral parameters. Characters are
    used in chat completions and other AI interactions to provide specific
    personalities, knowledge bases, and response styles.
    """
    slug: str
    name: str
    description: Optional[str] = None
    system_prompt: Optional[str] = None
    user_prompt: Optional[str] = None
    vision_enabled: Optional[bool] = False
    image_url: Optional[str] = None
    voice_id: Optional[str] = None
    category_tags: Optional[List[str]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class CharacterList(BaseModel):
    """Represents a paginated collection of AI characters.
    
    This model serves as a container for multiple character objects, typically
    returned by character listing and search API endpoints. It follows the
    standard API response format with a data array containing character objects
    and an object type identifier for response validation and parsing.
    """
    object: str
    data: List[Character]