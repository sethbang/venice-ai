from typing import List, Optional, Literal

from pydantic import BaseModel, Field

class Stats(BaseModel):
    """Represents statistical information for an AI character.
    
    This model contains metrics and usage data associated with a character,
    such as the number of times the character has been imported or used.
    Used as a nested component within character definitions to provide
    analytics and usage insights.
    """
    imports: int = Field(..., description="Number of times the character has been imported or used")

class Character(BaseModel):
    """Represents an AI character definition in the Venice AI system.
    
    This model defines a complete AI character with all its attributes including
    metadata, configuration settings, and behavioral parameters. Characters are
    used in chat completions and other AI interactions to provide specific
    personalities, knowledge bases, and response styles. The character includes
    essential identifiers like name and slug, content classification flags,
    descriptive information, sharing capabilities, and usage statistics.
    """
    adult: bool = Field(..., description="Whether the character is considered adult content")
    createdAt: str = Field(..., description="Date when the character was created")
    description: Optional[str] = Field(None, description="Description of the character")
    name: str = Field(..., description="Name of the character")
    shareUrl: Optional[str] = Field(None, description="Share URL of the character")
    slug: str = Field(..., description="Slug of the character to be used in the completions API")
    stats: Stats = Field(..., description="Statistical information and usage metrics for the character")
    tags: List[str] = Field(..., description="Tags associated with the character")
    updatedAt: str = Field(..., description="Date when the character was last updated")
    webEnabled: bool = Field(..., description="Whether the character is enabled for web use")

class CharacterList(BaseModel):
    """Represents a paginated collection of AI characters.
    
    This model serves as a container for multiple character objects, typically
    returned by character listing and search API endpoints. It follows the
    standard API response format with a data array containing character objects
    and an object type identifier for response validation and parsing.
    """
    data: List[Character] = Field(..., description="List of character objects")
    object: Literal["list"] = Field(..., description="Object type identifier, always 'list' for character collections")