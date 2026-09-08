from enum import Enum

from pydantic import BaseModel, Field


class ModelTier(str, Enum):
    """Logical model tiers used by the adaptive router."""

    FAST = "fast"
    BALANCED = "balanced"
    REASONING = "reasoning"


class RoutingRequest(BaseModel):
    """Information used to select an appropriate model tier."""

    user_request: str

    sql_complexity: int = Field(
        default=1,
        ge=1,
        le=5,
    )

    ambiguity: int = Field(
        default=1,
        ge=1,
        le=5,
    )

    reasoning_required: bool = False

    output_complexity: int = Field(
        default=1,
        ge=1,
        le=5,
    )

    schema_size: int = Field(
        default=1,
        ge=1,
    )


class RoutingDecision(BaseModel):
    """Model-routing decision produced by the adaptive router."""

    tier: ModelTier

    reason: str

    complexity_score: int = Field(
        ge=1,
        le=100,
    )