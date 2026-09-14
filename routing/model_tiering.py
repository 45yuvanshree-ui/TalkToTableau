from __future__ import annotations

from .model_catalog import OpenRouterModelMetadata
from .models import ModelTier


DEFAULT_BALANCED_CONTEXT_LENGTH = 32_000


class ModelTieringPolicy:
    """
    Deterministic policy for assigning a discovered model
    to a TalkToTableau routing tier.

    The policy does not execute models and does not measure
    model quality dynamically. It only uses provider metadata
    already available in OpenRouterModelMetadata.
    """

    def __init__(
        self,
        *,
        balanced_context_length: int = DEFAULT_BALANCED_CONTEXT_LENGTH,
    ) -> None:
        if balanced_context_length < 1:
            raise ValueError(
                "balanced_context_length must be >= 1."
            )

        self.balanced_context_length = balanced_context_length

    def assign_tier(
        self,
        model: OpenRouterModelMetadata,
    ) -> ModelTier:
        """
        Assign the most appropriate routing tier to one model.

        Priority:
            1. Reasoning-capable models -> REASONING
            2. Structured/tool-capable or long-context models -> BALANCED
            3. Remaining text models -> FAST
        """

        if model.supports_reasoning:
            return ModelTier.REASONING

        if (
            model.supports_structured_outputs
            or model.supports_tools
            or self._has_balanced_context(model)
        ):
            return ModelTier.BALANCED

        return ModelTier.FAST

    def _has_balanced_context(
        self,
        model: OpenRouterModelMetadata,
    ) -> bool:
        context_length = model.context_length

        if context_length is None:
            return False

        return context_length >= self.balanced_context_length