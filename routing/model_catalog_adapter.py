from __future__ import annotations

from collections.abc import Iterable

from .model_catalog import OpenRouterModelMetadata
from .model_pool import ModelCandidate, ModelCapabilities
from .models import ModelTier


DEFAULT_LONG_CONTEXT_THRESHOLD = 32_000


def metadata_to_candidate(
    metadata: OpenRouterModelMetadata,
    *,
    tier: ModelTier,
    priority: int = 0,
    long_context_threshold: int = DEFAULT_LONG_CONTEXT_THRESHOLD,
) -> ModelCandidate:
    """
    Convert discovered OpenRouter metadata into a ModelCandidate.

    The adapter only translates provider metadata into the application's
    internal representation. It does not execute the model and does not
    modify ModelPool state.

    Logical tier is supplied by the caller because provider metadata does
    not, by itself, establish the application's FAST/BALANCED/REASONING
    classification.
    """

    if not metadata.model_id.strip():
        raise ValueError(
            "Discovered model ID must not be empty."
        )

    if long_context_threshold < 1:
        raise ValueError(
            "long_context_threshold must be >= 1."
        )

    capabilities = ModelCapabilities(
        text_generation=metadata.is_text_model,
        structured_output=metadata.supports_structured_outputs,
        tool_calling=metadata.supports_tools,
        reasoning=metadata.supports_reasoning,
        sql_generation=metadata.is_text_model,
        long_context=(
            metadata.context_length is not None
            and metadata.context_length >= long_context_threshold
        ),
    )

    return ModelCandidate(
        name=metadata.model_id,
        provider="openrouter",
        tier=tier,
        is_free=metadata.is_free,
        capabilities=capabilities,
        priority=priority,
        metadata={
            "display_name": metadata.name,
            "canonical_slug": metadata.canonical_slug,
            "context_length": metadata.context_length,
            "input_modalities": metadata.input_modalities,
            "output_modalities": metadata.output_modalities,
            "supported_parameters": metadata.supported_parameters,
            "prompt_price": metadata.prompt_price,
            "completion_price": metadata.completion_price,
        },
    )


def metadata_to_candidates(
    metadata_items: Iterable[OpenRouterModelMetadata],
    *,
    tier: ModelTier,
    priority: int = 0,
    long_context_threshold: int = DEFAULT_LONG_CONTEXT_THRESHOLD,
) -> list[ModelCandidate]:
    """
    Convert multiple discovered models into ModelCandidate objects.

    Invalid/empty model IDs are rejected by metadata_to_candidate rather
    than silently creating unusable candidates.
    """

    return [
        metadata_to_candidate(
            metadata,
            tier=tier,
            priority=priority,
            long_context_threshold=long_context_threshold,
        )
        for metadata in metadata_items
    ]