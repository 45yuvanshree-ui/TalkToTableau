from __future__ import annotations

import pytest

from routing.model_catalog import OpenRouterModelMetadata
from routing.model_catalog_adapter import (
    metadata_to_candidate,
    metadata_to_candidates,
)
from routing.model_pool import ModelCapabilities
from routing.models import ModelTier


def make_metadata(
    *,
    model_id: str = "example/model",
    name: str = "Example Model",
    context_length: int | None = 32768,
    input_modalities: tuple[str, ...] = ("text",),
    output_modalities: tuple[str, ...] = ("text",),
    supported_parameters: tuple[str, ...] = (),
    prompt_price: float | None = 0.0,
    completion_price: float | None = 0.0,
    is_free: bool = True,
) -> OpenRouterModelMetadata:
    return OpenRouterModelMetadata(
        model_id=model_id,
        name=name,
        canonical_slug="example/model",
        context_length=context_length,
        input_modalities=input_modalities,
        output_modalities=output_modalities,
        supported_parameters=supported_parameters,
        prompt_price=prompt_price,
        completion_price=completion_price,
        is_free=is_free,
    )


def test_metadata_to_candidate_maps_basic_model_information() -> None:
    metadata = make_metadata(
        model_id="example/free-model",
        name="Example Free Model",
    )

    candidate = metadata_to_candidate(
        metadata,
        tier=ModelTier.BALANCED,
    )

    assert candidate.name == "example/free-model"
    assert candidate.provider == "openrouter"
    assert candidate.tier == ModelTier.BALANCED
    assert candidate.is_free is True


def test_metadata_to_candidate_maps_capabilities() -> None:
    metadata = make_metadata(
        supported_parameters=(
            "tools",
            "structured_outputs",
            "reasoning",
        )
    )

    candidate = metadata_to_candidate(
        metadata,
        tier=ModelTier.REASONING,
    )

    assert candidate.capabilities == ModelCapabilities(
        text_generation=True,
        structured_output=True,
        tool_calling=True,
        reasoning=True,
        sql_generation=True,
        long_context=True,
    )


def test_non_text_model_does_not_get_sql_generation() -> None:
    metadata = make_metadata(
        input_modalities=("text",),
        output_modalities=("image",),
    )

    candidate = metadata_to_candidate(
        metadata,
        tier=ModelTier.FAST,
    )

    assert candidate.capabilities.text_generation is False
    assert candidate.capabilities.sql_generation is False


def test_long_context_threshold_is_configurable() -> None:
    metadata = make_metadata(
        context_length=16000,
    )

    candidate = metadata_to_candidate(
        metadata,
        tier=ModelTier.BALANCED,
        long_context_threshold=8000,
    )

    assert candidate.capabilities.long_context is True


def test_short_context_is_not_long_context() -> None:
    metadata = make_metadata(
        context_length=16000,
    )

    candidate = metadata_to_candidate(
        metadata,
        tier=ModelTier.BALANCED,
        long_context_threshold=32000,
    )

    assert candidate.capabilities.long_context is False


def test_priority_is_preserved() -> None:
    metadata = make_metadata()

    candidate = metadata_to_candidate(
        metadata,
        tier=ModelTier.FAST,
        priority=7,
    )

    assert candidate.priority == 7


def test_metadata_is_preserved_for_tracing_and_diagnostics() -> None:
    metadata = make_metadata(
        context_length=65536,
        supported_parameters=("tools", "reasoning"),
    )

    candidate = metadata_to_candidate(
        metadata,
        tier=ModelTier.REASONING,
    )

    assert candidate.metadata["display_name"] == "Example Model"
    assert candidate.metadata["context_length"] == 65536
    assert candidate.metadata["supported_parameters"] == (
        "tools",
        "reasoning",
    )


def test_empty_model_id_is_rejected() -> None:
    metadata = make_metadata(
        model_id="",
    )

    with pytest.raises(ValueError):
        metadata_to_candidate(
            metadata,
            tier=ModelTier.FAST,
        )


def test_invalid_long_context_threshold_is_rejected() -> None:
    metadata = make_metadata()

    with pytest.raises(ValueError):
        metadata_to_candidate(
            metadata,
            tier=ModelTier.FAST,
            long_context_threshold=0,
        )


def test_multiple_metadata_items_are_converted() -> None:
    metadata_items = [
        make_metadata(model_id="example/model-a"),
        make_metadata(model_id="example/model-b"),
        make_metadata(model_id="example/model-c"),
    ]

    candidates = metadata_to_candidates(
        metadata_items,
        tier=ModelTier.BALANCED,
    )

    assert len(candidates) == 3
    assert [candidate.name for candidate in candidates] == [
        "example/model-a",
        "example/model-b",
        "example/model-c",
    ]