from __future__ import annotations

import pytest

from routing.model_catalog import OpenRouterModelMetadata
from routing.model_tiering import (
    DEFAULT_BALANCED_CONTEXT_LENGTH,
    ModelTieringPolicy,
)
from routing.models import ModelTier


def make_metadata(
    model_id: str = "test-model",
    *,
    context_length: int | None = 16_000,
    supported_parameters: tuple[str, ...] = (),
) -> OpenRouterModelMetadata:
    return OpenRouterModelMetadata(
        model_id=model_id,
        name=model_id,
        canonical_slug=model_id,
        context_length=context_length,
        input_modalities=("text",),
        output_modalities=("text",),
        supported_parameters=supported_parameters,
        prompt_price=0.0,
        completion_price=0.0,
        is_free=True,
        raw={},
    )


def test_reasoning_model_is_assigned_reasoning_tier() -> None:
    policy = ModelTieringPolicy()

    model = make_metadata(
        "reasoning-model:free",
        supported_parameters=("reasoning",),
    )

    assert policy.assign_tier(model) == ModelTier.REASONING


def test_include_reasoning_parameter_is_assigned_reasoning_tier() -> None:
    policy = ModelTieringPolicy()

    model = make_metadata(
        "reasoning-model:free",
        supported_parameters=("include_reasoning",),
    )

    assert policy.assign_tier(model) == ModelTier.REASONING


def test_structured_output_model_is_assigned_balanced_tier() -> None:
    policy = ModelTieringPolicy()

    model = make_metadata(
        "structured-model:free",
        supported_parameters=("structured_outputs",),
    )

    assert policy.assign_tier(model) == ModelTier.BALANCED


def test_tool_calling_model_is_assigned_balanced_tier() -> None:
    policy = ModelTieringPolicy()

    model = make_metadata(
        "tool-model:free",
        supported_parameters=("tools",),
    )

    assert policy.assign_tier(model) == ModelTier.BALANCED


def test_long_context_model_is_assigned_balanced_tier() -> None:
    policy = ModelTieringPolicy()

    model = make_metadata(
        "long-context-model:free",
        context_length=64_000,
    )

    assert policy.assign_tier(model) == ModelTier.BALANCED


def test_exact_balanced_context_threshold_is_balanced() -> None:
    policy = ModelTieringPolicy()

    model = make_metadata(
        "threshold-model:free",
        context_length=DEFAULT_BALANCED_CONTEXT_LENGTH,
    )

    assert policy.assign_tier(model) == ModelTier.BALANCED


def test_context_below_balanced_threshold_is_fast() -> None:
    policy = ModelTieringPolicy()

    model = make_metadata(
        "short-context-model:free",
        context_length=DEFAULT_BALANCED_CONTEXT_LENGTH - 1,
    )

    assert policy.assign_tier(model) == ModelTier.FAST


def test_model_without_context_metadata_can_still_be_fast() -> None:
    policy = ModelTieringPolicy()

    model = make_metadata(
        "unknown-context-model:free",
        context_length=None,
    )

    assert policy.assign_tier(model) == ModelTier.FAST


def test_plain_text_model_is_assigned_fast_tier() -> None:
    policy = ModelTieringPolicy()

    model = make_metadata(
        "simple-model:free",
        context_length=16_000,
    )

    assert policy.assign_tier(model) == ModelTier.FAST


def test_reasoning_tier_has_priority_over_balanced_capabilities() -> None:
    policy = ModelTieringPolicy()

    model = make_metadata(
        "advanced-model:free",
        context_length=128_000,
        supported_parameters=(
            "reasoning",
            "structured_outputs",
            "tools",
        ),
    )

    assert policy.assign_tier(model) == ModelTier.REASONING


def test_custom_context_threshold_is_respected() -> None:
    policy = ModelTieringPolicy(
        balanced_context_length=16_000,
    )

    model = make_metadata(
        "custom-threshold-model:free",
        context_length=16_000,
    )

    assert policy.assign_tier(model) == ModelTier.BALANCED


def test_invalid_context_threshold_is_rejected() -> None:
    with pytest.raises(ValueError, match="balanced_context_length"):
        ModelTieringPolicy(
            balanced_context_length=0,
        )


def test_negative_context_threshold_is_rejected() -> None:
    with pytest.raises(ValueError, match="balanced_context_length"):
        ModelTieringPolicy(
            balanced_context_length=-1,
        )