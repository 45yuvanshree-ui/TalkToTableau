from __future__ import annotations

from routing.model_catalog import OpenRouterModelMetadata
from routing.model_eligibility import (
    ModelEligibilityPolicy,
    ModelEligibilityRequest,
)


def make_metadata(
    *,
    model_id: str = "example/model",
    context_length: int | None = 32768,
    input_modalities: tuple[str, ...] = ("text",),
    output_modalities: tuple[str, ...] = ("text",),
    supported_parameters: tuple[str, ...] = (),
    is_free: bool = True,
) -> OpenRouterModelMetadata:
    return OpenRouterModelMetadata(
        model_id=model_id,
        name=model_id,
        canonical_slug=model_id,
        context_length=context_length,
        input_modalities=input_modalities,
        output_modalities=output_modalities,
        supported_parameters=supported_parameters,
        prompt_price=0.0 if is_free else 0.000001,
        completion_price=0.0 if is_free else 0.000001,
        is_free=is_free,
    )


def test_text_sql_model_is_eligible() -> None:
    policy = ModelEligibilityPolicy()

    model = make_metadata()

    result = policy.evaluate(
        model,
        ModelEligibilityRequest(),
    )

    assert result.eligible is True


def test_image_output_model_is_rejected() -> None:
    policy = ModelEligibilityPolicy()

    model = make_metadata(
        output_modalities=("image",),
    )

    result = policy.evaluate(
        model,
        ModelEligibilityRequest(),
    )

    assert result.eligible is False
    assert (
        "model does not support text-to-text generation"
        in result.reasons
    )


def test_non_text_input_model_is_rejected() -> None:
    policy = ModelEligibilityPolicy()

    model = make_metadata(
        input_modalities=("image",),
        output_modalities=("text",),
    )

    result = policy.evaluate(
        model,
        ModelEligibilityRequest(),
    )

    assert result.eligible is False


def test_free_only_requirement_rejects_paid_model() -> None:
    policy = ModelEligibilityPolicy()

    model = make_metadata(
        is_free=False,
    )

    result = policy.evaluate(
        model,
        ModelEligibilityRequest(
            require_free=True,
        ),
    )

    assert result.eligible is False
    assert result.reasons == (
        "free model required",
    )


def test_free_only_requirement_accepts_free_model() -> None:
    policy = ModelEligibilityPolicy()

    model = make_metadata(
        is_free=True,
    )

    result = policy.evaluate(
        model,
        ModelEligibilityRequest(
            require_free=True,
        ),
    )

    assert result.eligible is True
    assert "model is free" in result.reasons


def test_reasoning_requirement_rejects_model_without_reasoning() -> None:
    policy = ModelEligibilityPolicy()

    model = make_metadata(
        supported_parameters=(),
    )

    result = policy.evaluate(
        model,
        ModelEligibilityRequest(
            require_reasoning=True,
        ),
    )

    assert result.eligible is False


def test_reasoning_requirement_accepts_reasoning_model() -> None:
    policy = ModelEligibilityPolicy()

    model = make_metadata(
        supported_parameters=("reasoning",),
    )

    result = policy.evaluate(
        model,
        ModelEligibilityRequest(
            require_reasoning=True,
        ),
    )

    assert result.eligible is True
    assert "reasoning capability supported" in result.reasons


def test_structured_output_requirement_is_enforced() -> None:
    policy = ModelEligibilityPolicy()

    model = make_metadata(
        supported_parameters=(),
    )

    result = policy.evaluate(
        model,
        ModelEligibilityRequest(
            require_structured_output=True,
        ),
    )

    assert result.eligible is False


def test_structured_output_requirement_accepts_capable_model() -> None:
    policy = ModelEligibilityPolicy()

    model = make_metadata(
        supported_parameters=("structured_outputs",),
    )

    result = policy.evaluate(
        model,
        ModelEligibilityRequest(
            require_structured_output=True,
        ),
    )

    assert result.eligible is True


def test_tool_calling_requirement_is_enforced() -> None:
    policy = ModelEligibilityPolicy()

    model = make_metadata(
        supported_parameters=(),
    )

    result = policy.evaluate(
        model,
        ModelEligibilityRequest(
            require_tool_calling=True,
        ),
    )

    assert result.eligible is False


def test_tool_calling_requirement_accepts_capable_model() -> None:
    policy = ModelEligibilityPolicy()

    model = make_metadata(
        supported_parameters=("tools",),
    )

    result = policy.evaluate(
        model,
        ModelEligibilityRequest(
            require_tool_calling=True,
        ),
    )

    assert result.eligible is True


def test_minimum_context_length_is_enforced() -> None:
    policy = ModelEligibilityPolicy()

    model = make_metadata(
        context_length=16000,
    )

    result = policy.evaluate(
        model,
        ModelEligibilityRequest(
            minimum_context_length=32000,
        ),
    )

    assert result.eligible is False


def test_minimum_context_length_accepts_sufficient_model() -> None:
    policy = ModelEligibilityPolicy()

    model = make_metadata(
        context_length=65536,
    )

    result = policy.evaluate(
        model,
        ModelEligibilityRequest(
            minimum_context_length=32000,
        ),
    )

    assert result.eligible is True


def test_missing_context_length_is_rejected_when_required() -> None:
    policy = ModelEligibilityPolicy()

    model = make_metadata(
        context_length=None,
    )

    result = policy.evaluate(
        model,
        ModelEligibilityRequest(
            minimum_context_length=32000,
        ),
    )

    assert result.eligible is False


def test_invalid_context_requirement_is_rejected() -> None:
    policy = ModelEligibilityPolicy()

    model = make_metadata()

    try:
        policy.evaluate(
            model,
            ModelEligibilityRequest(
                minimum_context_length=0,
            ),
        )
    except ValueError:
        return

    raise AssertionError(
        "Expected ValueError for invalid context requirement."
    )


def test_filter_models_returns_only_eligible_models() -> None:
    policy = ModelEligibilityPolicy()

    models = [
        make_metadata(
            model_id="example/free-text",
        ),
        make_metadata(
            model_id="example/image",
            output_modalities=("image",),
        ),
        make_metadata(
            model_id="example/paid",
            is_free=False,
        ),
    ]

    result = policy.filter_models(
        models,
        ModelEligibilityRequest(
            require_free=True,
        ),
    )

    assert [
        model.model_id
        for model in result
    ] == [
        "example/free-text",
    ]


def test_filter_unique_models_excludes_configured_models() -> None:
    policy = ModelEligibilityPolicy()

    models = [
        make_metadata(
            model_id="example/configured",
        ),
        make_metadata(
            model_id="example/discovered",
        ),
        make_metadata(
            model_id="example/discovered",
        ),
    ]

    result = policy.filter_unique_models(
        models,
        ModelEligibilityRequest(),
        excluded_model_ids=[
            "example/configured",
        ],
    )

    assert [
        model.model_id
        for model in result
    ] == [
        "example/discovered",
    ]


def test_filter_unique_models_strips_excluded_ids() -> None:
    policy = ModelEligibilityPolicy()

    models = [
        make_metadata(
            model_id="example/configured",
        ),
        make_metadata(
            model_id="example/new",
        ),
    ]

    result = policy.filter_unique_models(
        models,
        ModelEligibilityRequest(),
        excluded_model_ids=[
            "  example/configured  ",
        ],
    )

    assert [
        model.model_id
        for model in result
    ] == [
        "example/new",
    ]