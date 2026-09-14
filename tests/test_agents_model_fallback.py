from __future__ import annotations

import pytest

from routing import (
    AgentsModelFactory,
    ModelTier,
)
from routing.fallback import (
    FallbackExhaustedError,
    LLMFallbackExecutor,
    ProviderFailure,
)


class FakeProviderError(Exception):
    """Synthetic provider failure used for testing."""

    status_code = 502


class FakeRateLimitError(Exception):
    """Synthetic rate-limit failure used for testing."""

    status_code = 429


class FakeTimeoutError(Exception):
    """Synthetic timeout failure used for testing."""

    pass


def test_run_with_fallback_uses_primary_model_first(
    monkeypatch,
):
    factory = AgentsModelFactory()

    created_models = []
    executed_attempts = []

    def fake_create_model(
        *,
        model_name: str,
    ):
        created_models.append(
            model_name
        )

        return f"MODEL::{model_name}"

    monkeypatch.setattr(
        factory,
        "_create_model_from_name",
        fake_create_model,
    )

    monkeypatch.setattr(
        factory.provider_config,
        "get_model",
        lambda tier: {
            ModelTier.FAST: "fast-model",
            ModelTier.BALANCED: "balanced-model",
            ModelTier.REASONING: "reasoning-model",
        }[tier],
    )

    def fake_execute(
        model,
        tier,
        model_name,
    ):
        executed_attempts.append(
            (
                model,
                tier,
                model_name,
            )
        )

        return "success"

    result = factory.run_with_fallback(
        primary_tier=ModelTier.BALANCED,
        fallback_tiers=[
            ModelTier.FAST,
            ModelTier.REASONING,
        ],
        execute=fake_execute,
    )

    assert result.result == "success"

    assert created_models == [
        "balanced-model",
    ]

    assert len(
        executed_attempts
    ) == 1

    assert (
        executed_attempts[0][1]
        == ModelTier.BALANCED
    )

    assert (
        executed_attempts[0][2]
        == "balanced-model"
    )

    assert len(
        result.attempts
    ) == 1

    assert (
        result.attempts[0].success
        is True
    )


def test_run_with_fallback_switches_after_502(
    monkeypatch,
):
    factory = AgentsModelFactory()

    created_models = []
    executed_models = []

    monkeypatch.setattr(
        factory,
        "_create_model_from_name",
        lambda *, model_name: (
            created_models.append(
                model_name
            )
            or f"MODEL::{model_name}"
        ),
    )

    monkeypatch.setattr(
        factory.provider_config,
        "get_model",
        lambda tier: {
            ModelTier.FAST: "fast-model",
            ModelTier.BALANCED: "balanced-model",
        }[tier],
    )

    def fake_execute(
        _model,
        tier,
        model_name,
    ):
        executed_models.append(
            (
                tier,
                model_name,
            )
        )

        if model_name == "balanced-model":
            raise FakeProviderError(
                "upstream provider failure"
            )

        return "fallback-success"

    result = factory.run_with_fallback(
        primary_tier=ModelTier.BALANCED,
        fallback_tiers=[
            ModelTier.FAST,
        ],
        execute=fake_execute,
    )

    assert result.result == (
        "fallback-success"
    )

    assert created_models == [
        "balanced-model",
        "fast-model",
    ]

    assert executed_models == [
        (
            ModelTier.BALANCED,
            "balanced-model",
        ),
        (
            ModelTier.FAST,
            "fast-model",
        ),
    ]

    assert len(
        result.attempts
    ) == 2

    assert (
        result.attempts[0].success
        is False
    )

    assert (
        result.attempts[0].model_name
        == "balanced-model"
    )

    assert (
        result.attempts[1].success
        is True
    )

    assert (
        result.attempts[1].model_name
        == "fast-model"
    )


def test_run_with_fallback_switches_after_429(
    monkeypatch,
):
    factory = AgentsModelFactory()

    monkeypatch.setattr(
        factory,
        "_create_model_from_name",
        lambda *, model_name: (
            f"MODEL::{model_name}"
        ),
    )

    monkeypatch.setattr(
        factory.provider_config,
        "get_model",
        lambda tier: {
            ModelTier.FAST: "fast-model",
            ModelTier.BALANCED: "balanced-model",
        }[tier],
    )

    call_count = 0

    def fake_execute(
        _model,
        _tier,
        model_name,
    ):
        nonlocal call_count

        call_count += 1

        if model_name == "balanced-model":
            raise FakeRateLimitError(
                "too many requests"
            )

        return "success-after-rate-limit"

    result = factory.run_with_fallback(
        primary_tier=ModelTier.BALANCED,
        fallback_tiers=[
            ModelTier.FAST,
        ],
        execute=fake_execute,
    )

    assert result.result == (
        "success-after-rate-limit"
    )

    assert call_count == 2


def test_run_with_fallback_switches_after_timeout(
    monkeypatch,
):
    factory = AgentsModelFactory()

    monkeypatch.setattr(
        factory,
        "_create_model_from_name",
        lambda *, model_name: (
            f"MODEL::{model_name}"
        ),
    )

    monkeypatch.setattr(
        factory.provider_config,
        "get_model",
        lambda tier: {
            ModelTier.FAST: "fast-model",
            ModelTier.BALANCED: "balanced-model",
        }[tier],
    )

    def fake_execute(
        _model,
        _tier,
        model_name,
    ):
        if model_name == "balanced-model":
            raise FakeTimeoutError(
                "request timed out"
            )

        return "success-after-timeout"

    result = factory.run_with_fallback(
        primary_tier=ModelTier.BALANCED,
        fallback_tiers=[
            ModelTier.FAST,
        ],
        execute=fake_execute,
    )

    assert result.result == (
        "success-after-timeout"
    )

    assert len(
        result.attempts
    ) == 2


def test_run_with_fallback_does_not_fallback_on_invalid_json(
    monkeypatch,
):
    factory = AgentsModelFactory()

    created_models = []

    monkeypatch.setattr(
        factory,
        "_create_model_from_name",
        lambda *, model_name: (
            created_models.append(
                model_name
            )
            or f"MODEL::{model_name}"
        ),
    )

    monkeypatch.setattr(
        factory.provider_config,
        "get_model",
        lambda tier: {
            ModelTier.FAST: "fast-model",
            ModelTier.BALANCED: "balanced-model",
        }[tier],
    )

    def fake_execute(
        _model,
        _tier,
        _model_name,
    ):
        raise ValueError(
            "Invalid JSON when parsing model output"
        )

    with pytest.raises(
        ValueError,
        match="Invalid JSON",
    ):
        factory.run_with_fallback(
            primary_tier=ModelTier.BALANCED,
            fallback_tiers=[
                ModelTier.FAST,
            ],
            execute=fake_execute,
        )

    assert created_models == [
        "balanced-model",
    ]


def test_run_with_fallback_does_not_fallback_on_application_error(
    monkeypatch,
):
    factory = AgentsModelFactory()

    created_models = []

    monkeypatch.setattr(
        factory,
        "_create_model_from_name",
        lambda *, model_name: (
            created_models.append(
                model_name
            )
            or f"MODEL::{model_name}"
        ),
    )

    monkeypatch.setattr(
        factory.provider_config,
        "get_model",
        lambda tier: {
            ModelTier.FAST: "fast-model",
            ModelTier.BALANCED: "balanced-model",
        }[tier],
    )

    def fake_execute(
        _model,
        _tier,
        _model_name,
    ):
        raise ValueError(
            "SQL validation failed"
        )

    with pytest.raises(
        ValueError,
        match="SQL validation failed",
    ):
        factory.run_with_fallback(
            primary_tier=ModelTier.BALANCED,
            fallback_tiers=[
                ModelTier.FAST,
            ],
            execute=fake_execute,
        )

    assert created_models == [
        "balanced-model",
    ]


def test_provider_word_alone_does_not_trigger_fallback():
    error = ValueError(
        "provider field missing from analytical plan"
    )

    assert (
        LLMFallbackExecutor.is_provider_failure(
            error
        )
        is False
    )


def test_explicit_provider_failure_triggers_fallback():
    error = ProviderFailure(
        "temporary provider failure"
    )

    assert (
        LLMFallbackExecutor.is_provider_failure(
            error
        )
        is True
    )


def test_http_5xx_statuses_trigger_fallback():
    for status_code in (
        500,
        502,
        503,
        504,
    ):
        error = Exception(
            f"request failed with status {status_code}"
        )

        error.status_code = status_code

        assert (
            LLMFallbackExecutor.is_provider_failure(
                error
            )
            is True
        )


def test_http_429_triggers_fallback():
    error = Exception(
        "rate limited"
    )

    error.status_code = 429

    assert (
        LLMFallbackExecutor.is_provider_failure(
            error
        )
        is True
    )


def test_http_400_does_not_trigger_fallback():
    error = Exception(
        "bad request"
    )

    error.status_code = 400

    assert (
        LLMFallbackExecutor.is_provider_failure(
            error
        )
        is False
    )


def test_invalid_json_does_not_trigger_fallback():
    error = ValueError(
        "Invalid JSON when parsing model output"
    )

    assert (
        LLMFallbackExecutor.is_provider_failure(
            error
        )
        is False
    )


def test_structured_output_error_does_not_trigger_fallback():
    error = ValueError(
        "structured output validation failed"
    )

    assert (
        LLMFallbackExecutor.is_provider_failure(
            error
        )
        is False
    )


def test_pydantic_validation_error_does_not_trigger_fallback():
    error = ValueError(
        "Pydantic validation error"
    )

    assert (
        LLMFallbackExecutor.is_provider_failure(
            error
        )
        is False
    )


def test_maximum_fallback_attempts_are_bounded(
    monkeypatch,
):
    factory = AgentsModelFactory()

    monkeypatch.setattr(
        factory,
        "_create_model_from_name",
        lambda *, model_name: (
            f"MODEL::{model_name}"
        ),
    )

    monkeypatch.setattr(
        factory.provider_config,
        "get_model",
        lambda tier: {
            ModelTier.FAST: "fast-model",
            ModelTier.BALANCED: "balanced-model",
            ModelTier.REASONING: "reasoning-model",
        }[tier],
    )

    calls = []

    def fake_execute(
        _model,
        _tier,
        model_name,
    ):
        calls.append(
            model_name
        )

        raise FakeProviderError(
            "temporary upstream failure"
        )

    with pytest.raises(
        FallbackExhaustedError,
    ) as exc_info:
        factory.run_with_fallback(
            primary_tier=ModelTier.BALANCED,
            fallback_tiers=[
                ModelTier.FAST,
                ModelTier.REASONING,
            ],
            execute=fake_execute,
        )

    assert calls == [
        "balanced-model",
        "fast-model",
        "reasoning-model",
    ]

    assert len(
        exc_info.value.attempts
    ) == 3


def test_run_agent_with_fallback_uses_runner(
    monkeypatch,
):
    factory = AgentsModelFactory()

    monkeypatch.setattr(
        factory.provider_config,
        "get_model",
        lambda tier: {
            ModelTier.FAST: "fast-model",
            ModelTier.BALANCED: "balanced-model",
        }[tier],
    )

    monkeypatch.setattr(
        factory,
        "_create_model_from_name",
        lambda *, model_name: (
            f"MODEL::{model_name}"
        ),
    )

    runner_calls = []

    def fake_runner_run_sync(
        agent,
        input_text,
    ):
        runner_calls.append(
            (
                agent.name,
                agent.instructions,
                agent.model,
                input_text,
            )
        )

        return "runner-success"

    monkeypatch.setattr(
        "routing.agents_model.Runner.run_sync",
        fake_runner_run_sync,
    )

    result = factory.run_agent_with_fallback(
        primary_tier=ModelTier.BALANCED,
        fallback_tiers=[
            ModelTier.FAST,
        ],
        agent_name="Test Agent",
        instructions="Test instructions",
        input_text="Test request",
    )

    assert result.result == (
        "runner-success"
    )

    assert len(
        runner_calls
    ) == 1

    assert runner_calls[0][0] == (
        "Test Agent"
    )

    assert runner_calls[0][1] == (
        "Test instructions"
    )

    assert runner_calls[0][2] == (
        "MODEL::balanced-model"
    )

    assert runner_calls[0][3] == (
        "Test request"
    )

    assert len(
        result.attempts
    ) == 1

    assert (
        result.attempts[0].success
        is True
    )