from __future__ import annotations

from routing import (
    AgentsModelFactory,
    ModelTier,
)
from routing.fallback import (
    ProviderFailure,
)
from routing.model_pool import (
    ModelCandidate,
    ModelCapabilities,
    ModelPool,
)


def test_factory_builds_model_pool_from_provider_configuration(
    monkeypatch,
):
    monkeypatch.setenv(
        "LLM_PROVIDER",
        "openrouter",
    )

    monkeypatch.setenv(
        "OPENROUTER_FAST_MODELS",
        "fast-a:free,fast-b:free",
    )

    monkeypatch.setenv(
        "OPENROUTER_BALANCED_MODELS",
        "balanced-a:free,balanced-b:free",
    )

    monkeypatch.setenv(
        "OPENROUTER_REASONING_MODELS",
        "reasoning-a:free,reasoning-b:free",
    )

    monkeypatch.setenv(
        "OPENROUTER_API_KEY",
        "test-key",
    )

    factory = AgentsModelFactory()

    candidates = factory.model_pool.list_candidates()

    assert len(candidates) == 6

    names = [
        candidate.name
        for candidate in candidates
    ]

    assert names == [
        "fast-a:free",
        "fast-b:free",
        "balanced-a:free",
        "balanced-b:free",
        "reasoning-a:free",
        "reasoning-b:free",
    ]


def test_factory_preserves_single_model_configuration(
    monkeypatch,
):
    monkeypatch.setenv(
        "LLM_PROVIDER",
        "openrouter",
    )

    monkeypatch.delenv(
        "OPENROUTER_FAST_MODELS",
        raising=False,
    )

    monkeypatch.delenv(
        "OPENROUTER_BALANCED_MODELS",
        raising=False,
    )

    monkeypatch.delenv(
        "OPENROUTER_REASONING_MODELS",
        raising=False,
    )

    monkeypatch.setenv(
        "OPENROUTER_FAST_MODEL",
        "fast-model:free",
    )

    monkeypatch.setenv(
        "OPENROUTER_BALANCED_MODEL",
        "balanced-model:free",
    )

    monkeypatch.setenv(
        "OPENROUTER_REASONING_MODEL",
        "reasoning-model:free",
    )

    monkeypatch.setenv(
        "OPENROUTER_API_KEY",
        "test-key",
    )

    factory = AgentsModelFactory()

    assert (
        factory.provider_config.get_model(
            ModelTier.FAST
        )
        == "fast-model:free"
    )

    assert (
        factory.provider_config.get_model(
            ModelTier.BALANCED
        )
        == "balanced-model:free"
    )

    assert (
        factory.provider_config.get_model(
            ModelTier.REASONING
        )
        == "reasoning-model:free"
    )

    assert len(
        factory.model_pool.list_candidates()
    ) == 3


def test_factory_ranks_models_within_requested_tier(
    monkeypatch,
):
    monkeypatch.setenv(
        "LLM_PROVIDER",
        "openrouter",
    )

    monkeypatch.setenv(
        "OPENROUTER_FAST_MODELS",
        "fast-a:free,fast-b:free",
    )

    monkeypatch.setenv(
        "OPENROUTER_BALANCED_MODELS",
        "balanced-a:free,balanced-b:free",
    )

    monkeypatch.setenv(
        "OPENROUTER_REASONING_MODELS",
        "reasoning-a:free,reasoning-b:free",
    )

    monkeypatch.setenv(
        "OPENROUTER_API_KEY",
        "test-key",
    )

    factory = AgentsModelFactory()

    ranked = factory.rank_models(
        tier=ModelTier.BALANCED,
    )

    assert ranked

    balanced_models = [
        item
        for item in ranked
        if item.candidate.tier
        == ModelTier.BALANCED
    ]

    assert len(balanced_models) == 2

    assert balanced_models[0].candidate.name == (
        "balanced-a:free"
    )

    assert balanced_models[1].candidate.name == (
        "balanced-b:free"
    )


def test_factory_selects_model_from_requested_tier(
    monkeypatch,
):
    monkeypatch.setenv(
        "LLM_PROVIDER",
        "openrouter",
    )

    monkeypatch.setenv(
        "OPENROUTER_FAST_MODELS",
        "fast-a:free,fast-b:free",
    )

    monkeypatch.setenv(
        "OPENROUTER_BALANCED_MODELS",
        "balanced-a:free,balanced-b:free",
    )

    monkeypatch.setenv(
        "OPENROUTER_REASONING_MODELS",
        "reasoning-a:free,reasoning-b:free",
    )

    monkeypatch.setenv(
        "OPENROUTER_API_KEY",
        "test-key",
    )

    factory = AgentsModelFactory()

    selected = factory.select_model_name(
        tier=ModelTier.BALANCED,
    )

    assert selected == "balanced-a:free"


def test_factory_select_model_for_tier_uses_selected_candidate(
    monkeypatch,
):
    monkeypatch.setenv(
        "LLM_PROVIDER",
        "openrouter",
    )

    monkeypatch.setenv(
        "OPENROUTER_FAST_MODELS",
        "fast-a:free",
    )

    monkeypatch.setenv(
        "OPENROUTER_BALANCED_MODELS",
        "balanced-a:free,balanced-b:free",
    )

    monkeypatch.setenv(
        "OPENROUTER_REASONING_MODELS",
        "reasoning-a:free",
    )

    monkeypatch.setenv(
        "OPENROUTER_API_KEY",
        "test-key",
    )

    factory = AgentsModelFactory()

    monkeypatch.setattr(
        factory,
        "_create_model_from_name",
        lambda *, model_name: (
            f"MODEL::{model_name}"
        ),
    )

    model, model_name = (
        factory.select_model_for_tier(
            ModelTier.BALANCED
        )
    )

    assert model == (
        "MODEL::balanced-a:free"
    )

    assert model_name == (
        "balanced-a:free"
    )


def test_factory_records_model_health(
    monkeypatch,
):
    monkeypatch.setenv(
        "LLM_PROVIDER",
        "openrouter",
    )

    monkeypatch.setenv(
        "OPENROUTER_FAST_MODELS",
        "fast-a:free",
    )

    monkeypatch.setenv(
        "OPENROUTER_BALANCED_MODELS",
        "balanced-a:free",
    )

    monkeypatch.setenv(
        "OPENROUTER_REASONING_MODELS",
        "reasoning-a:free",
    )

    monkeypatch.setenv(
        "OPENROUTER_API_KEY",
        "test-key",
    )

    factory = AgentsModelFactory()

    factory.record_model_provider_failure(
        tier=ModelTier.FAST,
        model_name="fast-a:free",
    )

    candidate = factory.model_pool.get_candidate(
        provider="openrouter",
        model_name="fast-a:free",
    )

    assert candidate is not None

    assert (
        candidate.health.provider_failure_count
        == 1
    )

    factory.record_model_success(
        tier=ModelTier.FAST,
        model_name="fast-a:free",
        latency_ms=500,
    )

    assert (
        candidate.health.success_count
        == 1
    )

    assert (
        candidate.health.average_latency_ms
        == 500
    )


def test_factory_records_semantic_failure(
    monkeypatch,
):
    monkeypatch.setenv(
        "LLM_PROVIDER",
        "openrouter",
    )

    monkeypatch.setenv(
        "OPENROUTER_FAST_MODELS",
        "fast-a:free",
    )

    monkeypatch.setenv(
        "OPENROUTER_BALANCED_MODELS",
        "balanced-a:free",
    )

    monkeypatch.setenv(
        "OPENROUTER_REASONING_MODELS",
        "reasoning-a:free",
    )

    monkeypatch.setenv(
        "OPENROUTER_API_KEY",
        "test-key",
    )

    factory = AgentsModelFactory()

    factory.record_model_success(
        tier=ModelTier.FAST,
        model_name="fast-a:free",
    )

    factory.record_model_semantic_failure(
        tier=ModelTier.FAST,
        model_name="fast-a:free",
    )

    candidate = factory.model_pool.get_candidate(
        provider="openrouter",
        model_name="fast-a:free",
    )

    assert candidate is not None

    assert (
        candidate.health.success_count
        == 1
    )

    assert (
        candidate.health.semantic_failure_count
        == 1
    )

    assert (
        candidate.health.success_rate
        == 0.5
    )


def test_factory_accepts_injected_model_pool(
    monkeypatch,
):
    monkeypatch.setenv(
        "LLM_PROVIDER",
        "openrouter",
    )

    monkeypatch.setenv(
        "OPENROUTER_FAST_MODEL",
        "configured-fast:free",
    )

    monkeypatch.setenv(
        "OPENROUTER_BALANCED_MODEL",
        "configured-balanced:free",
    )

    monkeypatch.setenv(
        "OPENROUTER_REASONING_MODEL",
        "configured-reasoning:free",
    )

    monkeypatch.setenv(
        "OPENROUTER_API_KEY",
        "test-key",
    )

    injected_pool = ModelPool(
        candidates=[
            ModelCandidate(
                name="injected-fast:free",
                provider="openrouter",
                tier=ModelTier.FAST,
                is_free=True,
                capabilities=ModelCapabilities(
                    sql_generation=True,
                ),
            ),
        ]
    )

    factory = AgentsModelFactory(
        model_pool=injected_pool,
    )

    assert (
        factory.model_pool
        is injected_pool
    )

    assert (
        factory.select_model_name(
            tier=ModelTier.FAST,
        )
        == "injected-fast:free"
    )


def test_factory_refreshes_model_pool(
    monkeypatch,
):
    monkeypatch.setenv(
        "LLM_PROVIDER",
        "openrouter",
    )

    monkeypatch.setenv(
        "OPENROUTER_FAST_MODELS",
        "fast-a:free",
    )

    monkeypatch.setenv(
        "OPENROUTER_BALANCED_MODELS",
        "balanced-a:free",
    )

    monkeypatch.setenv(
        "OPENROUTER_REASONING_MODELS",
        "reasoning-a:free",
    )

    monkeypatch.setenv(
        "OPENROUTER_API_KEY",
        "test-key",
    )

    factory = AgentsModelFactory()

    assert (
        factory.select_model_name(
            tier=ModelTier.FAST,
        )
        == "fast-a:free"
    )

    monkeypatch.setenv(
        "OPENROUTER_FAST_MODELS",
        "new-fast-a:free,new-fast-b:free",
    )

    factory.refresh_model_pool()

    assert (
        factory.select_model_name(
            tier=ModelTier.FAST,
        )
        == "new-fast-a:free"
    )

    names = [
        candidate.name
        for candidate
        in factory.model_pool.list_candidates()
    ]

    assert names == [
        "new-fast-a:free",
        "new-fast-b:free",
        "balanced-a:free",
        "reasoning-a:free",
    ]


def test_unknown_exception_does_not_mark_model_as_failed(
    monkeypatch,
):
    monkeypatch.setenv(
        "LLM_PROVIDER",
        "openrouter",
    )

    monkeypatch.setenv(
        "OPENROUTER_FAST_MODEL",
        "fast-model:free",
    )

    monkeypatch.setenv(
        "OPENROUTER_BALANCED_MODEL",
        "balanced-model:free",
    )

    monkeypatch.setenv(
        "OPENROUTER_REASONING_MODEL",
        "reasoning-model:free",
    )

    monkeypatch.setenv(
        "OPENROUTER_API_KEY",
        "test-key",
    )

    factory = AgentsModelFactory()

    model_name = factory.provider_config.get_models(
        ModelTier.BALANCED
    )[0]

    candidate = factory.model_pool.get_candidate(
        provider=factory.provider_config.provider,
        model_name=model_name,
    )

    assert candidate is not None

    initial_failure_count = (
        candidate.health.failure_count
    )

    initial_semantic_failure_count = (
        candidate.health.semantic_failure_count
    )

    initial_provider_failure_count = (
        candidate.health.provider_failure_count
    )

    class InternalApplicationError(Exception):
        pass

    factory._record_failure_from_exception(
        tier=ModelTier.BALANCED,
        model_name=model_name,
        latency_ms=100.0,
        error=InternalApplicationError(
            "unexpected internal failure"
        ),
    )

    assert (
        candidate.health.failure_count
        == initial_failure_count
    )

    assert (
        candidate.health.semantic_failure_count
        == initial_semantic_failure_count
    )

    assert (
        candidate.health.provider_failure_count
        == initial_provider_failure_count
    )

    assert candidate.is_available() is True


def test_known_semantic_exception_updates_model_health(
    monkeypatch,
):
    monkeypatch.setenv(
        "LLM_PROVIDER",
        "openrouter",
    )

    monkeypatch.setenv(
        "OPENROUTER_FAST_MODEL",
        "fast-model:free",
    )

    monkeypatch.setenv(
        "OPENROUTER_BALANCED_MODEL",
        "balanced-model:free",
    )

    monkeypatch.setenv(
        "OPENROUTER_REASONING_MODEL",
        "reasoning-model:free",
    )

    monkeypatch.setenv(
        "OPENROUTER_API_KEY",
        "test-key",
    )

    factory = AgentsModelFactory()

    model_name = factory.provider_config.get_models(
        ModelTier.BALANCED
    )[0]

    candidate = factory.model_pool.get_candidate(
        provider=factory.provider_config.provider,
        model_name=model_name,
    )

    assert candidate is not None

    initial_semantic_failure_count = (
        candidate.health.semantic_failure_count
    )

    class SQLValidationError(Exception):
        pass

    factory._record_failure_from_exception(
        tier=ModelTier.BALANCED,
        model_name=model_name,
        latency_ms=100.0,
        error=SQLValidationError(
            "generated SQL failed validation"
        ),
    )

    assert (
        candidate.health.semantic_failure_count
        == initial_semantic_failure_count + 1
    )

    assert candidate.is_available() is True


def test_provider_failure_still_updates_provider_health(
    monkeypatch,
):
    monkeypatch.setenv(
        "LLM_PROVIDER",
        "openrouter",
    )

    monkeypatch.setenv(
        "OPENROUTER_FAST_MODEL",
        "fast-model:free",
    )

    monkeypatch.setenv(
        "OPENROUTER_BALANCED_MODEL",
        "balanced-model:free",
    )

    monkeypatch.setenv(
        "OPENROUTER_REASONING_MODEL",
        "reasoning-model:free",
    )

    monkeypatch.setenv(
        "OPENROUTER_API_KEY",
        "test-key",
    )

    factory = AgentsModelFactory()

    model_name = factory.provider_config.get_models(
        ModelTier.BALANCED
    )[0]

    candidate = factory.model_pool.get_candidate(
        provider=factory.provider_config.provider,
        model_name=model_name,
    )

    assert candidate is not None

    initial_provider_failure_count = (
        candidate.health.provider_failure_count
    )

    provider_error = ProviderFailure(
        "temporary provider failure"
    )

    factory._record_failure_from_exception(
        tier=ModelTier.BALANCED,
        model_name=model_name,
        latency_ms=100.0,
        error=provider_error,
    )

    assert (
        candidate.health.provider_failure_count
        == initial_provider_failure_count + 1
    )

    assert candidate.is_available() is False