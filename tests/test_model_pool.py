from routing.model_pool import (
    ModelCapabilities,
    ModelCandidate,
    ModelPool,
    ModelSelectionRequest,
)
from routing.models import ModelTier


def create_test_pool() -> ModelPool:
    return ModelPool(
        candidates=[
            ModelCandidate(
                name="fast-free",
                provider="openrouter",
                tier=ModelTier.FAST,
                is_free=True,
                priority=5,
                capabilities=ModelCapabilities(
                    structured_output=True,
                    tool_calling=True,
                    reasoning=False,
                    sql_generation=True,
                ),
            ),
            ModelCandidate(
                name="balanced-free",
                provider="openrouter",
                tier=ModelTier.BALANCED,
                is_free=True,
                priority=3,
                capabilities=ModelCapabilities(
                    structured_output=True,
                    tool_calling=True,
                    reasoning=True,
                    sql_generation=True,
                ),
            ),
            ModelCandidate(
                name="reasoning-paid",
                provider="openrouter",
                tier=ModelTier.REASONING,
                is_free=False,
                priority=1,
                capabilities=ModelCapabilities(
                    structured_output=True,
                    tool_calling=True,
                    reasoning=True,
                    sql_generation=True,
                ),
            ),
        ]
    )


def test_register_and_list_candidates():
    pool = create_test_pool()

    candidates = pool.list_candidates()

    assert len(candidates) == 3


def test_selects_matching_free_fast_model():
    pool = create_test_pool()

    request = ModelSelectionRequest(
        tier=ModelTier.FAST,
        prefer_free=True,
    )

    selected = pool.select(request)

    assert selected.candidate.name == "fast-free"
    assert selected.candidate.is_free is True


def test_reasoning_requirement_filters_non_reasoning_models():
    pool = create_test_pool()

    request = ModelSelectionRequest(
        tier=ModelTier.REASONING,
        reasoning_required=True,
        prefer_free=True,
    )

    selected = pool.select(request)

    assert selected.candidate.name == "balanced-free"
    assert selected.candidate.capabilities.reasoning is True


def test_provider_failure_temporarily_removes_model():
    pool = create_test_pool()

    pool.record_provider_failure(
        provider="openrouter",
        model_name="fast-free",
        cooldown_seconds=60,
    )

    request = ModelSelectionRequest(
        tier=ModelTier.FAST,
        prefer_free=True,
    )

    selected = pool.select(request)

    assert selected.candidate.name != "fast-free"


def test_success_clears_temporary_unavailability():
    pool = create_test_pool()

    pool.record_provider_failure(
        provider="openrouter",
        model_name="fast-free",
        cooldown_seconds=60,
    )

    candidate = pool.get_candidate(
        provider="openrouter",
        model_name="fast-free",
    )

    assert candidate is not None
    assert candidate.is_available() is False

    pool.record_success(
        provider="openrouter",
        model_name="fast-free",
        latency_ms=100,
    )

    assert candidate.is_available() is True
    assert candidate.health.success_count == 1


def test_semantic_failure_reduces_success_rate():
    pool = create_test_pool()

    pool.record_success(
        provider="openrouter",
        model_name="fast-free",
        latency_ms=100,
    )

    pool.record_semantic_failure(
        provider="openrouter",
        model_name="fast-free",
    )

    candidate = pool.get_candidate(
        provider="openrouter",
        model_name="fast-free",
    )

    assert candidate is not None
    assert candidate.health.success_count == 1
    assert candidate.health.semantic_failure_count == 1
    assert candidate.health.success_rate == 0.5


def test_unavailable_model_is_not_returned():
    pool = create_test_pool()

    pool.record_provider_failure(
        provider="openrouter",
        model_name="fast-free",
        cooldown_seconds=60,
    )

    available = pool.available_candidates(
        ModelSelectionRequest(
            tier=ModelTier.FAST,
            prefer_free=True,
        )
    )

    names = {
        candidate.name
        for candidate in available
    }

    assert "fast-free" not in names


def test_unregister_model():
    pool = create_test_pool()

    removed = pool.unregister(
        provider="openrouter",
        model_name="fast-free",
    )

    assert removed is True

    assert pool.get_candidate(
        provider="openrouter",
        model_name="fast-free",
    ) is None


def test_duplicate_registration_replaces_existing_model():
    pool = ModelPool()

    pool.register(
        ModelCandidate(
            name="example-model",
            provider="openrouter",
            tier=ModelTier.FAST,
        )
    )

    pool.register(
        ModelCandidate(
            name="example-model",
            provider="openrouter",
            tier=ModelTier.BALANCED,
        )
    )

    candidates = pool.list_candidates()

    assert len(candidates) == 1
    assert candidates[0].tier == ModelTier.BALANCED


def test_no_matching_capability_raises_error():
    pool = ModelPool(
        candidates=[
            ModelCandidate(
                name="basic-model",
                provider="openrouter",
                tier=ModelTier.FAST,
                capabilities=ModelCapabilities(
                    reasoning=False,
                    structured_output=False,
                    tool_calling=False,
                    sql_generation=True,
                ),
            )
        ]
    )

    request = ModelSelectionRequest(
        tier=ModelTier.REASONING,
        reasoning_required=True,
    )

    try:
        pool.select(request)
    except RuntimeError as error:
        assert (
            "No available LLM model"
            in str(error)
        )
    else:
        raise AssertionError(
            "Expected RuntimeError when no compatible model exists."
        )