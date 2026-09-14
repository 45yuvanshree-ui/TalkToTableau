from __future__ import annotations

import pytest

from agents import Agent

from routing import (
    AgentsModelFactory,
    AgentsSDKExecutor,
    ModelTier,
)
from routing.fallback import (
    FallbackExhaustedError,
)


class FakeProviderError(Exception):
    """Synthetic transient provider failure."""

    status_code = 502


def test_executor_runs_existing_agent_factory(
    monkeypatch,
):
    model_factory = AgentsModelFactory()

    monkeypatch.setattr(
        model_factory.provider_config,
        "get_model",
        lambda tier: {
            ModelTier.FAST: "fast-model",
            ModelTier.BALANCED: "balanced-model",
        }[tier],
    )

    monkeypatch.setattr(
        model_factory,
        "_create_model_from_name",
        lambda *, model_name: (
            f"MODEL::{model_name}"
        ),
    )

    runner_calls = []

    def fake_runner_run_sync(
        agent,
        prompt,
    ):
        runner_calls.append(
            (
                agent.name,
                agent.model,
                prompt,
            )
        )

        return "runner-success"

    monkeypatch.setattr(
        "routing.agent_executor.Runner.run_sync",
        fake_runner_run_sync,
    )

    def create_test_agent(
        model,
    ):
        return Agent(
            name="Test Agent",
            instructions="Test instructions",
            model=model,
        )

    executor = AgentsSDKExecutor(
        model_factory=model_factory
    )

    result = executor.run(
        primary_tier=ModelTier.BALANCED,
        fallback_tiers=[
            ModelTier.FAST,
        ],
        agent_factory=create_test_agent,
        prompt="Test prompt",
    )

    assert result.result == (
        "runner-success"
    )

    assert runner_calls == [
        (
            "Test Agent",
            "MODEL::balanced-model",
            "Test prompt",
        )
    ]

    assert len(
        result.attempts
    ) == 1

    assert (
        result.attempts[0].model_name
        == "balanced-model"
    )

    assert (
        result.attempts[0].success
        is True
    )


def test_executor_rebuilds_agent_with_fallback_model(
    monkeypatch,
):
    model_factory = AgentsModelFactory()

    monkeypatch.setattr(
        model_factory.provider_config,
        "get_model",
        lambda tier: {
            ModelTier.FAST: "fast-model",
            ModelTier.BALANCED: "balanced-model",
        }[tier],
    )

    monkeypatch.setattr(
        model_factory,
        "_create_model_from_name",
        lambda *, model_name: (
            f"MODEL::{model_name}"
        ),
    )

    runner_calls = []

    def fake_runner_run_sync(
        agent,
        prompt,
    ):
        runner_calls.append(
            (
                agent.model,
                prompt,
            )
        )

        if agent.model == (
            "MODEL::balanced-model"
        ):
            raise FakeProviderError(
                "502 upstream failure"
            )

        return "fallback-success"

    monkeypatch.setattr(
        "routing.agent_executor.Runner.run_sync",
        fake_runner_run_sync,
    )

    def create_test_agent(
        model,
    ):
        return Agent(
            name="Fallback Test Agent",
            instructions="Fallback instructions",
            model=model,
        )

    executor = AgentsSDKExecutor(
        model_factory=model_factory
    )

    result = executor.run(
        primary_tier=ModelTier.BALANCED,
        fallback_tiers=[
            ModelTier.FAST,
        ],
        agent_factory=create_test_agent,
        prompt="Fallback prompt",
    )

    assert result.result == (
        "fallback-success"
    )

    assert runner_calls == [
        (
            "MODEL::balanced-model",
            "Fallback prompt",
        ),
        (
            "MODEL::fast-model",
            "Fallback prompt",
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


def test_executor_does_not_fallback_on_semantic_error(
    monkeypatch,
):
    model_factory = AgentsModelFactory()

    created_models = []

    monkeypatch.setattr(
        model_factory.provider_config,
        "get_model",
        lambda tier: {
            ModelTier.FAST: "fast-model",
            ModelTier.BALANCED: "balanced-model",
        }[tier],
    )

    monkeypatch.setattr(
        model_factory,
        "_create_model_from_name",
        lambda *, model_name: (
            created_models.append(
                model_name
            )
            or f"MODEL::{model_name}"
        ),
    )

    def fake_runner_run_sync(
        _agent,
        _prompt,
    ):
        raise ValueError(
            "invalid structured output"
        )

    monkeypatch.setattr(
        "routing.agent_executor.Runner.run_sync",
        fake_runner_run_sync,
    )

    def create_test_agent(
        model,
    ):
        return Agent(
            name="Semantic Error Agent",
            instructions="Test",
            model=model,
        )

    executor = AgentsSDKExecutor(
        model_factory=model_factory
    )

    with pytest.raises(
        ValueError,
        match="invalid structured output",
    ):
        executor.run(
            primary_tier=ModelTier.BALANCED,
            fallback_tiers=[
                ModelTier.FAST,
            ],
            agent_factory=create_test_agent,
            prompt="Semantic failure",
        )

    assert created_models == [
        "balanced-model",
    ]


def test_executor_exhausts_bounded_fallback(
    monkeypatch,
):
    model_factory = AgentsModelFactory()

    monkeypatch.setattr(
        model_factory.provider_config,
        "get_model",
        lambda tier: {
            ModelTier.FAST: "fast-model",
            ModelTier.BALANCED: "balanced-model",
            ModelTier.REASONING: "reasoning-model",
        }[tier],
    )

    monkeypatch.setattr(
        model_factory,
        "_create_model_from_name",
        lambda *, model_name: (
            f"MODEL::{model_name}"
        ),
    )

    runner_calls = []

    def fake_runner_run_sync(
        _agent,
        _prompt,
    ):
        runner_calls.append(True)

        raise FakeProviderError(
            "502 upstream provider failure"
        )

    monkeypatch.setattr(
        "routing.agent_executor.Runner.run_sync",
        fake_runner_run_sync,
    )

    def create_test_agent(
        model,
    ):
        return Agent(
            name="Exhaustion Agent",
            instructions="Test",
            model=model,
        )

    executor = AgentsSDKExecutor(
        model_factory=model_factory
    )

    with pytest.raises(
        FallbackExhaustedError,
    ) as exc_info:
        executor.run(
            primary_tier=ModelTier.BALANCED,
            fallback_tiers=[
                ModelTier.FAST,
                ModelTier.REASONING,
            ],
            agent_factory=create_test_agent,
            prompt="Exhaustion test",
        )

    assert len(
        runner_calls
    ) == 3

    assert len(
        exc_info.value.attempts
    ) == 3