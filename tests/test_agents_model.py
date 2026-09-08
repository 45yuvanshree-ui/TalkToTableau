from agents import Model

from routing import (
    AgentsModelFactory,
    LLMProviderConfig,
    ModelTier,
)


def test_agents_model_factory_creates_fast_model(
    monkeypatch,
):
    monkeypatch.setenv(
        "LLM_PROVIDER",
        "local",
    )

    monkeypatch.setenv(
        "OLLAMA_BASE_URL",
        "http://localhost:11434/v1",
    )

    monkeypatch.setenv(
        "OLLAMA_API_KEY",
        "ollama",
    )

    monkeypatch.setenv(
        "OLLAMA_FAST_MODEL",
        "test-fast-model",
    )

    config = LLMProviderConfig()

    factory = AgentsModelFactory(config)

    model = factory.create_model(ModelTier.FAST)

    assert isinstance(model, Model)


def test_agents_model_factory_creates_reasoning_model(
    monkeypatch,
):
    monkeypatch.setenv(
        "LLM_PROVIDER",
        "local",
    )

    monkeypatch.setenv(
        "OLLAMA_BASE_URL",
        "http://localhost:11434/v1",
    )

    monkeypatch.setenv(
        "OLLAMA_API_KEY",
        "ollama",
    )

    monkeypatch.setenv(
        "OLLAMA_REASONING_MODEL",
        "test-reasoning-model",
    )

    config = LLMProviderConfig()

    factory = AgentsModelFactory(config)

    model = factory.create_model(ModelTier.REASONING)

    assert isinstance(model, Model)