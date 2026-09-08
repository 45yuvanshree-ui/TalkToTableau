from routing import LLMProviderConfig, ModelTier


def test_local_provider_uses_ollama_configuration(monkeypatch):
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
        "fast-model",
    )

    monkeypatch.setenv(
        "OLLAMA_BALANCED_MODEL",
        "balanced-model",
    )

    monkeypatch.setenv(
        "OLLAMA_REASONING_MODEL",
        "reasoning-model",
    )

    config = LLMProviderConfig()

    assert config.provider == "local"
    assert config.get_base_url() == (
        "http://localhost:11434/v1"
    )
    assert config.get_api_key() == "ollama"

    assert config.get_model(ModelTier.FAST) == "fast-model"
    assert config.get_model(ModelTier.BALANCED) == (
        "balanced-model"
    )
    assert config.get_model(ModelTier.REASONING) == (
        "reasoning-model"
    )


def test_openai_compatible_provider_uses_configuration(
    monkeypatch,
):
    monkeypatch.setenv(
        "LLM_PROVIDER",
        "openai_compatible",
    )

    monkeypatch.setenv(
        "OPENAI_BASE_URL",
        "https://api.example.com/v1",
    )

    monkeypatch.setenv(
        "OPENAI_API_KEY",
        "test-key",
    )

    monkeypatch.setenv(
        "OPENAI_FAST_MODEL",
        "fast-model",
    )

    monkeypatch.setenv(
        "OPENAI_BALANCED_MODEL",
        "balanced-model",
    )

    monkeypatch.setenv(
        "OPENAI_REASONING_MODEL",
        "reasoning-model",
    )

    config = LLMProviderConfig()

    assert config.provider == "openai_compatible"
    assert config.get_base_url() == (
        "https://api.example.com/v1"
    )
    assert config.get_api_key() == "test-key"

    assert config.get_model(ModelTier.FAST) == "fast-model"
    assert config.get_model(ModelTier.BALANCED) == (
        "balanced-model"
    )
    assert config.get_model(ModelTier.REASONING) == (
        "reasoning-model"
    )


def test_missing_openai_api_key_is_rejected(monkeypatch):
    monkeypatch.setenv(
        "LLM_PROVIDER",
        "openai_compatible",
    )

    monkeypatch.delenv(
        "OPENAI_API_KEY",
        raising=False,
    )

    config = LLMProviderConfig()

    try:
        config.get_api_key()
    except ValueError as error:
        assert "OPENAI_API_KEY" in str(error)
    else:
        raise AssertionError(
            "Expected missing API key to raise ValueError."
        )


def test_missing_local_model_is_rejected(monkeypatch):
    monkeypatch.setenv(
        "LLM_PROVIDER",
        "local",
    )

    monkeypatch.delenv(
        "OLLAMA_FAST_MODEL",
        raising=False,
    )

    config = LLMProviderConfig()

    try:
        config.get_model(ModelTier.FAST)
    except ValueError as error:
        assert "OLLAMA_FAST_MODEL" in str(error)
    else:
        raise AssertionError(
            "Expected missing local model to raise ValueError."
        )