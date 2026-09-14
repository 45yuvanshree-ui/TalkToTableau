from __future__ import annotations

from routing.agents_model import AgentsModelFactory
from routing.model_catalog import ModelCatalogUnavailableError
from routing.model_catalog import OpenRouterModelCatalog
from routing.model_pool import ModelSelectionRequest
from routing.models import ModelTier


class FakeOpenRouterConfig:
    provider = "openrouter"

    def get_api_key(self) -> str:
        return "test-key"

    def get_base_url(self) -> str:
        return "https://openrouter.ai/api/v1"

    def get_models(self, tier: ModelTier) -> list[str]:
        configured = {
            ModelTier.FAST: [
                "configured-fast:free",
            ],
            ModelTier.BALANCED: [
                "configured-balanced:free",
            ],
            ModelTier.REASONING: [
                "configured-reasoning:free",
            ],
        }
        return configured[tier]


def make_metadata(
    model_id: str,
    *,
    is_free: bool = True,
    supported_parameters: tuple[str, ...] = (),
    context_length: int = 16_000,
):
    from routing.model_catalog import OpenRouterModelMetadata

    return OpenRouterModelMetadata(
        model_id=model_id,
        name=model_id,
        canonical_slug=model_id,
        context_length=context_length,
        input_modalities=("text",),
        output_modalities=("text",),
        supported_parameters=supported_parameters,
        prompt_price=0.0 if is_free else 1.0,
        completion_price=0.0 if is_free else 1.0,
        is_free=is_free,
        raw={},
    )


def get_pool_model_names(factory: AgentsModelFactory) -> set[str]:
    names: set[str] = set()

    for tier in ModelTier:
        request = ModelSelectionRequest(
            tier=tier,
        )

        names.update(
            candidate.name
            for candidate in factory.model_pool.available_candidates(
                request
            )
        )

    return names


def test_dynamic_models_are_registered(monkeypatch) -> None:
    discovered = [
        make_metadata("dynamic-balanced-1:free"),
        make_metadata("dynamic-balanced-2:free"),
    ]

    monkeypatch.setattr(
        OpenRouterModelCatalog,
        "list_models",
        lambda self, output_modalities="text": discovered,
    )

    factory = AgentsModelFactory(
        provider_config=FakeOpenRouterConfig(),
    )

    registered = factory.discover_and_register_models(
        tier=ModelTier.BALANCED,
    )

    assert [candidate.name for candidate in registered] == [
        "dynamic-balanced-1:free",
        "dynamic-balanced-2:free",
    ]

    pool_names = get_pool_model_names(factory)

    assert "dynamic-balanced-1:free" in pool_names
    assert "dynamic-balanced-2:free" in pool_names


def test_configured_models_are_not_replaced(monkeypatch) -> None:
    discovered = [
        make_metadata("configured-balanced:free"),
        make_metadata("new-balanced:free"),
    ]

    monkeypatch.setattr(
        OpenRouterModelCatalog,
        "list_models",
        lambda self, output_modalities="text": discovered,
    )

    factory = AgentsModelFactory(
        provider_config=FakeOpenRouterConfig(),
    )

    registered = factory.discover_and_register_models(
        tier=ModelTier.BALANCED,
    )

    registered_names = [candidate.name for candidate in registered]

    assert "configured-balanced:free" not in registered_names
    assert "new-balanced:free" in registered_names


def test_duplicate_discovered_models_are_skipped(monkeypatch) -> None:
    discovered = [
        make_metadata("duplicate-model:free"),
        make_metadata("duplicate-model:free"),
    ]

    monkeypatch.setattr(
        OpenRouterModelCatalog,
        "list_models",
        lambda self, output_modalities="text": discovered,
    )

    factory = AgentsModelFactory(
        provider_config=FakeOpenRouterConfig(),
    )

    registered = factory.discover_and_register_models(
        tier=ModelTier.BALANCED,
    )

    assert [candidate.name for candidate in registered] == [
        "duplicate-model:free",
    ]


def test_paid_models_are_rejected_by_default(monkeypatch) -> None:
    discovered = [
        make_metadata(
            "paid-model",
            is_free=False,
        ),
    ]

    monkeypatch.setattr(
        OpenRouterModelCatalog,
        "list_models",
        lambda self, output_modalities="text": discovered,
    )

    factory = AgentsModelFactory(
        provider_config=FakeOpenRouterConfig(),
    )

    registered = factory.discover_and_register_models(
        tier=ModelTier.BALANCED,
    )

    assert registered == []


def test_reasoning_requirement_filters_models(monkeypatch) -> None:
    discovered = [
        make_metadata(
            "reasoning-model:free",
            supported_parameters=("reasoning",),
        ),
        make_metadata(
            "normal-model:free",
        ),
    ]

    monkeypatch.setattr(
        OpenRouterModelCatalog,
        "list_models",
        lambda self, output_modalities="text": discovered,
    )

    factory = AgentsModelFactory(
        provider_config=FakeOpenRouterConfig(),
    )

    registered = factory.discover_and_register_models(
        tier=ModelTier.REASONING,
        require_reasoning=True,
    )

    assert [candidate.name for candidate in registered] == [
        "reasoning-model:free",
    ]

    assert registered[0].capabilities.reasoning is True


def test_structured_output_requirement_filters_models(monkeypatch) -> None:
    discovered = [
        make_metadata(
            "structured-model:free",
            supported_parameters=("structured_outputs",),
        ),
        make_metadata(
            "normal-model:free",
        ),
    ]

    monkeypatch.setattr(
        OpenRouterModelCatalog,
        "list_models",
        lambda self, output_modalities="text": discovered,
    )

    factory = AgentsModelFactory(
        provider_config=FakeOpenRouterConfig(),
    )

    registered = factory.discover_and_register_models(
        tier=ModelTier.BALANCED,
        require_structured_output=True,
    )

    assert [candidate.name for candidate in registered] == [
        "structured-model:free",
    ]

    assert registered[0].capabilities.structured_output is True


def test_tool_calling_requirement_filters_models(monkeypatch) -> None:
    discovered = [
        make_metadata(
            "tool-model:free",
            supported_parameters=("tools",),
        ),
        make_metadata(
            "normal-model:free",
        ),
    ]

    monkeypatch.setattr(
        OpenRouterModelCatalog,
        "list_models",
        lambda self, output_modalities="text": discovered,
    )

    factory = AgentsModelFactory(
        provider_config=FakeOpenRouterConfig(),
    )

    registered = factory.discover_and_register_models(
        tier=ModelTier.BALANCED,
        require_tool_calling=True,
    )

    assert [candidate.name for candidate in registered] == [
        "tool-model:free",
    ]

    assert registered[0].capabilities.tool_calling is True


def test_context_length_requirement_filters_models(monkeypatch) -> None:
    discovered = [
        make_metadata(
            "short-model:free",
            context_length=8_000,
        ),
        make_metadata(
            "long-model:free",
            context_length=64_000,
        ),
    ]

    monkeypatch.setattr(
        OpenRouterModelCatalog,
        "list_models",
        lambda self, output_modalities="text": discovered,
    )

    factory = AgentsModelFactory(
        provider_config=FakeOpenRouterConfig(),
    )

    registered = factory.discover_and_register_models(
        tier=ModelTier.BALANCED,
        minimum_context_length=32_000,
    )

    assert [candidate.name for candidate in registered] == [
        "long-model:free",
    ]

    assert registered[0].metadata["context_length"] == 64_000


def test_discovery_failure_keeps_configured_pool(
    monkeypatch,
) -> None:
    def raise_catalog_failure(
        self,
        output_modalities="text",
    ):
        raise ModelCatalogUnavailableError(
            "catalog unavailable"
        )

    monkeypatch.setattr(
        OpenRouterModelCatalog,
        "list_models",
        raise_catalog_failure,
    )

    factory = AgentsModelFactory(
        provider_config=FakeOpenRouterConfig(),
    )

    registered = factory.discover_and_register_models(
        tier=ModelTier.BALANCED,
    )

    assert registered == []

    pool_names = get_pool_model_names(factory)

    assert "configured-fast:free" in pool_names
    assert "configured-balanced:free" in pool_names
    assert "configured-reasoning:free" in pool_names


def test_non_openrouter_provider_skips_dynamic_discovery() -> None:
    class FakeLocalConfig:
        provider = "local"

        def get_models(self, tier: ModelTier) -> list[str]:
            return ["local-model"]

    factory = AgentsModelFactory(
        provider_config=FakeLocalConfig(),
    )

    registered = factory.discover_and_register_models(
        tier=ModelTier.BALANCED,
    )

    assert registered == []


def test_invalid_context_length_is_rejected() -> None:
    factory = AgentsModelFactory(
        provider_config=FakeOpenRouterConfig(),
    )

    try:
        factory.discover_and_register_models(
            tier=ModelTier.BALANCED,
            minimum_context_length=0,
        )
    except ValueError as exc:
        assert "minimum_context_length" in str(exc)
    else:
        raise AssertionError(
            "Expected ValueError for invalid minimum_context_length"
        )