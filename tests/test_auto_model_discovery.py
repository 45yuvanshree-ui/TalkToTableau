from __future__ import annotations

from routing.agents_model import AgentsModelFactory
from routing.model_catalog import (
    ModelCatalogUnavailableError,
    OpenRouterModelCatalog,
    OpenRouterModelMetadata,
)
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
    context_length: int | None = 16_000,
    supported_parameters: tuple[str, ...] = (),
    is_free: bool = True,
) -> OpenRouterModelMetadata:
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


def get_pool_model_names(
    factory: AgentsModelFactory,
) -> set[str]:
    names: set[str] = set()

    for tier in ModelTier:
        request = ModelSelectionRequest(
            tier=tier,
        )

        names.update(
            candidate.name
            for candidate in (
                factory.model_pool.available_candidates(
                    request
                )
            )
        )

    return names


def get_pool_candidate(
    factory: AgentsModelFactory,
    model_name: str,
):
    for tier in ModelTier:
        request = ModelSelectionRequest(
            tier=tier,
        )

        for candidate in (
            factory.model_pool.available_candidates(
                request
            )
        ):
            if candidate.name == model_name:
                return candidate

    return None


def test_auto_discovery_assigns_reasoning_models() -> None:
    discovered = [
        make_metadata(
            "dynamic-reasoning:free",
            supported_parameters=("reasoning",),
        ),
    ]

    original = OpenRouterModelCatalog.list_models

    try:
        OpenRouterModelCatalog.list_models = (
            lambda self, output_modalities="text": discovered
        )

        factory = AgentsModelFactory(
            provider_config=FakeOpenRouterConfig(),
        )

        registered = (
            factory.discover_and_register_models_auto()
        )

        assert len(registered) == 1
        assert registered[0].name == (
            "dynamic-reasoning:free"
        )
        assert registered[0].tier == ModelTier.REASONING

    finally:
        OpenRouterModelCatalog.list_models = original


def test_auto_discovery_assigns_balanced_for_structured_output() -> None:
    discovered = [
        make_metadata(
            "dynamic-structured:free",
            supported_parameters=(
                "structured_outputs",
            ),
        ),
    ]

    original = OpenRouterModelCatalog.list_models

    try:
        OpenRouterModelCatalog.list_models = (
            lambda self, output_modalities="text": discovered
        )

        factory = AgentsModelFactory(
            provider_config=FakeOpenRouterConfig(),
        )

        registered = (
            factory.discover_and_register_models_auto()
        )

        assert len(registered) == 1
        assert registered[0].tier == ModelTier.BALANCED

    finally:
        OpenRouterModelCatalog.list_models = original


def test_auto_discovery_assigns_balanced_for_tools() -> None:
    discovered = [
        make_metadata(
            "dynamic-tools:free",
            supported_parameters=("tools",),
        ),
    ]

    original = OpenRouterModelCatalog.list_models

    try:
        OpenRouterModelCatalog.list_models = (
            lambda self, output_modalities="text": discovered
        )

        factory = AgentsModelFactory(
            provider_config=FakeOpenRouterConfig(),
        )

        registered = (
            factory.discover_and_register_models_auto()
        )

        assert len(registered) == 1
        assert registered[0].tier == ModelTier.BALANCED

    finally:
        OpenRouterModelCatalog.list_models = original


def test_auto_discovery_assigns_balanced_for_long_context() -> None:
    discovered = [
        make_metadata(
            "dynamic-long-context:free",
            context_length=64_000,
        ),
    ]

    original = OpenRouterModelCatalog.list_models

    try:
        OpenRouterModelCatalog.list_models = (
            lambda self, output_modalities="text": discovered
        )

        factory = AgentsModelFactory(
            provider_config=FakeOpenRouterConfig(),
        )

        registered = (
            factory.discover_and_register_models_auto()
        )

        assert len(registered) == 1
        assert registered[0].tier == ModelTier.BALANCED

    finally:
        OpenRouterModelCatalog.list_models = original


def test_auto_discovery_assigns_fast_to_plain_model() -> None:
    discovered = [
        make_metadata(
            "dynamic-fast:free",
            context_length=16_000,
        ),
    ]

    original = OpenRouterModelCatalog.list_models

    try:
        OpenRouterModelCatalog.list_models = (
            lambda self, output_modalities="text": discovered
        )

        factory = AgentsModelFactory(
            provider_config=FakeOpenRouterConfig(),
        )

        registered = (
            factory.discover_and_register_models_auto()
        )

        assert len(registered) == 1
        assert registered[0].tier == ModelTier.FAST

    finally:
        OpenRouterModelCatalog.list_models = original


def test_reasoning_tier_takes_priority_over_balanced_features() -> None:
    discovered = [
        make_metadata(
            "dynamic-advanced:free",
            context_length=128_000,
            supported_parameters=(
                "reasoning",
                "structured_outputs",
                "tools",
            ),
        ),
    ]

    original = OpenRouterModelCatalog.list_models

    try:
        OpenRouterModelCatalog.list_models = (
            lambda self, output_modalities="text": discovered
        )

        factory = AgentsModelFactory(
            provider_config=FakeOpenRouterConfig(),
        )

        registered = (
            factory.discover_and_register_models_auto()
        )

        assert len(registered) == 1
        assert registered[0].tier == ModelTier.REASONING

    finally:
        OpenRouterModelCatalog.list_models = original


def test_auto_discovery_preserves_configured_models() -> None:
    discovered = [
        make_metadata(
            "configured-balanced:free",
        ),
        make_metadata(
            "dynamic-new-model:free",
        ),
    ]

    original = OpenRouterModelCatalog.list_models

    try:
        OpenRouterModelCatalog.list_models = (
            lambda self, output_modalities="text": discovered
        )

        factory = AgentsModelFactory(
            provider_config=FakeOpenRouterConfig(),
        )

        registered = (
            factory.discover_and_register_models_auto()
        )

        registered_names = {
            candidate.name
            for candidate in registered
        }

        assert "configured-balanced:free" not in (
            registered_names
        )

        assert "dynamic-new-model:free" in (
            registered_names
        )

        pool_names = get_pool_model_names(factory)

        assert "configured-balanced:free" in pool_names
        assert "dynamic-new-model:free" in pool_names

    finally:
        OpenRouterModelCatalog.list_models = original


def test_auto_discovery_skips_duplicate_models() -> None:
    discovered = [
        make_metadata(
            "duplicate-model:free",
        ),
        make_metadata(
            "duplicate-model:free",
        ),
    ]

    original = OpenRouterModelCatalog.list_models

    try:
        OpenRouterModelCatalog.list_models = (
            lambda self, output_modalities="text": discovered
        )

        factory = AgentsModelFactory(
            provider_config=FakeOpenRouterConfig(),
        )

        registered = (
            factory.discover_and_register_models_auto()
        )

        assert [
            candidate.name
            for candidate in registered
        ] == [
            "duplicate-model:free",
        ]

    finally:
        OpenRouterModelCatalog.list_models = original


def test_auto_discovery_rejects_paid_models_by_default() -> None:
    discovered = [
        make_metadata(
            "paid-model",
            is_free=False,
        ),
    ]

    original = OpenRouterModelCatalog.list_models

    try:
        OpenRouterModelCatalog.list_models = (
            lambda self, output_modalities="text": discovered
        )

        factory = AgentsModelFactory(
            provider_config=FakeOpenRouterConfig(),
        )

        registered = (
            factory.discover_and_register_models_auto()
        )

        assert registered == []

    finally:
        OpenRouterModelCatalog.list_models = original


def test_auto_discovery_can_allow_paid_models() -> None:
    discovered = [
        make_metadata(
            "paid-model",
            is_free=False,
        ),
    ]

    original = OpenRouterModelCatalog.list_models

    try:
        OpenRouterModelCatalog.list_models = (
            lambda self, output_modalities="text": discovered
        )

        factory = AgentsModelFactory(
            provider_config=FakeOpenRouterConfig(),
        )

        registered = (
            factory.discover_and_register_models_auto(
                require_free=False,
            )
        )

        assert len(registered) == 1
        assert registered[0].name == "paid-model"

    finally:
        OpenRouterModelCatalog.list_models = original


def test_auto_discovery_failure_preserves_configured_pool() -> None:
    original = OpenRouterModelCatalog.list_models

    try:
        def raise_catalog_failure(
            self,
            output_modalities="text",
        ):
            raise ModelCatalogUnavailableError(
                "catalog unavailable"
            )

        OpenRouterModelCatalog.list_models = (
            raise_catalog_failure
        )

        factory = AgentsModelFactory(
            provider_config=FakeOpenRouterConfig(),
        )

        registered = (
            factory.discover_and_register_models_auto()
        )

        assert registered == []

        pool_names = get_pool_model_names(factory)

        assert "configured-fast:free" in pool_names
        assert "configured-balanced:free" in pool_names
        assert "configured-reasoning:free" in pool_names

    finally:
        OpenRouterModelCatalog.list_models = original


def test_auto_discovery_skips_non_openrouter_provider() -> None:
    class FakeLocalConfig:
        provider = "local"

        def get_models(
            self,
            tier: ModelTier,
        ) -> list[str]:
            return ["local-model"]

    factory = AgentsModelFactory(
        provider_config=FakeLocalConfig(),
    )

    registered = (
        factory.discover_and_register_models_auto()
    )

    assert registered == []


def test_auto_discovery_rejects_invalid_context_requirement() -> None:
    factory = AgentsModelFactory(
        provider_config=FakeOpenRouterConfig(),
    )

    try:
        factory.discover_and_register_models_auto(
            minimum_context_length=0,
        )
    except ValueError as exc:
        assert "minimum_context_length" in str(exc)
    else:
        raise AssertionError(
            "Expected ValueError for invalid "
            "minimum_context_length"
        )


def test_auto_discovered_candidate_has_catalog_metadata() -> None:
    discovered = [
        make_metadata(
            "dynamic-metadata-model:free",
            context_length=64_000,
            supported_parameters=(
                "structured_outputs",
            ),
        ),
    ]

    original = OpenRouterModelCatalog.list_models

    try:
        OpenRouterModelCatalog.list_models = (
            lambda self, output_modalities="text": discovered
        )

        factory = AgentsModelFactory(
            provider_config=FakeOpenRouterConfig(),
        )

        registered = (
            factory.discover_and_register_models_auto()
        )

        assert len(registered) == 1

        candidate = registered[0]

        assert candidate.metadata[
            "context_length"
        ] == 64_000

        assert candidate.metadata[
            "supported_parameters"
        ] == (
            "structured_outputs",
        )

        assert candidate.capabilities.structured_output is True
        assert candidate.capabilities.long_context is True

    finally:
        OpenRouterModelCatalog.list_models = original


def test_auto_discovered_model_is_retrievable_from_pool() -> None:
    discovered = [
        make_metadata(
            "dynamic-retrievable:free",
        ),
    ]

    original = OpenRouterModelCatalog.list_models

    try:
        OpenRouterModelCatalog.list_models = (
            lambda self, output_modalities="text": discovered
        )

        factory = AgentsModelFactory(
            provider_config=FakeOpenRouterConfig(),
        )

        registered = (
            factory.discover_and_register_models_auto()
        )

        assert len(registered) == 1

        candidate = get_pool_candidate(
            factory,
            "dynamic-retrievable:free",
        )

        assert candidate is not None
        assert candidate.name == (
            "dynamic-retrievable:free"
        )
        assert candidate.tier == ModelTier.FAST

    finally:
        OpenRouterModelCatalog.list_models = original