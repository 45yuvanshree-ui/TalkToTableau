from __future__ import annotations

from time import perf_counter
from typing import Callable, TypeVar

from agents import Agent, Model, OpenAIProvider, Runner

from .fallback import (
    FallbackResult,
    LLMFallbackExecutor,
    build_fallback_attempts,
)
from .models import ModelTier
from .model_pool import (
    ModelCandidate,
    ModelCapabilities,
    ModelPool,
    ModelSelectionRequest,
    RankedModel,
)
from .provider import LLMProviderConfig
from .model_catalog import (
    ModelCatalogError,
    OpenRouterModelCatalog,
    OpenRouterModelMetadata,
)
from .model_catalog_adapter import (
    metadata_to_candidate,
)
from .model_eligibility import (
    ModelEligibilityPolicy,
    ModelEligibilityRequest,
)
from .model_tiering import (
    ModelTieringPolicy,
)


T = TypeVar("T")


class AgentsModelFactory:
    """
    Build and execute OpenAI Agents SDK models using
    TalkToTableau's provider configuration.

    The factory is responsible for:

    - resolving logical model tiers
    - creating Agents SDK models
    - building the Omni Router model pool
    - selecting compatible models from the pool
    - providing bounded provider/model fallback
    - recording model health
    - optionally discovering additional OpenRouter models
    - automatically assigning discovered models to routing tiers

    It does not decide whether a request should be FAST,
    BALANCED, or REASONING from the user's business intent.
    That decision belongs to AdaptiveRouter.

    ModelTieringPolicy only classifies discovered provider models
    using deterministic provider metadata.
    """

    def __init__(
        self,
        provider_config: LLMProviderConfig | None = None,
        model_pool: ModelPool | None = None,
    ) -> None:
        self.provider_config = (
            provider_config or LLMProviderConfig()
        )

        self.fallback_executor = LLMFallbackExecutor()

        self.model_pool = (
            model_pool
            if model_pool is not None
            else self._build_model_pool()
        )

        self.model_eligibility_policy = (
            ModelEligibilityPolicy()
        )

        self.model_tiering_policy = (
            ModelTieringPolicy()
        )

    def create_model(
        self,
        model_name: str | ModelTier | None = None,
    ) -> Model:
        """
        Create an Agents SDK model.

        Backward compatibility:

        - If a concrete model-name string is supplied, it is used
          directly.
        - If a ModelTier is supplied, the appropriate model is
          selected through create_model_for_tier().
        - If nothing is supplied, the BALANCED configured model
          is used.

        This distinction is important because ModelTier values
        such as ``ModelTier.FAST`` are logical routing tiers, not
        concrete provider model IDs.
        """

        if isinstance(model_name, ModelTier):
            return self.create_model_for_tier(
                model_name
            )

        selected_model = (
            model_name
            if model_name is not None
            else self.provider_config.get_model(
                ModelTier.BALANCED
            )
        )

        return self._create_model_from_name(
            model_name=selected_model,
        )

    def create_model_for_tier(
        self,
        tier: ModelTier,
    ) -> Model:
        """
        Create a model for the requested logical routing tier.

        The ModelPool is consulted first. If no suitable pool model
        exists for the requested tier, the configured provider model
        for that tier is used.

        The returned Agents SDK model always receives a concrete
        provider model ID, never a logical tier name.
        """

        try:
            model, _model_name = self.select_model_for_tier(
                tier
            )

            return model

        except RuntimeError:
            selected_model = self.provider_config.get_model(
                tier
            )

            return self._create_model_from_name(
                model_name=selected_model,
            )

    def rank_models(
        self,
        *,
        tier: ModelTier,
        reasoning_required: bool = False,
        structured_output_required: bool = False,
        tool_calling_required: bool = False,
        sql_generation_required: bool = True,
        prefer_free: bool = True,
    ) -> list[RankedModel]:
        """
        Return compatible models ranked by the Omni Router model pool.

        The requested logical tier is used as the primary selection
        preference.

        This method does not make a network request.
        """

        request = ModelSelectionRequest(
            tier=tier,
            reasoning_required=reasoning_required,
            structured_output_required=(
                structured_output_required
            ),
            tool_calling_required=tool_calling_required,
            sql_generation_required=(
                sql_generation_required
            ),
            prefer_free=prefer_free,
        )

        return self.model_pool.rank(
            request
        )

    def select_model_name(
        self,
        *,
        tier: ModelTier,
        reasoning_required: bool = False,
        structured_output_required: bool = False,
        tool_calling_required: bool = False,
        sql_generation_required: bool = True,
        prefer_free: bool = True,
    ) -> str:
        """
        Select the highest-ranked compatible model.

        This is a deterministic local selection operation based on
        the current model-pool state.

        It does not contact the LLM provider.
        """

        ranked_models = self.rank_models(
            tier=tier,
            reasoning_required=reasoning_required,
            structured_output_required=(
                structured_output_required
            ),
            tool_calling_required=tool_calling_required,
            sql_generation_required=(
                sql_generation_required
            ),
            prefer_free=prefer_free,
        )

        if not ranked_models:
            raise RuntimeError(
                "No available LLM model matches the requested "
                "capabilities."
            )

        return ranked_models[0].candidate.name

    def select_model_for_tier(
        self,
        tier: ModelTier,
    ) -> tuple[Model, str]:
        """
        Select a model from the Omni Router pool and create it.

        This method is the factory's first direct integration point
        with ModelPool.

        The selected candidate is required to belong to the
        requested logical tier. This prevents health differences
        between unrelated tiers from silently changing the logical
        routing decision made by AdaptiveRouter.
        """

        ranked_models = self.rank_models(
            tier=tier,
        )

        tier_models = [
            ranked
            for ranked in ranked_models
            if ranked.candidate.tier == tier
        ]

        if not tier_models:
            raise RuntimeError(
                f"No available LLM model is configured for tier: "
                f"{tier.value}"
            )

        selected = tier_models[0]

        model = self._create_model_from_name(
            model_name=selected.candidate.name,
        )

        return (
            model,
            selected.candidate.name,
        )

    def record_model_success(
        self,
        *,
        tier: ModelTier,
        model_name: str,
        latency_ms: float | None = None,
    ) -> None:
        """
        Record a successful model execution in the model pool.
        """

        candidate = self.model_pool.get_candidate(
            provider=self.provider_config.provider,
            model_name=model_name,
        )

        if candidate is None:
            return

        self.model_pool.record_success(
            provider=candidate.provider,
            model_name=candidate.name,
            latency_ms=latency_ms,
        )

    def record_model_provider_failure(
        self,
        *,
        tier: ModelTier,
        model_name: str,
        cooldown_seconds: int | None = None,
    ) -> None:
        """
        Record a provider/transient failure in the model pool.

        The tier argument is retained so callers can keep the
        logical routing context alongside the concrete model.
        """

        candidate = self.model_pool.get_candidate(
            provider=self.provider_config.provider,
            model_name=model_name,
        )

        if candidate is None:
            return

        self.model_pool.record_provider_failure(
            provider=candidate.provider,
            model_name=candidate.name,
            cooldown_seconds=cooldown_seconds,
        )

    def record_model_semantic_failure(
        self,
        *,
        tier: ModelTier,
        model_name: str,
    ) -> None:
        """
        Record a semantic/application failure in the model pool.

        Semantic failures do not temporarily disable the model.
        They reduce its historical health score.
        """

        candidate = self.model_pool.get_candidate(
            provider=self.provider_config.provider,
            model_name=model_name,
        )

        if candidate is None:
            return

        self.model_pool.record_semantic_failure(
            provider=candidate.provider,
            model_name=candidate.name,
        )

    def refresh_model_pool(self) -> None:
        """
        Rebuild the model pool from the current provider
        configuration.

        This method remains local and deterministic.

        Dynamic provider discovery is intentionally separate and
        is performed through discover_and_register_models().
        """

        self.model_pool = self._build_model_pool()

    def discover_and_register_models(
        self,
        *,
        tier: ModelTier,
        require_free: bool = True,
        require_reasoning: bool = False,
        require_structured_output: bool = False,
        require_tool_calling: bool = False,
        minimum_context_length: int | None = None,
        priority: int = 0,
    ) -> list[ModelCandidate]:
        """
        Discover eligible OpenRouter models and register them in
        the existing ModelPool.

        Discovery is additive:

        - configured models remain in the pool
        - discovered models are added when eligible
        - duplicates are skipped
        - unsuitable models are skipped
        - provider/catalog failures do not break the application

        The logical tier is supplied explicitly by the caller.
        The catalog itself does not claim to determine FAST,
        BALANCED, or REASONING quality.

        This method is intentionally limited to OpenRouter because
        the current dynamic catalog implementation is provider-specific.
        """

        if self.provider_config.provider != "openrouter":
            return []

        if (
            minimum_context_length is not None
            and minimum_context_length < 1
        ):
            raise ValueError(
                "minimum_context_length must be >= 1."
            )

        try:
            catalog = OpenRouterModelCatalog(
                api_key=self.provider_config.get_api_key(),
                base_url=self.provider_config.get_base_url(),
            )

            discovered_models = catalog.list_models(
                output_modalities="text",
            )

        except (
            ModelCatalogError,
            ValueError,
        ):
            return []

        configured_model_ids = self._configured_model_ids()

        eligibility_request = ModelEligibilityRequest(
            require_text_generation=True,
            require_text_output=True,
            require_sql_generation=True,
            require_free=require_free,
            require_reasoning=require_reasoning,
            require_structured_output=(
                require_structured_output
            ),
            require_tool_calling=require_tool_calling,
            minimum_context_length=minimum_context_length,
        )

        eligible_models = (
            self.model_eligibility_policy.filter_unique_models(
                discovered_models,
                eligibility_request,
                excluded_model_ids=configured_model_ids,
            )
        )

        registered_candidates: list[
            ModelCandidate
        ] = []

        for metadata in eligible_models:
            candidate = metadata_to_candidate(
                metadata,
                tier=tier,
                priority=priority,
            )

            self.model_pool.register(
                candidate
            )

            registered_candidates.append(
                candidate
            )

        return registered_candidates

    def discover_and_register_models_auto(
        self,
        *,
        require_free: bool = True,
        require_reasoning: bool = False,
        require_structured_output: bool = False,
        require_tool_calling: bool = False,
        minimum_context_length: int | None = None,
        priority: int = 0,
    ) -> list[ModelCandidate]:
        """
        Discover eligible OpenRouter models and automatically
        assign each model to FAST, BALANCED, or REASONING.

        Tier assignment is deterministic and based only on the
        provider metadata:

        - reasoning capability -> REASONING
        - structured output, tool calling, or sufficiently long
          context -> BALANCED
        - otherwise -> FAST

        Existing configured models are never replaced.

        Discovery is additive and provider-specific to OpenRouter.
        If catalog discovery fails, the configured model pool
        remains operational and this method returns an empty list.
        """

        if self.provider_config.provider != "openrouter":
            return []

        if (
            minimum_context_length is not None
            and minimum_context_length < 1
        ):
            raise ValueError(
                "minimum_context_length must be >= 1."
            )

        try:
            catalog = OpenRouterModelCatalog(
                api_key=self.provider_config.get_api_key(),
                base_url=self.provider_config.get_base_url(),
            )

            discovered_models = catalog.list_models(
                output_modalities="text",
            )

        except (
            ModelCatalogError,
            ValueError,
        ):
            return []

        configured_model_ids = self._configured_model_ids()

        eligibility_request = ModelEligibilityRequest(
            require_text_generation=True,
            require_text_output=True,
            require_sql_generation=True,
            require_free=require_free,
            require_reasoning=require_reasoning,
            require_structured_output=(
                require_structured_output
            ),
            require_tool_calling=require_tool_calling,
            minimum_context_length=minimum_context_length,
        )

        eligible_models = (
            self.model_eligibility_policy.filter_unique_models(
                discovered_models,
                eligibility_request,
                excluded_model_ids=configured_model_ids,
            )
        )

        registered_candidates: list[
            ModelCandidate
        ] = []

        for metadata in eligible_models:
            assigned_tier = (
                self.model_tiering_policy.assign_tier(
                    metadata
                )
            )

            candidate = metadata_to_candidate(
                metadata,
                tier=assigned_tier,
                priority=priority,
            )

            self.model_pool.register(
                candidate
            )

            registered_candidates.append(
                candidate
            )

        return registered_candidates

    def _configured_model_ids(self) -> set[str]:
        """
        Return all concrete model IDs currently configured through
        provider environment variables.
        """

        configured: set[str] = set()

        for tier in ModelTier:
            for model_name in self.provider_config.get_models(
                tier
            ):
                model_name = model_name.strip()

                if model_name:
                    configured.add(model_name)

        return configured

    def run_with_fallback(
        self,
        *,
        primary_tier: ModelTier,
        execute: Callable[[Model, ModelTier, str], T],
        fallback_tiers: list[ModelTier] | None = None,
    ) -> FallbackResult[T]:
        """
        Execute an Agents SDK operation using bounded model fallback.

        This is the existing backward-compatible fallback path.

        Provider/transient failures are eligible for fallback.

        Semantic/application errors are propagated and are NOT
        silently retried through another model.
        """

        primary_model_name = self.provider_config.get_model(
            primary_tier
        )

        fallback_pairs: list[tuple[ModelTier, str]] = []

        if fallback_tiers:
            for tier in fallback_tiers:
                model_name = self.provider_config.get_model(
                    tier
                )

                fallback_pairs.append(
                    (
                        tier,
                        model_name,
                    )
                )

        attempts = build_fallback_attempts(
            primary_tier=primary_tier,
            primary_model=primary_model_name,
            fallback_tiers=fallback_pairs,
        )

        def execute_attempt(
            tier: ModelTier,
            model_name: str,
        ) -> T:
            model = self._create_model_from_name(
                model_name=model_name,
            )

            return execute(
                model,
                tier,
                model_name,
            )

        return self.fallback_executor.run(
            attempts=attempts,
            execute=execute_attempt,
        )

    def run_agent_with_fallback(
        self,
        *,
        primary_tier: ModelTier,
        agent_name: str,
        instructions: str,
        input_text: str,
        fallback_tiers: list[ModelTier] | None = None,
    ):
        """
        Convenience wrapper for executing a real Agents SDK Agent
        through Runner.run_sync() with bounded fallback.

        A new Agent is created for each model attempt so that the
        fallback model is actually used by the Agents SDK.

        This method preserves the original fallback API and remains
        separate from the Omni Router model-pool execution path.
        """

        def execute(
            model: Model,
            _tier: ModelTier,
            _model_name: str,
        ):
            agent = Agent(
                name=agent_name,
                instructions=instructions,
                model=model,
            )

            return Runner.run_sync(
                agent,
                input_text,
            )

        return self.run_with_fallback(
            primary_tier=primary_tier,
            fallback_tiers=fallback_tiers,
            execute=execute,
        )

    def run_with_model_pool(
        self,
        *,
        tier: ModelTier,
        execute: Callable[[Model, ModelTier, str], T],
        reasoning_required: bool = False,
        structured_output_required: bool = False,
        tool_calling_required: bool = False,
        sql_generation_required: bool = True,
        prefer_free: bool = True,
        max_attempts: int = 3,
    ) -> FallbackResult[T]:
        """
        Execute an operation using the ranked Omni Router model pool.

        The pool determines the initial model ordering.

        Provider/transient failures:

        - record provider failure
        - temporarily remove the model from selection
        - continue to the next ranked candidate

        Known semantic/application failures:

        - record semantic failure
        - continue to the next candidate

        Unknown/internal failures:

        - do not modify model health
        - propagate immediately

        Successful execution:

        - record model health and latency
        - return immediately

        The number of model attempts is bounded to prevent
        uncontrolled retry loops.
        """

        if max_attempts < 1:
            raise ValueError(
                "max_attempts must be at least 1."
            )

        if max_attempts > 3:
            raise ValueError(
                "max_attempts cannot exceed 3."
            )

        request = ModelSelectionRequest(
            tier=tier,
            reasoning_required=reasoning_required,
            structured_output_required=(
                structured_output_required
            ),
            tool_calling_required=tool_calling_required,
            sql_generation_required=(
                sql_generation_required
            ),
            prefer_free=prefer_free,
        )

        ranked_models = self.model_pool.rank(
            request
        )

        if not ranked_models:
            raise RuntimeError(
                "No available LLM model matches the requested "
                "capabilities."
            )

        selected_models = ranked_models[
            :max_attempts
        ]

        attempts: list[tuple[ModelTier, str]] = [
            (
                ranked.candidate.tier,
                ranked.candidate.name,
            )
            for ranked in selected_models
        ]

        def execute_attempt(
            attempt_tier: ModelTier,
            model_name: str,
        ) -> T:
            started = perf_counter()

            try:
                model = self._create_model_from_name(
                    model_name=model_name,
                )

                result = execute(
                    model,
                    attempt_tier,
                    model_name,
                )

                latency_ms = (
                    perf_counter() - started
                ) * 1000.0

                self.record_model_success(
                    tier=attempt_tier,
                    model_name=model_name,
                    latency_ms=latency_ms,
                )

                return result

            except Exception as error:
                latency_ms = (
                    perf_counter() - started
                ) * 1000.0

                self._record_failure_from_exception(
                    tier=attempt_tier,
                    model_name=model_name,
                    latency_ms=latency_ms,
                    error=error,
                )

                raise

        return self.fallback_executor.run(
            attempts=attempts,
            execute=execute_attempt,
        )

    def run_agent_with_model_pool(
        self,
        *,
        tier: ModelTier,
        agent_name: str,
        instructions: str,
        input_text: str,
        reasoning_required: bool = False,
        structured_output_required: bool = False,
        tool_calling_required: bool = False,
        sql_generation_required: bool = True,
        prefer_free: bool = True,
        max_attempts: int = 3,
    ):
        """
        Convenience wrapper around run_with_model_pool() for a real
        Agents SDK Agent.

        A new Agent is created for every model attempt.
        """

        def execute(
            model: Model,
            _tier: ModelTier,
            _model_name: str,
        ):
            agent = Agent(
                name=agent_name,
                instructions=instructions,
                model=model,
            )

            return Runner.run_sync(
                agent,
                input_text,
            )

        return self.run_with_model_pool(
            tier=tier,
            execute=execute,
            reasoning_required=reasoning_required,
            structured_output_required=(
                structured_output_required
            ),
            tool_calling_required=tool_calling_required,
            sql_generation_required=(
                sql_generation_required
            ),
            prefer_free=prefer_free,
            max_attempts=max_attempts,
        )

    def _record_failure_from_exception(
        self,
        *,
        tier: ModelTier,
        model_name: str,
        latency_ms: float,
        error: Exception,
    ) -> None:
        """
        Classify an execution exception and update model health.

        Provider/transient failures temporarily remove the model.

        Known semantic/application failures reduce the model's
        historical health score.

        Unexpected internal failures do not modify model health,
        because there is insufficient evidence that the selected
        model caused the problem.
        """

        candidate = self.model_pool.get_candidate(
            provider=self.provider_config.provider,
            model_name=model_name,
        )

        if candidate is None:
            return

        if LLMFallbackExecutor.is_provider_failure(
            error
        ):
            self.record_model_provider_failure(
                tier=tier,
                model_name=model_name,
            )
            return

        if self._is_semantic_failure(error):
            self.record_model_semantic_failure(
                tier=tier,
                model_name=model_name,
            )
            return

        return

    @staticmethod
    def _is_semantic_failure(
        error: Exception,
    ) -> bool:
        """
        Determine whether an exception represents a known
        model/application semantic failure.

        This classifier is intentionally conservative.
        """

        semantic_exception_names = {
            "ValidationError",
            "OutputValidationError",
            "StructuredOutputError",
            "SQLValidationError",
            "SQLExecutionError",
            "EvidenceValidationError",
            "VisualizationValidationError",
            "StoryValidationError",
        }

        if error.__class__.__name__ in semantic_exception_names:
            return True

        return False

    def _build_model_pool(self) -> ModelPool:
        """
        Build the local Omni Router model pool from provider
        configuration.

        Every configured concrete model becomes one ModelCandidate.
        """

        candidates: list[ModelCandidate] = []

        for tier in ModelTier:
            model_names = self.provider_config.get_models(
                tier
            )

            for priority, model_name in enumerate(
                model_names,
                start=1,
            ):
                capabilities = ModelCapabilities(
                    text_generation=True,
                    structured_output=False,
                    tool_calling=False,
                    reasoning=(
                        tier == ModelTier.REASONING
                    ),
                    sql_generation=True,
                    long_context=False,
                )

                candidate = ModelCandidate(
                    name=model_name,
                    provider=self.provider_config.provider,
                    tier=tier,
                    is_free=model_name.endswith(
                        ":free"
                    ),
                    capabilities=capabilities,
                    priority=max(
                        0,
                        11 - priority,
                    ),
                    metadata={
                        "configured_tier": tier.value,
                        "source": "provider_config",
                    },
                )

                candidates.append(
                    candidate
                )

        return ModelPool(
            candidates=candidates,
        )

    def _create_model_from_name(
        self,
        *,
        model_name: str,
    ) -> Model:
        """
        Create an Agents SDK model from a concrete model name.

        Provider configuration is resolved once from the factory,
        while the concrete model name may change during fallback.
        """

        base_url = self.provider_config.get_base_url()
        api_key = self.provider_config.get_api_key()

        provider = OpenAIProvider(
            api_key=api_key,
            base_url=base_url,
            use_responses=False,
            strict_feature_validation=True,
        )

        return provider.get_model(
            model_name
        )


def build_model_pool(
    provider_config: LLMProviderConfig | None = None,
) -> ModelPool:
    """
    Build a ModelPool directly from provider configuration.

    This convenience function is useful for diagnostics and tests
    without constructing an AgentsModelFactory.
    """

    factory = AgentsModelFactory(
        provider_config=provider_config,
    )

    return factory.model_pool