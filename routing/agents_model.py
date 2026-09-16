from __future__ import annotations

from time import perf_counter
from typing import Callable, TypeVar

from agents import Agent, Model, OpenAIProvider, Runner
from openai import AsyncOpenAI

from .fallback import (
    FallbackResult,
    LLMFallbackExecutor,
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
    - discovering additional OpenRouter models
    - automatically assigning discovered models to routing tiers

    It does not decide whether a request should be FAST,
    BALANCED, or REASONING from the user's business intent.
    That decision belongs to AdaptiveRouter.
    """

    def __init__(
        self,
        provider_config: LLMProviderConfig | None = None,
        model_pool: ModelPool | None = None,
    ) -> None:
        self.provider_config = (
            provider_config or LLMProviderConfig()
        )

        self.fallback_executor = (
            LLMFallbackExecutor()
        )

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

    # ==============================================================
    # MODEL CREATION
    # ==============================================================

    def create_model(
        self,
        model_name: str | ModelTier | None = None,
    ) -> Model:
        """
        Create an Agents SDK model.

        If a concrete model name is supplied, it is used directly.

        If a ModelTier is supplied, the model is selected from
        that logical tier.

        If nothing is supplied, the configured BALANCED model
        is used.
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

        The ModelPool is consulted first.

        If no suitable pool model exists, the configured
        provider model for that tier is used.
        """

        try:
            model, _model_name = (
                self.select_model_for_tier(
                    tier
                )
            )

            return model

        except RuntimeError:
            selected_model = (
                self.provider_config.get_model(
                    tier
                )
            )

            return self._create_model_from_name(
                model_name=selected_model,
            )

    # ==============================================================
    # MODEL POOL SELECTION
    # ==============================================================

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

        The requested tier is used as the primary routing preference.

        This operation does not contact the provider.
        """

        request = ModelSelectionRequest(
            tier=tier,
            reasoning_required=(
                reasoning_required
            ),
            structured_output_required=(
                structured_output_required
            ),
            tool_calling_required=(
                tool_calling_required
            ),
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

        This is a local deterministic selection operation.
        """

        ranked_models = self.rank_models(
            tier=tier,
            reasoning_required=(
                reasoning_required
            ),
            structured_output_required=(
                structured_output_required
            ),
            tool_calling_required=(
                tool_calling_required
            ),
            sql_generation_required=(
                sql_generation_required
            ),
            prefer_free=prefer_free,
        )

        tier_models = [
            ranked
            for ranked in ranked_models
            if ranked.candidate.tier == tier
        ]

        if not tier_models:
            raise RuntimeError(
                "No available LLM model matches the "
                f"requested tier: {tier.value}"
            )

        return tier_models[0].candidate.name

    def select_model_for_tier(
        self,
        tier: ModelTier,
    ) -> tuple[Model, str]:
        """
        Select and create a model belonging to the requested
        logical tier.

        Candidates from unrelated tiers are never silently
        substituted here.
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
                "No available LLM model is configured "
                f"for tier: {tier.value}"
            )

        selected = tier_models[0]

        model = self._create_model_from_name(
            model_name=selected.candidate.name,
        )

        return (
            model,
            selected.candidate.name,
        )

    # ==============================================================
    # MODEL HEALTH
    # ==============================================================

    def record_model_success(
        self,
        *,
        tier: ModelTier,
        model_name: str,
        latency_ms: float | None = None,
    ) -> None:
        """
        Record successful execution of a concrete model.
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
        Record a provider/transient model failure.
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
        Record a semantic/application model failure.

        Semantic failures do not temporarily disable the model.
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
        """

        self.model_pool = self._build_model_pool()

    # ==============================================================
    # DYNAMIC MODEL DISCOVERY
    # ==============================================================

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
        Discover eligible OpenRouter models and register them
        in the existing ModelPool.

        Discovery is additive.
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
                api_key=(
                    self.provider_config.get_api_key()
                ),
                base_url=(
                    self.provider_config.get_base_url()
                ),
            )

            discovered_models = catalog.list_models(
                output_modalities="text",
            )

        except (
            ModelCatalogError,
            ValueError,
        ):
            return []

        configured_model_ids = (
            self._configured_model_ids()
        )

        eligibility_request = (
            ModelEligibilityRequest(
                require_text_generation=True,
                require_text_output=True,
                require_sql_generation=True,
                require_free=require_free,
                require_reasoning=(
                    require_reasoning
                ),
                require_structured_output=(
                    require_structured_output
                ),
                require_tool_calling=(
                    require_tool_calling
                ),
                minimum_context_length=(
                    minimum_context_length
                ),
            )
        )

        eligible_models = (
            self.model_eligibility_policy
            .filter_unique_models(
                discovered_models,
                eligibility_request,
                excluded_model_ids=(
                    configured_model_ids
                ),
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
        assign each discovered model to FAST, BALANCED, or
        REASONING.
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
                api_key=(
                    self.provider_config.get_api_key()
                ),
                base_url=(
                    self.provider_config.get_base_url()
                ),
            )

            discovered_models = catalog.list_models(
                output_modalities="text",
            )

        except (
            ModelCatalogError,
            ValueError,
        ):
            return []

        configured_model_ids = (
            self._configured_model_ids()
        )

        eligibility_request = (
            ModelEligibilityRequest(
                require_text_generation=True,
                require_text_output=True,
                require_sql_generation=True,
                require_free=require_free,
                require_reasoning=(
                    require_reasoning
                ),
                require_structured_output=(
                    require_structured_output
                ),
                require_tool_calling=(
                    require_tool_calling
                ),
                minimum_context_length=(
                    minimum_context_length
                ),
            )
        )

        eligible_models = (
            self.model_eligibility_policy
            .filter_unique_models(
                discovered_models,
                eligibility_request,
                excluded_model_ids=(
                    configured_model_ids
                ),
            )
        )

        registered_candidates: list[
            ModelCandidate
        ] = []

        for metadata in eligible_models:
            assigned_tier = (
                self.model_tiering_policy
                .assign_tier(metadata)
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

    def _configured_model_ids(
        self,
    ) -> set[str]:
        """
        Return all concrete model IDs currently configured
        through provider environment variables.
        """

        configured: set[str] = set()

        for tier in ModelTier:
            for model_name in (
                self.provider_config.get_models(
                    tier
                )
            ):
                model_name = model_name.strip()

                if model_name:
                    configured.add(
                        model_name
                    )

        return configured

    # ==============================================================
    # BOUNDED FALLBACK
    # ==============================================================

    def run_with_fallback(
        self,
        *,
        primary_tier: ModelTier,
        execute: Callable[
            [Model, ModelTier, str],
            T,
        ],
        fallback_tiers: list[ModelTier] | None = None,
    ) -> FallbackResult[T]:
        """
        Execute one concrete model for each requested logical tier.

        AdaptiveRouter determines the logical tier.

        For each tier:
        1. Use the explicitly configured model when available.
        2. If no configured model exists, use the best available model
           from ModelPool for that same tier.

        Fallback progression remains tier-by-tier and bounded by
        LLMFallbackExecutor.

        ModelPool alternatives are NOT added as separate fallback
        attempts because doing so could consume the bounded fallback
        budget before the next logical tier is attempted.
        """

        tiers = [
            primary_tier,
            *(
                fallback_tiers
                or []
            ),
        ]

        attempts: list[
            tuple[ModelTier, str]
        ] = []

        seen_tiers: set[ModelTier] = set()

        for tier in tiers:
            # ---------------------------------------------------------
            # Avoid duplicate logical tiers.
            # ---------------------------------------------------------
            if tier in seen_tiers:
                continue

            seen_tiers.add(tier)

            # ---------------------------------------------------------
            # 1. Prefer the explicitly configured model.
            #
            # This is important for:
            # - environment configuration
            # - deterministic tests
            # - predictable routing
            # ---------------------------------------------------------
            configured_model = (
                self.provider_config.get_model(
                    tier
                )
            )

            model_name: str | None = None

            if configured_model:
                model_name = configured_model

            # ---------------------------------------------------------
            # 2. If there is no configured model for this tier,
            #    select ONE model from ModelPool for this tier.
            #
            # Never add multiple pool candidates here because fallback
            # attempts are tier-level and bounded.
            # ---------------------------------------------------------
            if not model_name:
                try:
                    ranked_models = self.rank_models(
                        tier=tier
                    )
                except Exception:
                    ranked_models = []

                for ranked in ranked_models:
                    candidate = ranked.candidate

                    if candidate.tier != tier:
                        continue

                    model_name = candidate.name
                    break

            # ---------------------------------------------------------
            # 3. Add exactly one concrete model for this logical tier.
            # ---------------------------------------------------------
            if model_name:
                attempts.append(
                    (
                        tier,
                        model_name,
                    )
                )

        # -------------------------------------------------------------
        # No usable models at all.
        # -------------------------------------------------------------
        if not attempts:
            raise RuntimeError(
                "No LLM models are available for the requested tiers."
            )

        # -------------------------------------------------------------
        # Execute through the existing bounded fallback executor.
        # -------------------------------------------------------------
        def execute_attempt(
            tier: ModelTier,
            model_name: str,
        ) -> T:
            model = self._create_model_from_name(
                model_name=model_name
            )

            start_time = perf_counter()

            try:
                result = execute(
                    model,
                    tier,
                    model_name,
                )

                latency_ms = (
                    perf_counter()
                    - start_time
                ) * 1000.0

                self.record_model_success(
                    tier=tier,
                    model_name=model_name,
                    latency_ms=latency_ms,
                )

                return result

            except Exception as exc:
                latency_ms = (
                    perf_counter()
                    - start_time
                ) * 1000.0

                self._record_failure_from_exception(
                    tier=tier,
                    model_name=model_name,
                    latency_ms=latency_ms,
                    error=exc,
                )

                raise

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
    ) -> FallbackResult:
        """
        Run an OpenAI Agents SDK Agent with bounded model fallback.

        The requested logical tier is selected first. If a transient
        provider failure occurs, the fallback executor tries the next
        eligible model.
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
    # ==============================================================
    # DIRECT MODEL-POOL EXECUTION
    # ==============================================================

    def run_with_model_pool(
        self,
        *,
        tier: ModelTier,
        execute: Callable[
            [Model, ModelTier, str],
            T,
        ],
        reasoning_required: bool = False,
        structured_output_required: bool = False,
        tool_calling_required: bool = False,
        sql_generation_required: bool = True,
        prefer_free: bool = True,
        max_attempts: int = 3,
    ) -> FallbackResult[T]:
        """
        Execute an operation using the ranked Omni Router
        model pool.

        This path is intended for direct model-pool execution.

        Provider failures:
        - update provider health
        - temporarily remove the model
        - continue to another model

        Known semantic failures:
        - update semantic health
        - continue according to fallback policy

        Unknown/internal failures:
        - do not blame the model
        - propagate immediately

        Successful execution:
        - record latency and health
        - return immediately

        Attempts are bounded.
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
            reasoning_required=(
                reasoning_required
            ),
            structured_output_required=(
                structured_output_required
            ),
            tool_calling_required=(
                tool_calling_required
            ),
            sql_generation_required=(
                sql_generation_required
            ),
            prefer_free=prefer_free,
        )

        ranked_models = (
            self.model_pool.rank(
                request
            )
        )

        if not ranked_models:
            raise RuntimeError(
                "No available LLM model matches "
                "the requested capabilities."
            )

        # Direct model-pool execution must also respect
        # the requested logical tier.
        tier_models = [
            ranked
            for ranked in ranked_models
            if ranked.candidate.tier == tier
        ]

        if not tier_models:
            raise RuntimeError(
                "No available LLM model matches "
                f"the requested tier: {tier.value}"
            )

        selected_models = tier_models[
            :max_attempts
        ]

        attempts: list[
            tuple[ModelTier, str]
        ] = [
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
                model = (
                    self._create_model_from_name(
                        model_name=model_name,
                    )
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
        Execute an Agents SDK Agent through the direct
        model-pool execution path.
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
            reasoning_required=(
                reasoning_required
            ),
            structured_output_required=(
                structured_output_required
            ),
            tool_calling_required=(
                tool_calling_required
            ),
            sql_generation_required=(
                sql_generation_required
            ),
            prefer_free=prefer_free,
            max_attempts=max_attempts,
        )

    # ==============================================================
    # FAILURE CLASSIFICATION
    # ==============================================================

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

        Known semantic/application failures reduce historical
        model health.

        Unexpected internal failures do not modify model health.
        """

        candidate = (
            self.model_pool.get_candidate(
                provider=(
                    self.provider_config.provider
                ),
                model_name=model_name,
            )
        )

        if candidate is None:
            return

        if (
            LLMFallbackExecutor.is_provider_failure(
                error
            )
        ):
            self.record_model_provider_failure(
                tier=tier,
                model_name=model_name,
            )
            return

        if self._is_semantic_failure(
            error
        ):
            self.record_model_semantic_failure(
                tier=tier,
                model_name=model_name,
            )
            return

        # Unknown/internal errors are deliberately
        # not attributed to the selected model.

    @staticmethod
    def _is_semantic_failure(
        error: Exception,
    ) -> bool:
        """
        Conservative semantic/application failure classifier.
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

        return (
            error.__class__.__name__
            in semantic_exception_names
        )

    # ==============================================================
    # MODEL POOL CONSTRUCTION
    # ==============================================================

    def _build_model_pool(
        self,
    ) -> ModelPool:
        """
        Build the Omni Router model pool from provider
        configuration.

        Every configured concrete model becomes a candidate.

        Conservative capabilities are used until provider
        metadata discovery provides stronger information.
        """

        candidates: list[
            ModelCandidate
        ] = []

        for tier in ModelTier:
            model_names = (
                self.provider_config.get_models(
                    tier
                )
            )

            for priority, model_name in enumerate(
                model_names,
                start=1,
            ):
                capabilities = (
                    ModelCapabilities(
                        text_generation=True,
                        structured_output=False,
                        tool_calling=False,
                        reasoning=(
                            tier
                            == ModelTier.REASONING
                        ),
                        sql_generation=True,
                        long_context=False,
                    )
                )

                candidate = ModelCandidate(
                    name=model_name,
                    provider=(
                        self.provider_config.provider
                    ),
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
                        "configured_tier": (
                            tier.value
                        ),
                        "source": (
                            "provider_config"
                        ),
                    },
                )

                candidates.append(
                    candidate
                )

        return ModelPool(
            candidates=candidates,
        )

    # ==============================================================
    # AGENTS SDK MODEL CREATION
    # ==============================================================

    def _create_model_from_name(
        self,
        *,
        model_name: str,
    ) -> Model:
        """
        Create an Agents SDK model from a concrete provider
        model ID.

        The provider endpoint and credentials come from the
        provider configuration.

        The concrete model name can change during fallback.
        """

        base_url = (
            self.provider_config.get_base_url()
        )

        api_key = (
            self.provider_config.get_api_key()
        )

        client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=60.0,
            max_retries=0,
        )

        provider = OpenAIProvider(
            openai_client=client,
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
    """

    factory = AgentsModelFactory(
        provider_config=provider_config,
    )

    return factory.model_pool