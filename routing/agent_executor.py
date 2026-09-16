"""
Agents SDK execution adapter for TalkToTableau.

This module connects the provider-neutral fallback engine with
OpenAI Agents SDK agents.

The adapter deliberately separates:

1. Model selection
2. Agent construction
3. Runner execution
4. Provider fallback

It does not perform semantic validation, SQL validation, insight
validation, or story validation.
"""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter
from typing import Callable, TypeVar

from agents import Agent, ModelSettings, Runner
from agents.model_settings import ModelRetrySettings

from .fallback import FallbackResult
from .models import ModelTier
from .agents_model import AgentsModelFactory


T = TypeVar("T")


AgentFactory = Callable[[object], Agent]


# Maximum time allowed for one individual LLM model-call attempt.
#
# This is intentionally separate from the fallback attempt limit.
# If a provider hangs, the Agents SDK will raise ModelTimeoutError
# and the outer LLMFallbackExecutor can move to the next model.
MODEL_CALL_TIMEOUT_SECONDS = 60.0


class AgentsSDKExecutor:
    """
    Execute an Agents SDK agent factory with bounded
    provider/model fallback.

    The caller supplies a function that knows how to construct
    the required agent from a model.

    The executor applies two reliability controls:

    1. A bounded timeout for each individual model call.
    2. SDK-level model retries are disabled because the project's
       LLMFallbackExecutor owns provider/model fallback.

    Example:

        executor.run(
            primary_tier=ModelTier.BALANCED,
            fallback_tiers=[ModelTier.FAST],
            agent_factory=create_business_analyst,
            prompt=prompt,
        )
    """

    def __init__(
        self,
        model_factory: AgentsModelFactory | None = None,
    ) -> None:
        self.model_factory = (
            model_factory
            if model_factory is not None
            else AgentsModelFactory()
        )

    def run(
        self,
        *,
        primary_tier: ModelTier,
        agent_factory: AgentFactory,
        prompt: str,
        fallback_tiers: list[ModelTier] | None = None,
    ) -> FallbackResult:
        """
        Execute an Agents SDK agent with bounded fallback.

        Args:
            primary_tier:
                Tier selected by AdaptiveRouter.

            agent_factory:
                Existing project function that receives a model
                and returns an Agents SDK Agent.

            prompt:
                User/task prompt passed to Runner.run_sync().

            fallback_tiers:
                Ordered fallback tiers.

        Returns:
            FallbackResult containing the Runner result and
            model-attempt metadata.

        Reliability behavior:

        Each individual model call is limited to
        MODEL_CALL_TIMEOUT_SECONDS.

        Agents SDK automatic model retries are disabled so that
        TalkToTableau's LLMFallbackExecutor remains the single
        owner of provider/model fallback.
        """

        def execute(
            model,
            tier: ModelTier,
            model_name: str,
        ):
            print(
                "\n"
                + "-" * 70
                + "\n"
                + "LLM ATTEMPT START"
                + "\n"
                + f"Tier: {tier.value}"
                + "\n"
                + f"Model: {model_name}"
                + "\n"
                + "-" * 70
            )

            started = perf_counter()

            try:
                agent = agent_factory(
                    model
                )

                agent = self._apply_execution_limits(
                    agent
                )

                result = Runner.run_sync(
                    agent,
                    prompt,
                )

                elapsed_seconds = (
                    perf_counter() - started
                )

                print(
                    "\n"
                    + "-" * 70
                    + "\n"
                    + "LLM ATTEMPT SUCCESS"
                    + "\n"
                    + f"Tier: {tier.value}"
                    + "\n"
                    + f"Model: {model_name}"
                    + "\n"
                    + f"Duration: {elapsed_seconds:.2f} seconds"
                    + "\n"
                    + "-" * 70
                )

                return result

            except Exception as error:
                elapsed_seconds = (
                    perf_counter() - started
                )

                print(
                    "\n"
                    + "-" * 70
                    + "\n"
                    + "LLM ATTEMPT FAILED"
                    + "\n"
                    + f"Tier: {tier.value}"
                    + "\n"
                    + f"Model: {model_name}"
                    + "\n"
                    + f"Duration: {elapsed_seconds:.2f} seconds"
                    + "\n"
                    + f"Error: {type(error).__name__}"
                    + "\n"
                    + f"Message: {error}"
                    + "\n"
                    + "-" * 70
                )

                raise

        return self.model_factory.run_with_fallback(
            primary_tier=primary_tier,
            fallback_tiers=fallback_tiers,
            execute=execute,
        )

    @staticmethod
    def _apply_execution_limits(
        agent: Agent,
    ) -> Agent:
        """
        Apply bounded model-call execution settings.

        Existing agent model settings are preserved.

        Only the reliability controls owned by this adapter are
        changed:

        - timeout: bounded model-call duration
        - retry.max_retries: zero, because the outer fallback
          executor owns provider/model retries
        """

        current_settings = agent.model_settings

        current_retry = getattr(
            current_settings,
            "retry",
            None,
        )

        if current_retry is None:
            retry_settings = ModelRetrySettings(
                max_retries=0,
            )
        else:
            retry_settings = replace(
                current_retry,
                max_retries=0,
            )

        return replace(
            agent,
            model_settings=replace(
                current_settings,
                timeout=MODEL_CALL_TIMEOUT_SECONDS,
                retry=retry_settings,
            ),
        )


def run_agent_with_fallback(
    *,
    model_factory: AgentsModelFactory,
    primary_tier: ModelTier,
    agent_factory: AgentFactory,
    prompt: str,
    fallback_tiers: list[ModelTier] | None = None,
) -> FallbackResult:
    """
    Functional convenience wrapper around AgentsSDKExecutor.

    This is useful when the orchestration layer already owns
    the AgentsModelFactory instance.
    """

    executor = AgentsSDKExecutor(
        model_factory=model_factory
    )

    return executor.run(
        primary_tier=primary_tier,
        fallback_tiers=fallback_tiers,
        agent_factory=agent_factory,
        prompt=prompt,
    )