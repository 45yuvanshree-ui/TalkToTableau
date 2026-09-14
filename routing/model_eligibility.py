from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .model_catalog import OpenRouterModelMetadata


@dataclass(frozen=True)
class ModelEligibilityRequest:
    """
    Requirements used to determine whether a discovered model
    is suitable for a TalkToTableau workload.
    """

    require_text_generation: bool = True

    require_text_output: bool = True

    require_sql_generation: bool = True

    require_free: bool = False

    require_reasoning: bool = False

    require_structured_output: bool = False

    require_tool_calling: bool = False

    minimum_context_length: int | None = None


@dataclass(frozen=True)
class ModelEligibilityResult:
    """
    Explainable result of an eligibility decision.
    """

    eligible: bool

    reasons: tuple[str, ...]


class ModelEligibilityPolicy:
    """
    Deterministic eligibility policy for dynamically discovered models.

    This class does not rank models and does not execute them.

    Its only responsibility is deciding whether a discovered model
    is suitable for consideration by the Omni Router.
    """

    def evaluate(
        self,
        model: OpenRouterModelMetadata,
        request: ModelEligibilityRequest,
    ) -> ModelEligibilityResult:
        """
        Evaluate one discovered model against the supplied requirements.
        """

        reasons: list[str] = []

        if request.require_text_generation:
            if not model.is_text_model:
                return ModelEligibilityResult(
                    eligible=False,
                    reasons=(
                        "model does not support text-to-text generation",
                    ),
                )

            reasons.append(
                "text-to-text generation supported"
            )

        if request.require_text_output:
            if "text" not in model.output_modalities:
                return ModelEligibilityResult(
                    eligible=False,
                    reasons=(
                        "model does not support text output",
                    ),
                )

            reasons.append(
                "text output supported"
            )

        if request.require_sql_generation:
            if not model.is_text_model:
                return ModelEligibilityResult(
                    eligible=False,
                    reasons=(
                        "model is not suitable for SQL generation",
                    ),
                )

            reasons.append(
                "text model is eligible for SQL generation"
            )

        if request.require_free:
            if not model.is_free:
                return ModelEligibilityResult(
                    eligible=False,
                    reasons=(
                        "free model required",
                    ),
                )

            reasons.append(
                "model is free"
            )

        if request.require_reasoning:
            if not model.supports_reasoning:
                return ModelEligibilityResult(
                    eligible=False,
                    reasons=(
                        "reasoning capability required",
                    ),
                )

            reasons.append(
                "reasoning capability supported"
            )

        if request.require_structured_output:
            if not model.supports_structured_outputs:
                return ModelEligibilityResult(
                    eligible=False,
                    reasons=(
                        "structured output capability required",
                    ),
                )

            reasons.append(
                "structured output supported"
            )

        if request.require_tool_calling:
            if not model.supports_tools:
                return ModelEligibilityResult(
                    eligible=False,
                    reasons=(
                        "tool calling capability required",
                    ),
                )

            reasons.append(
                "tool calling supported"
            )

        if request.minimum_context_length is not None:
            if request.minimum_context_length < 1:
                raise ValueError(
                    "minimum_context_length must be >= 1."
                )

            if model.context_length is None:
                return ModelEligibilityResult(
                    eligible=False,
                    reasons=(
                        "context length metadata unavailable",
                    ),
                )

            if model.context_length < request.minimum_context_length:
                return ModelEligibilityResult(
                    eligible=False,
                    reasons=(
                        (
                            "context length below required minimum "
                            f"of {request.minimum_context_length}"
                        ),
                    ),
                )

            reasons.append(
                (
                    "context length meets minimum "
                    f"of {request.minimum_context_length}"
                )
            )

        if not reasons:
            reasons.append(
                "model satisfies eligibility requirements"
            )

        return ModelEligibilityResult(
            eligible=True,
            reasons=tuple(reasons),
        )

    def filter_models(
        self,
        models: Iterable[OpenRouterModelMetadata],
        request: ModelEligibilityRequest,
    ) -> list[OpenRouterModelMetadata]:
        """
        Return only models that satisfy the supplied requirements.
        """

        eligible_models: list[
            OpenRouterModelMetadata
        ] = []

        for model in models:
            result = self.evaluate(
                model,
                request,
            )

            if result.eligible:
                eligible_models.append(model)

        return eligible_models

    def filter_unique_models(
        self,
        models: Iterable[OpenRouterModelMetadata],
        request: ModelEligibilityRequest,
        *,
        excluded_model_ids: Iterable[str] = (),
    ) -> list[OpenRouterModelMetadata]:
        """
        Filter eligible models while excluding already configured
        model IDs and duplicate discovered IDs.

        Model IDs are compared exactly after stripping whitespace.
        """

        excluded = {
            model_id.strip()
            for model_id in excluded_model_ids
            if model_id.strip()
        }

        result: list[
            OpenRouterModelMetadata
        ] = []

        seen: set[str] = set()

        for model in models:
            model_id = model.model_id.strip()

            if not model_id:
                continue

            if model_id in excluded:
                continue

            if model_id in seen:
                continue

            eligibility = self.evaluate(
                model,
                request,
            )

            if not eligibility.eligible:
                continue

            seen.add(model_id)
            result.append(model)

        return result