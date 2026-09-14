from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterable

from .models import ModelTier


@dataclass(frozen=True)
class ModelCapabilities:
    """
    Capabilities supported by a concrete LLM model.

    The pool uses these capabilities to avoid selecting a model
    that is unsuitable for the current task.
    """

    text_generation: bool = True
    structured_output: bool = False
    tool_calling: bool = False
    reasoning: bool = False
    sql_generation: bool = True
    long_context: bool = False


@dataclass
class ModelHealth:
    """
    Runtime health information for a model.

    Health is intentionally kept in application memory for now.
    Persistence can be added later if required.
    """

    success_count: int = 0
    failure_count: int = 0
    provider_failure_count: int = 0
    semantic_failure_count: int = 0
    total_latency_ms: float = 0.0
    last_success_at: datetime | None = None
    last_failure_at: datetime | None = None
    unavailable_until: datetime | None = None

    @property
    def total_attempts(self) -> int:
        return self.success_count + self.failure_count

    @property
    def success_rate(self) -> float:
        """
        Return historical success rate.

        A model without history receives a neutral 1.0 value so that
        lack of history does not unfairly penalize a new model.
        """

        if self.total_attempts == 0:
            return 1.0

        return self.success_count / self.total_attempts

    @property
    def average_latency_ms(self) -> float | None:
        if self.success_count == 0:
            return None

        return self.total_latency_ms / self.success_count

    @property
    def is_temporarily_unavailable(self) -> bool:
        if self.unavailable_until is None:
            return False

        return datetime.now(timezone.utc) < self.unavailable_until


@dataclass
class ModelCandidate:
    """
    A concrete model that can be considered by the Omni Router.
    """

    name: str
    provider: str
    tier: ModelTier

    is_free: bool = False

    capabilities: ModelCapabilities = field(
        default_factory=ModelCapabilities
    )

    priority: int = 0

    health: ModelHealth = field(
        default_factory=ModelHealth
    )

    metadata: dict[str, object] = field(
        default_factory=dict
    )

    def is_available(self) -> bool:
        """
        Return whether this candidate is currently eligible for use.
        """

        return not self.health.is_temporarily_unavailable


@dataclass(frozen=True)
class ModelSelectionRequest:
    """
    Requirements used to rank model candidates.
    """

    tier: ModelTier

    reasoning_required: bool = False

    structured_output_required: bool = False

    tool_calling_required: bool = False

    sql_generation_required: bool = True

    prefer_free: bool = True


@dataclass(frozen=True)
class RankedModel:
    """
    A model candidate together with its calculated routing score.
    """

    candidate: ModelCandidate
    score: float
    reasons: tuple[str, ...]


class ModelPool:
    """
    Registry and health-aware ranking system for concrete LLM models.

    The ModelPool does not execute models.

    Its responsibilities are:

    - register models
    - remove models
    - retrieve candidates
    - filter incompatible models
    - track model health
    - temporarily suppress unhealthy models
    - rank candidates
    - return the best suitable model

    Actual model execution remains in the Agents SDK execution layer.
    """

    DEFAULT_FAILURE_COOLDOWN_SECONDS = 60

    def __init__(
        self,
        candidates: Iterable[ModelCandidate] | None = None,
        *,
        failure_cooldown_seconds: int = (
            DEFAULT_FAILURE_COOLDOWN_SECONDS
        ),
    ) -> None:
        if failure_cooldown_seconds < 0:
            raise ValueError(
                "failure_cooldown_seconds must be >= 0."
            )

        self.failure_cooldown_seconds = (
            failure_cooldown_seconds
        )

        self._candidates: dict[
            tuple[str, str],
            ModelCandidate,
        ] = {}

        if candidates:
            for candidate in candidates:
                self.register(candidate)

    def register(
        self,
        candidate: ModelCandidate,
    ) -> None:
        """
        Register or replace a model candidate.

        Provider + model name form the unique identity.
        """

        name = candidate.name.strip()
        provider = candidate.provider.strip().lower()

        if not name:
            raise ValueError(
                "Model candidate name must not be empty."
            )

        if not provider:
            raise ValueError(
                "Model candidate provider must not be empty."
            )

        candidate.name = name
        candidate.provider = provider

        key = self._key(
            provider,
            name,
        )

        self._candidates[key] = candidate

    def unregister(
        self,
        *,
        provider: str,
        model_name: str,
    ) -> bool:
        """
        Remove a registered model.

        Returns True when a model was removed.
        """

        key = self._key(
            provider,
            model_name,
        )

        return self._candidates.pop(
            key,
            None,
        ) is not None

    def get_candidate(
        self,
        *,
        provider: str,
        model_name: str,
    ) -> ModelCandidate | None:
        """
        Retrieve a concrete registered model.
        """

        return self._candidates.get(
            self._key(
                provider,
                model_name,
            )
        )

    def list_candidates(
        self,
    ) -> list[ModelCandidate]:
        """
        Return all registered candidates.
        """

        return list(
            self._candidates.values()
        )

    def available_candidates(
        self,
        request: ModelSelectionRequest,
    ) -> list[ModelCandidate]:
        """
        Return candidates that are compatible with the request
        and are not temporarily unavailable.
        """

        candidates: list[ModelCandidate] = []

        for candidate in self._candidates.values():
            if not candidate.is_available():
                continue

            if not self._supports_request(
                candidate,
                request,
            ):
                continue

            candidates.append(candidate)

        return candidates

    def rank(
        self,
        request: ModelSelectionRequest,
    ) -> list[RankedModel]:
        """
        Rank compatible models from best to worst.

        Ranking considers:

        - requested logical tier
        - capability compatibility
        - free-model preference
        - historical reliability
        - latency
        - explicit priority

        This is intentionally not based on question length alone.
        """

        ranked: list[RankedModel] = []

        for candidate in self.available_candidates(
            request
        ):
            score, reasons = self._calculate_score(
                candidate,
                request,
            )

            ranked.append(
                RankedModel(
                    candidate=candidate,
                    score=score,
                    reasons=tuple(reasons),
                )
            )

        ranked.sort(
            key=lambda item: (
                item.score,
                item.candidate.priority,
            ),
            reverse=True,
        )

        return ranked

    def select(
        self,
        request: ModelSelectionRequest,
    ) -> RankedModel:
        """
        Select the highest-ranked compatible model.

        Raises RuntimeError when no suitable model is available.
        """

        ranked = self.rank(request)

        if not ranked:
            raise RuntimeError(
                (
                    "No available LLM model matches the "
                    "requested capabilities."
                )
            )

        return ranked[0]

    def record_success(
        self,
        *,
        provider: str,
        model_name: str,
        latency_ms: float | None = None,
    ) -> None:
        """
        Record a successful model execution.
        """

        candidate = self.get_candidate(
            provider=provider,
            model_name=model_name,
        )

        if candidate is None:
            return

        health = candidate.health

        health.success_count += 1
        health.last_success_at = (
            datetime.now(timezone.utc)
        )

        if latency_ms is not None:
            if latency_ms >= 0:
                health.total_latency_ms += latency_ms

        # A successful request clears temporary unavailability.
        health.unavailable_until = None

    def record_provider_failure(
        self,
        *,
        provider: str,
        model_name: str,
        cooldown_seconds: int | None = None,
    ) -> None:
        """
        Record a provider/model availability failure.

        Provider failures temporarily suppress the model so that
        another compatible candidate can be selected.
        """

        candidate = self.get_candidate(
            provider=provider,
            model_name=model_name,
        )

        if candidate is None:
            return

        health = candidate.health

        health.failure_count += 1
        health.provider_failure_count += 1
        health.last_failure_at = (
            datetime.now(timezone.utc)
        )

        cooldown = (
            self.failure_cooldown_seconds
            if cooldown_seconds is None
            else cooldown_seconds
        )

        if cooldown < 0:
            raise ValueError(
                "cooldown_seconds must be >= 0."
            )

        health.unavailable_until = (
            datetime.now(timezone.utc)
            + timedelta(seconds=cooldown)
        )

    def record_semantic_failure(
        self,
        *,
        provider: str,
        model_name: str,
    ) -> None:
        """
        Record an application/semantic failure.

        Semantic failures do not automatically make the model
        unavailable because the provider itself may still be healthy.

        They do, however, reduce the model's reliability score.
        """

        candidate = self.get_candidate(
            provider=provider,
            model_name=model_name,
        )

        if candidate is None:
            return

        health = candidate.health

        health.failure_count += 1
        health.semantic_failure_count += 1
        health.last_failure_at = (
            datetime.now(timezone.utc)
        )

    def clear_health(
        self,
        *,
        provider: str,
        model_name: str,
    ) -> None:
        """
        Reset runtime health information for a model.
        """

        candidate = self.get_candidate(
            provider=provider,
            model_name=model_name,
        )

        if candidate is None:
            return

        candidate.health = ModelHealth()

    def _supports_request(
        self,
        candidate: ModelCandidate,
        request: ModelSelectionRequest,
    ) -> bool:
        """
        Determine whether a candidate satisfies mandatory
        capability requirements.
        """

        capabilities = candidate.capabilities

        if request.reasoning_required:
            if not capabilities.reasoning:
                return False

        if request.structured_output_required:
            if not capabilities.structured_output:
                return False

        if request.tool_calling_required:
            if not capabilities.tool_calling:
                return False

        if request.sql_generation_required:
            if not capabilities.sql_generation:
                return False

        if not capabilities.text_generation:
            return False

        return True

    def _calculate_score(
        self,
        candidate: ModelCandidate,
        request: ModelSelectionRequest,
    ) -> tuple[float, list[str]]:
        """
        Calculate a routing score.

        The score is deliberately explainable so the application
        can expose routing decisions through tracing/UI.
        """

        capabilities = candidate.capabilities
        health = candidate.health

        score = 0.0
        reasons: list[str] = []

        # ---------------------------------------------------------
        # Tier compatibility
        # ---------------------------------------------------------

        if candidate.tier == request.tier:
            score += 30.0
            reasons.append(
                "logical tier matches request"
            )
        else:
            score += self._tier_distance_score(
                candidate.tier,
                request.tier,
            )

        # ---------------------------------------------------------
        # Required capabilities
        # ---------------------------------------------------------

        if capabilities.text_generation:
            score += 5.0

        if request.sql_generation_required:
            score += 10.0
            reasons.append(
                "SQL generation capability available"
            )

        if request.reasoning_required:
            score += 15.0
            reasons.append(
                "reasoning capability available"
            )

        if request.structured_output_required:
            score += 15.0
            reasons.append(
                "structured-output capability available"
            )

        if request.tool_calling_required:
            score += 15.0
            reasons.append(
                "tool-calling capability available"
            )

        # ---------------------------------------------------------
        # Free-model preference
        # ---------------------------------------------------------

        if request.prefer_free:
            if candidate.is_free:
                score += 15.0
                reasons.append(
                    "free model preferred"
                )
            else:
                score -= 5.0
        elif not candidate.is_free:
            score += 2.0

        # ---------------------------------------------------------
        # Reliability
        # ---------------------------------------------------------

        success_rate = health.success_rate

        reliability_score = (
            success_rate * 20.0
        )

        score += reliability_score

        if health.total_attempts > 0:
            reasons.append(
                (
                    "historical success rate "
                    f"{success_rate:.2f}"
                )
            )
        else:
            reasons.append(
                "no historical failures recorded"
            )

        # ---------------------------------------------------------
        # Latency
        # ---------------------------------------------------------

        average_latency = (
            health.average_latency_ms
        )

        if average_latency is not None:
            latency_penalty = min(
                average_latency / 1000.0,
                10.0,
            )

            score -= latency_penalty

            reasons.append(
                (
                    "latency penalty "
                    f"{latency_penalty:.2f}"
                )
            )

        # ---------------------------------------------------------
        # Explicit priority
        # ---------------------------------------------------------

        priority_bonus = min(
            max(candidate.priority, 0),
            10,
        )

        score += priority_bonus

        if priority_bonus:
            reasons.append(
                f"priority bonus {priority_bonus}"
            )

        return score, reasons

    @staticmethod
    def _tier_distance_score(
        candidate_tier: ModelTier,
        requested_tier: ModelTier,
    ) -> float:
        """
        Provide a small compatibility score for candidates whose
        logical tier differs from the requested tier.

        This allows the Omni Router to use another capable model
        instead of failing simply because the exact tier is
        unavailable.
        """

        if candidate_tier == requested_tier:
            return 30.0

        tier_order = {
            ModelTier.FAST: 1,
            ModelTier.BALANCED: 2,
            ModelTier.REASONING: 3,
        }

        distance = abs(
            tier_order[candidate_tier]
            - tier_order[requested_tier]
        )

        if distance == 1:
            return 18.0

        return 8.0

    @staticmethod
    def _key(
        provider: str,
        model_name: str,
    ) -> tuple[str, str]:
        return (
            provider.strip().lower(),
            model_name.strip(),
        )