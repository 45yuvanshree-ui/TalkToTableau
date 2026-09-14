from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

from .models import ModelTier


T = TypeVar("T")


# Maximum number of fallback attempts after the primary model.
MAX_FALLBACK_ATTEMPTS = 2


@dataclass(frozen=True)
class ModelAttempt:
    """
    Metadata describing one model execution attempt.
    """

    tier: ModelTier
    model_name: str
    attempt_number: int
    success: bool
    error_type: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class FallbackResult(Generic[T]):
    """
    Result returned by the controlled model execution layer.
    """

    result: T
    attempts: list[ModelAttempt]


class ProviderFailure(Exception):
    """
    Explicit exception type for callers that want to identify
    a transient provider failure directly.

    This exception is always eligible for fallback.
    """


class FallbackExhaustedError(RuntimeError):
    """
    Raised when all configured model attempts fail because of
    provider/transient failures.

    The individual attempts are preserved so higher layers can
    distinguish quota exhaustion from ordinary transient failures.
    """

    def __init__(
        self,
        message: str,
        attempts: list[ModelAttempt],
    ) -> None:
        super().__init__(message)
        self.attempts = attempts


class ProviderQuotaExhaustedError(FallbackExhaustedError):
    """
    Raised when the configured provider reports that its quota
    or allowance has been exhausted.

    This is intentionally separate from ordinary transient
    provider failures because an evaluation should not classify
    provider quota exhaustion as a system/model failure.
    """


class LLMFallbackExecutor:
    """
    Execute an LLM operation with bounded model fallback.

    The executor does not depend on OpenRouter, Ollama, OpenAI,
    or the Agents SDK.

    The caller supplies:
    - an ordered list of model attempts
    - an execution function

    Only transient provider failures trigger fallback.
    Semantic/application errors are propagated unchanged.
    """

    def __init__(
        self,
        max_fallback_attempts: int = MAX_FALLBACK_ATTEMPTS,
    ) -> None:
        if max_fallback_attempts < 0:
            raise ValueError(
                "max_fallback_attempts must be >= 0."
            )

        self.max_fallback_attempts = (
            max_fallback_attempts
        )

    def run(
        self,
        *,
        attempts: list[tuple[ModelTier, str]],
        execute: Callable[[ModelTier, str], T],
    ) -> FallbackResult[T]:
        """
        Execute the supplied operation against configured models.

        The total number of executions is bounded by:

            1 + max_fallback_attempts
        """

        if not attempts:
            raise ValueError(
                "At least one model attempt must be configured."
            )

        maximum_attempts = min(
            len(attempts),
            self.max_fallback_attempts + 1,
        )

        execution_attempts = attempts[
            :maximum_attempts
        ]

        attempt_records: list[ModelAttempt] = []

        for index, (
            tier,
            model_name,
        ) in enumerate(
            execution_attempts,
            start=1,
        ):
            try:
                result = execute(
                    tier,
                    model_name,
                )

                attempt_records.append(
                    ModelAttempt(
                        tier=tier,
                        model_name=model_name,
                        attempt_number=index,
                        success=True,
                    )
                )

                return FallbackResult(
                    result=result,
                    attempts=attempt_records,
                )

            except Exception as error:
                if not self.is_provider_failure(
                    error
                ):
                    raise

                attempt_records.append(
                    ModelAttempt(
                        tier=tier,
                        model_name=model_name,
                        attempt_number=index,
                        success=False,
                        error_type=type(error).__name__,
                        error_message=str(error),
                    )
                )

                if index >= maximum_attempts:
                    break

        if self._attempts_indicate_quota_exhaustion(
            attempt_records
        ):
            raise ProviderQuotaExhaustedError(
                (
                    "Configured LLM provider quota or daily "
                    "allowance is exhausted."
                ),
                attempts=attempt_records,
            )

        raise FallbackExhaustedError(
            (
                "All configured LLM model attempts failed "
                "due to provider/transient failures."
            ),
            attempts=attempt_records,
        )

    @staticmethod
    def is_provider_failure(
        error: Exception,
    ) -> bool:
        """
        Determine whether an exception represents a transient
        provider/model failure that should trigger fallback.

        Classification priority:

        1. Explicit ProviderFailure
        2. Recognized HTTP status codes
        3. Clearly transient exception class names
        4. Explicit transient transport/provider messages

        Important:
        Generic words such as "provider", "model", "error", or
        "failure" are NOT sufficient by themselves.

        Semantic/application failures such as:
        - invalid JSON
        - structured-output validation
        - Pydantic validation
        - SQL validation
        - application ValueError

        are deliberately not classified as provider failures.
        """

        if isinstance(
            error,
            ProviderFailure,
        ):
            return True

        status_code = (
            LLMFallbackExecutor._extract_status_code(
                error
            )
        )

        if status_code in {
            408,
            425,
            429,
            500,
            502,
            503,
            504,
        }:
            return True

        error_name = (
            type(error).__name__.lower()
        )

        transient_exception_names = (
            "timeout",
            "timeouterror",
            "connecterror",
            "connectionerror",
            "connectionreseterror",
            "connectionabortederror",
            "brokenpipeerror",
        )

        if any(
            token in error_name
            for token in transient_exception_names
        ):
            return True

        message = str(error).strip().lower()

        if not message:
            return False

        transient_message_patterns = (
            "timed out",
            "timeout while",
            "connection reset",
            "connection refused",
            "connection aborted",
            "failed to connect",
            "temporary failure in name resolution",
            "temporary failure",
            "temporarily unavailable",
            "service unavailable",
            "bad gateway",
            "gateway timeout",
            "upstream timeout",
            "upstream connection",
            "upstream unavailable",
            "upstream error",
            "upstream failure",
            "rate limit exceeded",
            "rate limited",
            "too many requests",
            "http 429",
            "http 500",
            "http 502",
            "http 503",
            "http 504",
            "status code 429",
            "status code 500",
            "status code 502",
            "status code 503",
            "status code 504",
        )

        return any(
            pattern in message
            for pattern in transient_message_patterns
        )

    @staticmethod
    def is_quota_exhaustion(
        error: Exception,
    ) -> bool:
        """
        Determine whether an exception explicitly indicates
        provider quota exhaustion.

        This intentionally looks for specific quota signals
        instead of treating every HTTP 429 as permanent quota
        exhaustion.

        A generic 429 can still be a temporary rate limit and
        therefore remains an ordinary provider failure.
        """

        message = str(error).strip().lower()

        if not message:
            return False

        quota_patterns = (
            "free-models-per-day",
            "free models per day",
            "free-model daily limit",
            "free model daily limit",
            "daily free-model limit",
            "daily free model limit",
            "daily quota exhausted",
            "daily quota exceeded",
            "quota exhausted",
            "quota exceeded",
            "quota has been exhausted",
            "allowance has been exhausted",
            "limit_source': 'openrouter_free_tier_daily",
            "limit_source\": \"openrouter_free_tier_daily",
            "openrouter_free_tier_daily",
        )

        if any(
            pattern in message
            for pattern in quota_patterns
        ):
            return True

        return False

    @staticmethod
    def _attempts_indicate_quota_exhaustion(
        attempts: list[ModelAttempt],
    ) -> bool:
        """
        Return True when at least one failed attempt contains
        an explicit provider quota exhaustion signal.
        """

        for attempt in attempts:
            if attempt.success:
                continue

            error_message = (
                attempt.error_message
            )

            if not error_message:
                continue

            if LLMFallbackExecutor.is_quota_exhaustion(
                RuntimeError(error_message)
            ):
                return True

        return False

    @staticmethod
    def _extract_status_code(
        error: Exception,
    ) -> int | None:
        """
        Extract a conventional HTTP status code from common
        exception shapes without depending on a specific SDK.
        """

        for attribute_name in (
            "status_code",
            "status",
            "http_status",
        ):
            value = getattr(
                error,
                attribute_name,
                None,
            )

            if isinstance(value, int):
                return value

        response = getattr(
            error,
            "response",
            None,
        )

        if response is not None:
            value = getattr(
                response,
                "status_code",
                None,
            )

            if isinstance(value, int):
                return value

        return None


def build_fallback_attempts(
    *,
    primary_tier: ModelTier,
    primary_model: str,
    fallback_tiers: list[
        tuple[ModelTier, str]
    ]
    | None = None,
) -> list[tuple[ModelTier, str]]:
    """
    Build an ordered, duplicate-free model attempt list.

    The primary model is always first.
    """

    candidates = [
        (
            primary_tier,
            primary_model,
        )
    ]

    if fallback_tiers:
        candidates.extend(
            fallback_tiers
        )

    unique_attempts: list[
        tuple[ModelTier, str]
    ] = []

    seen: set[
        tuple[ModelTier, str]
    ] = set()

    for tier, model_name in candidates:
        normalized_model = model_name.strip()

        if not normalized_model:
            continue

        key = (
            tier,
            normalized_model,
        )

        if key in seen:
            continue

        seen.add(key)

        unique_attempts.append(
            (
                tier,
                normalized_model,
            )
        )

    if not unique_attempts:
        raise ValueError(
            "No valid LLM model attempts were configured."
        )

    return unique_attempts