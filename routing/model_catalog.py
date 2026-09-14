from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import json
import os


@dataclass(frozen=True)
class OpenRouterModelMetadata:
    """Normalized metadata for a model discovered from OpenRouter."""

    model_id: str
    name: str
    canonical_slug: str | None = None
    context_length: int | None = None
    input_modalities: tuple[str, ...] = ()
    output_modalities: tuple[str, ...] = ()
    supported_parameters: tuple[str, ...] = ()
    prompt_price: float | None = None
    completion_price: float | None = None
    is_free: bool = False
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def supports_tools(self) -> bool:
        return "tools" in self.supported_parameters

    @property
    def supports_structured_outputs(self) -> bool:
        return "structured_outputs" in self.supported_parameters

    @property
    def supports_reasoning(self) -> bool:
        return (
            "reasoning" in self.supported_parameters
            or "include_reasoning" in self.supported_parameters
        )

    @property
    def is_text_model(self) -> bool:
        return (
            "text" in self.input_modalities
            and "text" in self.output_modalities
        )


class ModelCatalogError(RuntimeError):
    """Base error for model catalog failures."""


class ModelCatalogUnavailableError(ModelCatalogError):
    """Raised when the provider model catalog cannot be reached."""


class ModelCatalogResponseError(ModelCatalogError):
    """Raised when the provider returns an invalid catalog response."""


class OpenRouterModelCatalog:
    """
    Read-only OpenRouter model discovery.

    This class intentionally does not modify the ModelPool yet.
    It provides a safe provider-independent metadata layer that can
    later feed the Omni Router's model pool.
    """

    DEFAULT_TIMEOUT_SECONDS = 10.0

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "").strip()
        self.base_url = (
            base_url or os.getenv(
                "OPENROUTER_BASE_URL",
                "https://openrouter.ai/api/v1",
            )
        ).rstrip("/")
        self.timeout_seconds = timeout_seconds

    def list_models(
        self,
        *,
        output_modalities: str = "text",
        supported_parameters: str | None = None,
        query: str | None = None,
        free_only: bool = False,
    ) -> list[OpenRouterModelMetadata]:
        """
        Fetch and normalize models from OpenRouter.

        Args:
            output_modalities:
                OpenRouter output modality filter. Defaults to text.
            supported_parameters:
                Optional comma-separated OpenRouter capability filter.
            query:
                Optional provider-side free-text model search.
            free_only:
                If True, retain only models whose prompt and completion
                pricing are both zero.
        """
        payload = self._request_models(
            output_modalities=output_modalities,
            supported_parameters=supported_parameters,
            query=query,
        )

        models = self._parse_models(payload)

        if free_only:
            models = [model for model in models if model.is_free]

        return models

    def list_free_text_models(self) -> list[OpenRouterModelMetadata]:
        """Return currently listed free text-to-text models."""
        models = self.list_models(
            output_modalities="text",
            free_only=True,
        )

        return [
            model
            for model in models
            if model.is_text_model
        ]

    def _request_models(
        self,
        *,
        output_modalities: str,
        supported_parameters: str | None,
        query: str | None,
    ) -> dict[str, Any]:
        if not self.api_key:
            raise ModelCatalogUnavailableError(
                "OPENROUTER_API_KEY is not configured."
            )

        params: dict[str, str] = {
            "output_modalities": output_modalities,
        }

        if supported_parameters:
            params["supported_parameters"] = supported_parameters

        if query:
            params["q"] = query

        url = f"{self.base_url}/models?{urlencode(params)}"

        request = Request(
            url,
            method="GET",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
            },
        )

        try:
            with urlopen(
                request,
                timeout=self.timeout_seconds,
            ) as response:
                status = getattr(response, "status", 200)

                if status < 200 or status >= 300:
                    raise ModelCatalogUnavailableError(
                        f"OpenRouter model catalog returned HTTP {status}."
                    )

                raw_body = response.read()

        except HTTPError as exc:
            raise ModelCatalogUnavailableError(
                f"OpenRouter model catalog request failed with HTTP "
                f"{exc.code}."
            ) from exc

        except (URLError, TimeoutError, OSError) as exc:
            raise ModelCatalogUnavailableError(
                "Unable to reach the OpenRouter model catalog."
            ) from exc

        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ModelCatalogResponseError(
                "OpenRouter returned an invalid JSON model catalog."
            ) from exc

        if not isinstance(payload, dict):
            raise ModelCatalogResponseError(
                "OpenRouter model catalog root must be a JSON object."
            )

        if not isinstance(payload.get("data"), list):
            raise ModelCatalogResponseError(
                "OpenRouter model catalog response is missing a "
                "valid 'data' list."
            )

        return payload

    @staticmethod
    def _parse_models(
        payload: dict[str, Any],
    ) -> list[OpenRouterModelMetadata]:
        models: list[OpenRouterModelMetadata] = []

        for item in payload["data"]:
            if not isinstance(item, dict):
                continue

            model_id = item.get("id")

            if not isinstance(model_id, str) or not model_id.strip():
                continue

            architecture = item.get("architecture")
            if not isinstance(architecture, dict):
                architecture = {}

            input_modalities = OpenRouterModelCatalog._string_tuple(
                architecture.get("input_modalities")
            )

            output_modalities = OpenRouterModelCatalog._string_tuple(
                architecture.get("output_modalities")
            )

            supported_parameters = (
                OpenRouterModelCatalog._string_tuple(
                    item.get("supported_parameters")
                )
            )

            pricing = item.get("pricing")
            if not isinstance(pricing, dict):
                pricing = {}

            prompt_price = OpenRouterModelCatalog._parse_price(
                pricing.get("prompt")
            )

            completion_price = OpenRouterModelCatalog._parse_price(
                pricing.get("completion")
            )

            is_free = (
                prompt_price is not None
                and completion_price is not None
                and prompt_price == 0.0
                and completion_price == 0.0
            )

            context_length = item.get("context_length")
            if not isinstance(context_length, int):
                context_length = None

            canonical_slug = item.get("canonical_slug")
            if not isinstance(canonical_slug, str):
                canonical_slug = None

            name = item.get("name")
            if not isinstance(name, str) or not name.strip():
                name = model_id

            models.append(
                OpenRouterModelMetadata(
                    model_id=model_id,
                    name=name,
                    canonical_slug=canonical_slug,
                    context_length=context_length,
                    input_modalities=input_modalities,
                    output_modalities=output_modalities,
                    supported_parameters=supported_parameters,
                    prompt_price=prompt_price,
                    completion_price=completion_price,
                    is_free=is_free,
                    raw=item,
                )
            )

        return models

    @staticmethod
    def _string_tuple(value: Any) -> tuple[str, ...]:
        if not isinstance(value, list):
            return ()

        return tuple(
            item
            for item in value
            if isinstance(item, str) and item.strip()
        )

    @staticmethod
    def _parse_price(value: Any) -> float | None:
        if isinstance(value, (int, float)):
            return float(value)

        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return None

        return None