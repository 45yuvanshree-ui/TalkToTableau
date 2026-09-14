import os

from dotenv import load_dotenv

from .models import ModelTier


load_dotenv()


class LLMProviderConfig:
    """
    Resolve LLM provider and model configuration from environment
    variables.

    This class does not make an LLM request.

    Supported providers:
    - local
        Uses Ollama's OpenAI-compatible API.
    - openai_compatible
        Uses a generic OpenAI-compatible API.
    - openrouter
        Uses OpenRouter's OpenAI-compatible API.

    The configuration supports both:

    1. A single model per tier.
    2. An ordered pool of models per tier.

    Existing singular environment variables remain fully
    backward-compatible.

    Example:

        OPENROUTER_FAST_MODEL=model-a

    or:

        OPENROUTER_FAST_MODELS=model-a,model-b,model-c

    If the plural variable is configured, it is used as the
    model pool.

    If the plural variable is not configured, the existing
    singular variable is used as a one-model pool.
    """

    def __init__(self) -> None:
        self.provider = os.getenv(
            "LLM_PROVIDER",
            "local",
        ).strip().lower()

    def get_model(
        self,
        tier: ModelTier,
    ) -> str:
        """
        Return the primary configured model for a routing tier.

        This method is intentionally preserved for backward
        compatibility.

        It returns the first model from the configured model pool.
        """

        models = self.get_models(tier)

        if not models:
            raise ValueError(
                f"No model is configured for tier: {tier.value}"
            )

        return models[0]

    def get_models(
        self,
        tier: ModelTier,
    ) -> list[str]:
        """
        Return the ordered model pool for a routing tier.

        The first model is the primary model.

        Models are deduplicated while preserving configuration
        order.
        """

        if self.provider == "local":
            return self._get_local_models(tier)

        if self.provider == "openai_compatible":
            return self._get_openai_compatible_models(tier)

        if self.provider == "openrouter":
            return self._get_openrouter_models(tier)

        raise ValueError(
            f"Unsupported LLM provider: {self.provider}"
        )

    def get_base_url(self) -> str:
        """Return the configured provider base URL."""

        if self.provider == "local":
            return os.getenv(
                "OLLAMA_BASE_URL",
                "http://localhost:11434/v1",
            )

        if self.provider == "openai_compatible":
            return os.getenv(
                "OPENAI_BASE_URL",
                "https://api.openai.com/v1",
            )

        if self.provider == "openrouter":
            return os.getenv(
                "OPENROUTER_BASE_URL",
                "https://openrouter.ai/api/v1",
            )

        raise ValueError(
            f"Unsupported LLM provider: {self.provider}"
        )

    def get_api_key(self) -> str | None:
        """
        Return the configured API key.

        Local Ollama uses a placeholder value because its local
        OpenAI-compatible endpoint does not require authentication.
        """

        if self.provider == "local":
            return os.getenv(
                "OLLAMA_API_KEY",
                "ollama",
            )

        if self.provider == "openai_compatible":
            api_key = os.getenv("OPENAI_API_KEY")

            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY is not configured."
                )

            return api_key

        if self.provider == "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY")

            if not api_key:
                raise ValueError(
                    "OPENROUTER_API_KEY is not configured."
                )

            return api_key

        raise ValueError(
            f"Unsupported LLM provider: {self.provider}"
        )

    @staticmethod
    def _get_local_model(
        tier: ModelTier,
    ) -> str:
        """
        Resolve a single model for the local Ollama provider.

        This method is preserved for backward compatibility.

        Explicit tier-specific models take priority.

        If BALANCED or REASONING is not configured, the configured
        FAST model is used as a fallback.
        """

        models = LLMProviderConfig._get_local_models(tier)

        return models[0]

    @staticmethod
    def _get_local_models(
        tier: ModelTier,
    ) -> list[str]:
        """
        Resolve an ordered model pool for local Ollama.

        Plural environment variables are preferred.

        If the plural variable is absent, the existing singular
        environment variable is used.

        If BALANCED or REASONING has no model configured, the
        FAST model is used as the local fallback.
        """

        fast_models = LLMProviderConfig._read_model_list(
            plural_name="OLLAMA_FAST_MODELS",
            singular_name="OLLAMA_FAST_MODEL",
        )

        if not fast_models:
            raise ValueError(
                "OLLAMA_FAST_MODEL is not configured."
            )

        model_environment_names = {
            ModelTier.FAST: (
                "OLLAMA_FAST_MODELS",
                "OLLAMA_FAST_MODEL",
            ),
            ModelTier.BALANCED: (
                "OLLAMA_BALANCED_MODELS",
                "OLLAMA_BALANCED_MODEL",
            ),
            ModelTier.REASONING: (
                "OLLAMA_REASONING_MODELS",
                "OLLAMA_REASONING_MODEL",
            ),
        }

        plural_name, singular_name = model_environment_names[tier]

        configured_models = LLMProviderConfig._read_model_list(
            plural_name=plural_name,
            singular_name=singular_name,
        )

        if configured_models:
            return configured_models

        return list(fast_models)

    @staticmethod
    def _get_openai_compatible_model(
        tier: ModelTier,
    ) -> str:
        """
        Resolve a single model for a generic OpenAI-compatible
        provider.

        This method is preserved for backward compatibility.
        """

        models = LLMProviderConfig._get_openai_compatible_models(
            tier
        )

        return models[0]

    @staticmethod
    def _get_openai_compatible_models(
        tier: ModelTier,
    ) -> list[str]:
        """Resolve an ordered model pool for a generic provider."""

        environment_names = {
            ModelTier.FAST: (
                "OPENAI_FAST_MODELS",
                "OPENAI_FAST_MODEL",
            ),
            ModelTier.BALANCED: (
                "OPENAI_BALANCED_MODELS",
                "OPENAI_BALANCED_MODEL",
            ),
            ModelTier.REASONING: (
                "OPENAI_REASONING_MODELS",
                "OPENAI_REASONING_MODEL",
            ),
        }

        plural_name, singular_name = environment_names[tier]

        models = LLMProviderConfig._read_model_list(
            plural_name=plural_name,
            singular_name=singular_name,
        )

        if not models:
            raise ValueError(
                f"{singular_name} is not configured."
            )

        return models

    @staticmethod
    def _get_openrouter_model(
        tier: ModelTier,
    ) -> str:
        """
        Resolve a single model for OpenRouter.

        This method is preserved for backward compatibility.
        """

        models = LLMProviderConfig._get_openrouter_models(
            tier
        )

        return models[0]

    @staticmethod
    def _get_openrouter_models(
        tier: ModelTier,
    ) -> list[str]:
        """Resolve an ordered model pool for OpenRouter."""

        environment_names = {
            ModelTier.FAST: (
                "OPENROUTER_FAST_MODELS",
                "OPENROUTER_FAST_MODEL",
            ),
            ModelTier.BALANCED: (
                "OPENROUTER_BALANCED_MODELS",
                "OPENROUTER_BALANCED_MODEL",
            ),
            ModelTier.REASONING: (
                "OPENROUTER_REASONING_MODELS",
                "OPENROUTER_REASONING_MODEL",
            ),
        }

        plural_name, singular_name = environment_names[tier]

        models = LLMProviderConfig._read_model_list(
            plural_name=plural_name,
            singular_name=singular_name,
        )

        if not models:
            raise ValueError(
                f"{singular_name} is not configured."
            )

        return models

    @staticmethod
    def _read_model_list(
        *,
        plural_name: str,
        singular_name: str,
    ) -> list[str]:
        """
        Read an ordered model list.

        Priority:

        1. PLURAL variable, for example:
           OPENROUTER_FAST_MODELS=a,b,c

        2. Existing SINGULAR variable, for example:
           OPENROUTER_FAST_MODEL=a

        Empty values are ignored.

        Duplicate model names are removed while preserving order.
        """

        plural_value = os.getenv(
            plural_name,
            "",
        ).strip()

        if plural_value:
            raw_models = plural_value.split(",")
        else:
            singular_value = os.getenv(
                singular_name,
                "",
            ).strip()

            if not singular_value:
                return []

            raw_models = [singular_value]

        models: list[str] = []
        seen: set[str] = set()

        for raw_model in raw_models:
            model = raw_model.strip()

            if not model:
                continue

            if model in seen:
                continue

            seen.add(model)
            models.append(model)

        return models