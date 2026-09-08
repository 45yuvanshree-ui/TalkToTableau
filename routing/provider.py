import os

from dotenv import load_dotenv

from .models import ModelTier


load_dotenv()


class LLMProviderConfig:
    """
    Resolve LLM provider and model configuration from environment
    variables.

    This class does not make an LLM request.
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
        """Return the configured model for a routing tier."""

        if self.provider == "local":
            return self._get_local_model(tier)

        if self.provider == "openai_compatible":
            return self._get_openai_compatible_model(tier)

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

        raise ValueError(
            f"Unsupported LLM provider: {self.provider}"
        )

    @staticmethod
    def _get_local_model(
        tier: ModelTier,
    ) -> str:
        """Resolve a model for the local Ollama provider."""

        model_map = {
            ModelTier.FAST: (
                "OLLAMA_FAST_MODEL",
                os.getenv("OLLAMA_FAST_MODEL", ""),
            ),
            ModelTier.BALANCED: (
                "OLLAMA_BALANCED_MODEL",
                os.getenv("OLLAMA_BALANCED_MODEL", ""),
            ),
            ModelTier.REASONING: (
                "OLLAMA_REASONING_MODEL",
                os.getenv("OLLAMA_REASONING_MODEL", ""),
            ),
        }

        environment_name, model = model_map[tier]

        model = model.strip()

        if not model:
            raise ValueError(
                f"{environment_name} is not configured."
            )

        return model

    @staticmethod
    def _get_openai_compatible_model(
        tier: ModelTier,
    ) -> str:
        """Resolve a model for an OpenAI-compatible provider."""

        model_map = {
            ModelTier.FAST: (
                "OPENAI_FAST_MODEL",
                os.getenv("OPENAI_FAST_MODEL", ""),
            ),
            ModelTier.BALANCED: (
                "OPENAI_BALANCED_MODEL",
                os.getenv("OPENAI_BALANCED_MODEL", ""),
            ),
            ModelTier.REASONING: (
                "OPENAI_REASONING_MODEL",
                os.getenv("OPENAI_REASONING_MODEL", ""),
            ),
        }

        environment_name, model = model_map[tier]

        model = model.strip()

        if not model:
            raise ValueError(
                f"{environment_name} is not configured."
            )

        return model