from agents import Model, OpenAIProvider

from .models import ModelTier
from .provider import LLMProviderConfig


class AgentsModelFactory:
    """
    Build an OpenAI Agents SDK model from TalkToTableau's
    provider configuration.

    This factory does not execute a model request.
    """

    def __init__(
        self,
        provider_config: LLMProviderConfig | None = None,
    ) -> None:
        self.provider_config = (
            provider_config or LLMProviderConfig()
        )

    def create_model(
        self,
        tier: ModelTier,
    ) -> Model:
        """Create an Agents SDK model for the selected tier."""

        model_name = self.provider_config.get_model(tier)
        base_url = self.provider_config.get_base_url()
        api_key = self.provider_config.get_api_key()

        provider = OpenAIProvider(
            api_key=api_key,
            base_url=base_url,
            use_responses=False,
            strict_feature_validation=True,
        )

        return provider.get_model(model_name)