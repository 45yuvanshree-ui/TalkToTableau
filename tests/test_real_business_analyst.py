import os

from agents import Runner

from agent_components import create_business_analyst
from routing import (
    AgentsModelFactory,
    LLMProviderConfig,
    ModelTier,
)


def test_real_business_analyst_with_ollama():
    """
    Verify that the real Business Analyst agent can use
    the local Ollama model through the Agents SDK.
    """

    os.environ["LLM_PROVIDER"] = "local"
    os.environ["OLLAMA_FAST_MODEL"] = "llama3.2:3b"

    provider_config = LLMProviderConfig()

    factory = AgentsModelFactory(
        provider_config=provider_config
    )

    model = factory.create_model(
        ModelTier.FAST
    )

    analyst = create_business_analyst(
        model=model
    )

    result = Runner.run_sync(
        analyst,
        """
        The sales dataset contains these columns:

        sale_id, sale_date, product, category, region,
        customer_type, quantity, revenue, cost.

        What was the total revenue by region?

        Return a structured analytical plan.
        """,
    )

    plan = result.final_output

    print("\nGenerated Analytical Plan:")
    print(plan.model_dump_json(indent=2))

    assert plan.analytical_objective
    assert plan.analytical_question