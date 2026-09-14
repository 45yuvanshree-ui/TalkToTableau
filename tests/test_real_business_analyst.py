import pytest

from agents import Runner

from agent_components import create_business_analyst
from routing import (
    AgentsModelFactory,
    LLMProviderConfig,
    ModelTier,
)


@pytest.mark.live_llm
def test_real_business_analyst_with_configured_provider():
    """
    Verify that the real Business Analyst agent can use the
    provider configured through the environment.
    """

    provider_config = LLMProviderConfig()

    factory = AgentsModelFactory(
        provider_config=provider_config
    )

    model = factory.create_model_for_tier(
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