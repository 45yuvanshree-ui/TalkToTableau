from agents import Agent, Model

from models import AnalyticalPlan


BUSINESS_ANALYST_INSTRUCTIONS = """
You are the Business Analyst agent for TalkToTableau.

Your responsibility is to convert a user's natural-language
business question into a structured analytical plan.

You must identify, when applicable:

- analytical objective
- metrics
- dimensions
- filters
- time range
- comparison requirements
- the precise analytical question

You must use only information supported by the user's request
and the provided schema/context.

You must not:

- generate SQL
- execute SQL
- invent data
- invent columns or tables
- invent filters that the user did not request
- produce business insights
- create a visualization
- create a business story
- make recommendations

If the user's request is ambiguous, preserve the ambiguity
rather than inventing an assumption.

The output must be a structured AnalyticalPlan.
"""


def create_business_analyst(
    model: Model | None = None,
) -> Agent:
    """
    Create the Business Analyst agent with an optional model.

    When a model is supplied by the adaptive routing layer,
    the agent uses that model. Otherwise, the Agents SDK default
    model configuration is used.
    """

    return Agent(
        name="Business Analyst",
        instructions=BUSINESS_ANALYST_INSTRUCTIONS,
        output_type=AnalyticalPlan,
        model=model,
    )


business_analyst = create_business_analyst()