from agent_components import business_analyst
from models import AnalyticalPlan


def test_business_analyst_configuration():
    assert business_analyst.name == "Business Analyst"
    assert business_analyst.output_type is AnalyticalPlan


def test_analytical_plan_structure():
    plan = AnalyticalPlan(
        analytical_objective="comparison",
        metrics=["revenue"],
        dimensions=["region"],
        filters=["year = 2025"],
        time_range="2025",
        comparison="North vs South",
        analytical_question=(
            "Compare revenue between North and South regions in 2025."
        ),
    )

    assert plan.analytical_objective == "comparison"
    assert plan.metrics == ["revenue"]
    assert plan.dimensions == ["region"]
    assert plan.time_range == "2025"