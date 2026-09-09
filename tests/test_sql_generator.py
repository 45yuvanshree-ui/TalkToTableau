from agents import Agent, Runner

from models import AnalyticalPlan
from schema import DataSchema, TableSchema, ColumnSchema
from sql import generate_sql


class FakeModel:
    pass


def test_sql_generation_output_model():
    """Verify the structured SQL output model."""

    from sql import SQLGenerationOutput

    result = SQLGenerationOutput(
        query="SELECT region, SUM(revenue) AS total_revenue "
        "FROM golden_sales "
        "GROUP BY region"
    )

    assert result.query.startswith("SELECT")