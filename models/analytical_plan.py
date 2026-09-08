from pydantic import BaseModel, Field


class AnalyticalPlan(BaseModel):
    """Structured analytical plan produced by Agent 1."""

    analytical_objective: str = Field(
        description=(
            "The main analytical objective, such as "
            "aggregation, comparison, trend, ranking, "
            "distribution, or anomaly analysis."
        )
    )

    metrics: list[str] = Field(
        default_factory=list,
        description="Measures the user wants to analyze."
    )

    dimensions: list[str] = Field(
        default_factory=list,
        description="Categorical or grouping dimensions."
    )

    filters: list[str] = Field(
        default_factory=list,
        description="Conditions that restrict the analysis."
    )

    time_range: str | None = Field(
        default=None,
        description="Requested time period, if one is specified."
    )

    comparison: str | None = Field(
        default=None,
        description="Comparison requested by the user, if any."
    )

    analytical_question: str = Field(
        description="A concise restatement of the analytical question."
    )