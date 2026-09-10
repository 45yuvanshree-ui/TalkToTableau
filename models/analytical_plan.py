from pydantic import BaseModel, Field


class AnalyticalPlan(BaseModel):
    """Structured analytical intent produced by Agent 1."""

    analytical_objective: str = Field(
        description=(
            "The main analytical objective, such as aggregation, "
            "comparison, ratio, percentage, trend, ranking, "
            "distribution, or anomaly analysis."
        )
    )

    operation: str | None = Field(
        default=None,
        description=(
            "The primary analytical operation requested by the user. "
            "Examples include aggregation, ratio, percentage, "
            "comparison, trend, ranking, distribution, or anomaly."
        ),
    )

    metric_definition: str | None = Field(
        default=None,
        description=(
            "A precise description of the base metric being calculated. "
            "Examples include total revenue, average order value, "
            "customer count, or distinct customer count."
        ),
    )

    metrics: list[str] = Field(
        default_factory=list,
        description=(
            "Measures or derived calculations requested by the user. "
            "Keep this compatible with the existing analytical workflow."
        ),
    )

    dimensions: list[str] = Field(
        default_factory=list,
        description=(
            "Fields explicitly requested as grouping or breakdown "
            "dimensions. A field used only as a condition inside a "
            "metric must not be included here."
        ),
    )

    filters: list[str] = Field(
        default_factory=list,
        description=(
            "Conditions that restrict the rows included in the overall "
            "analysis. Conditions used only to define a metric component "
            "should not be placed here."
        ),
    )

    metric_conditions: list[str] = Field(
        default_factory=list,
        description=(
            "Conditions that define metric components, such as "
            "Currency = EUR for a numerator or Currency = CZK for "
            "a denominator. These conditions are part of the metric "
            "calculation and do not automatically create dimensions "
            "or global filters."
        ),
    )

    numerator: str | None = Field(
        default=None,
        description=(
            "The numerator definition for ratio, percentage, proportion, "
            "share, or similar calculations."
        ),
    )

    denominator: str | None = Field(
        default=None,
        description=(
            "The denominator definition for ratio, percentage, proportion, "
            "share, or similar calculations."
        ),
    )

    time_range: str | None = Field(
        default=None,
        description="Requested time period, if one is specified.",
    )

    comparison: str | None = Field(
        default=None,
        description="Comparison requested by the user, if any.",
    )

    analytical_question: str = Field(
        description="A concise restatement of the analytical question."
    )