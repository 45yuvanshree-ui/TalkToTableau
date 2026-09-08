from pydantic import BaseModel, Field


class AnalysisSummary(BaseModel):
    """Basic deterministic summary of SQL results."""

    row_count: int
    columns: list[str] = Field(default_factory=list)


class ComparisonAnalysis(BaseModel):
    """Structured comparison between groups."""

    dimension: str
    metric: str
    values: dict[str, float] = Field(default_factory=dict)


class TrendAnalysis(BaseModel):
    """Structured analysis of a metric across ordered periods."""

    time_dimension: str
    metric: str
    values: list[dict[str, object]] = Field(default_factory=list)


class AnomalyAnalysis(BaseModel):
    """Structured anomaly detection result."""

    metric: str
    mean: float | None = None
    standard_deviation: float | None = None
    anomalous_rows: list[dict[str, object]] = Field(
        default_factory=list
    )


class AnalysisResult(BaseModel):
    """Complete deterministic analysis output."""

    summary: AnalysisSummary
    comparison: ComparisonAnalysis | None = None
    trend: TrendAnalysis | None = None
    anomaly: AnomalyAnalysis | None = None