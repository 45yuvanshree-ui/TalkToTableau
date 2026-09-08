from pydantic import BaseModel, Field


class VisualizationSpec(BaseModel):
    """Semantic specification for a requested visualization."""

    chart_type: str
    x_column: str | None = None
    y_column: str | None = None
    title: str
    reason: str


class VisualizationResult(BaseModel):
    """Validated visualization workflow result."""

    status: str
    spec: VisualizationSpec | None = None