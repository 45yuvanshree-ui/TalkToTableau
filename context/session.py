from datetime import datetime, timezone

from pydantic import BaseModel, Field

from analysis import AnalysisResult
from models import (
    AnalyticalPlan,
    BusinessStory,
    InsightAnalysis,
)
from schema import DataSchema
from sql import SQLResult
from visualization import VisualizationResult


class SessionContext(BaseModel):
    """
    Shared analytical context for one TalkToTableau session.

    The context stores the important artifacts produced during
    the analytical workflow without creating a separate memory agent
    or persistent memory service.
    """

    session_id: str

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    user_request: str | None = None

    data_source: str | None = None

    data_schema: DataSchema | None = None

    analytical_plan: AnalyticalPlan | None = None

    generated_sql: str | None = None

    sql_result: SQLResult | None = None

    analysis_result: AnalysisResult | None = None

    insights: InsightAnalysis | None = None

    visualization: VisualizationResult | None = None

    story: BusinessStory | None = None