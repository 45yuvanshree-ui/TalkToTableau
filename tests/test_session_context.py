from context import SessionContext
from models import (
    AnalyticalPlan,
    BusinessStory,
    Evidence,
    Insight,
    InsightAnalysis,
    StorySection,
)
from schema import (
    ColumnSchema,
    DataSchema,
    TableSchema,
)
from sql import SQLResult
from visualization import VisualizationResult, VisualizationSpec
from analysis import (
    AnalysisResult,
    AnalysisSummary,
)


def make_schema() -> DataSchema:
    return DataSchema(
        source_type="file",
        tables=[
            TableSchema(
                name="golden_sales",
                source_type="file",
                columns=[
                    ColumnSchema(
                        name="region",
                        data_type="object",
                    ),
                    ColumnSchema(
                        name="revenue",
                        data_type="int64",
                    ),
                ],
            )
        ],
    )


def test_session_context_can_store_workflow_state():
    plan = AnalyticalPlan(
        analytical_objective="comparison",
        metrics=["revenue"],
        dimensions=["region"],
        filters=[],
        time_range=None,
        comparison="North vs South",
        analytical_question=(
            "Compare revenue between North and South."
        ),
    )

    sql_result = SQLResult(
        columns=["region", "revenue"],
        rows=[
            ["North", 2200],
            ["South", 1800],
        ],
        row_count=2,
    )

    analysis_result = AnalysisResult(
        summary=AnalysisSummary(
            row_count=2,
            columns=["region", "revenue"],
        )
    )

    insights = InsightAnalysis(
        insights=[
            Insight(
                statement="North generated higher revenue than South.",
                evidence=[
                    Evidence(
                        claim=(
                            "North generated higher revenue than South."
                        ),
                        supporting_data=(
                            "North revenue was 2200 and "
                            "South revenue was 1800."
                        ),
                        is_supported=True,
                    )
                ],
                confidence="high",
            )
        ],
        unsupported_claims=[],
        verification_status="verified",
    )

    visualization = VisualizationResult(
        status="visualized",
        spec=VisualizationSpec(
            chart_type="bar",
            x_column="region",
            y_column="revenue",
            title="revenue by region",
            reason="A categorical comparison is available.",
        ),
    )

    story = BusinessStory(
        summary="North generated higher revenue than South.",
        sections=[
            StorySection(
                heading="Revenue comparison",
                content=(
                    "North generated 2200 in revenue "
                    "compared with 1800 for South."
                ),
            )
        ],
        visualization_explanation=(
            "The bar chart compares revenue between regions."
        ),
        recommendations=[],
    )

    context = SessionContext(
        session_id="test-session-001",
        user_request="Compare North and South revenue.",
        data_source="golden_sales.csv",
        data_schema=make_schema(),
        analytical_plan=plan,
        generated_sql=(
            "SELECT region, revenue FROM golden_sales"
        ),
        sql_result=sql_result,
        analysis_result=analysis_result,
        insights=insights,
        visualization=visualization,
        story=story,
    )

    assert context.session_id == "test-session-001"
    assert context.user_request == (
        "Compare North and South revenue."
    )
    assert context.data_source == "golden_sales.csv"
    assert context.data_schema is not None
    assert context.analytical_plan == plan
    assert context.generated_sql is not None
    assert context.sql_result == sql_result
    assert context.analysis_result == analysis_result
    assert context.insights == insights
    assert context.visualization == visualization
    assert context.story == story


def test_session_context_can_start_with_only_request():
    context = SessionContext(
        session_id="test-session-002",
        user_request="Show total revenue.",
    )

    assert context.session_id == "test-session-002"
    assert context.user_request == "Show total revenue."

    assert context.data_schema is None
    assert context.analytical_plan is None
    assert context.generated_sql is None
    assert context.sql_result is None
    assert context.analysis_result is None
    assert context.insights is None
    assert context.visualization is None
    assert context.story is None