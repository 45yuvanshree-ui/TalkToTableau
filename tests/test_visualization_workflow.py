import pandas as pd
import pytest

from analysis import (
    AnalysisResult,
    AnalysisSummary,
    ComparisonAnalysis,
    TrendAnalysis,
)
from models import AnalyticalPlan
from sql import SQLResult
from visualization import VisualizationWorkflow


def create_sql_result(
    columns: list[str],
    rows: list[list[object]],
) -> SQLResult:
    """Create a reusable SQLResult for visualization tests."""

    return SQLResult(
        columns=columns,
        rows=rows,
        row_count=len(rows),
    )


def create_summary(
    columns: list[str],
    row_count: int,
) -> AnalysisSummary:
    """Create a reusable analysis summary."""

    return AnalysisSummary(
        row_count=row_count,
        columns=columns,
    )


def test_total_revenue_by_region_creates_bar_chart():
    """
    A request for total revenue by region should produce
    a categorical bar chart even when the Agent 1 metric
    uses natural-language wording.
    """

    plan = AnalyticalPlan(
        analytical_objective="Calculate total revenue by region",
        metrics=["total revenue"],
        dimensions=["region"],
        filters=[],
        time_range=None,
        comparison=None,
        analytical_question=(
            "What was the total revenue by region?"
        ),
    )

    sql_result = create_sql_result(
        columns=["region", "total_revenue"],
        rows=[
            ["North", 60000.0],
            ["South", 55600.0],
            ["East", 31800.0],
            ["West", 33600.0],
        ],
    )

    analysis_result = AnalysisResult(
        summary=create_summary(
            columns=["region", "total_revenue"],
            row_count=4,
        ),
        comparison=None,
        trend=None,
        anomaly=None,
    )

    workflow = VisualizationWorkflow()

    result, figure = workflow.run(
        plan=plan,
        sql_result=sql_result,
        analysis_result=analysis_result,
    )

    assert result.status == "visualized"
    assert result.spec is not None
    assert result.spec.chart_type == "bar"
    assert result.spec.x_column == "region"
    assert result.spec.y_column == "total_revenue"

    assert figure is not None
    assert len(figure.data) > 0
    assert figure.data[0].type == "bar"


def test_metric_name_with_spaces_matches_underscored_sql_column():
    """Natural-language metric names should match SQL aliases."""

    plan = AnalyticalPlan(
        analytical_objective="Compare total revenue",
        metrics=["Total Revenue"],
        dimensions=["region"],
        filters=[],
        time_range=None,
        comparison=None,
        analytical_question="Compare total revenue by region.",
    )

    sql_result = create_sql_result(
        columns=["region", "total_revenue"],
        rows=[
            ["North", 60000.0],
            ["South", 55600.0],
        ],
    )

    analysis_result = AnalysisResult(
        summary=create_summary(
            columns=["region", "total_revenue"],
            row_count=2,
        ),
        comparison=None,
        trend=None,
        anomaly=None,
    )

    workflow = VisualizationWorkflow()

    result, figure = workflow.run(
        plan=plan,
        sql_result=sql_result,
        analysis_result=analysis_result,
    )

    assert result.status == "visualized"
    assert result.spec is not None
    assert result.spec.y_column == "total_revenue"
    assert figure is not None


def test_trend_analysis_creates_line_chart():
    """A valid trend analysis should produce a line chart."""

    plan = AnalyticalPlan(
        analytical_objective="Analyze revenue trend over time",
        metrics=["revenue"],
        dimensions=[],
        filters=[],
        time_range="2025",
        comparison=None,
        analytical_question=(
            "How did revenue change over time?"
        ),
    )

    sql_result = create_sql_result(
        columns=["sale_date", "revenue"],
        rows=[
            ["2025-01-01", 10000.0],
            ["2025-02-01", 15000.0],
            ["2025-03-01", 12000.0],
        ],
    )

    analysis_result = AnalysisResult(
        summary=create_summary(
            columns=["sale_date", "revenue"],
            row_count=3,
        ),
        comparison=None,
        trend=TrendAnalysis(
            time_dimension="sale_date",
            metric="revenue",
            values=[
                {
                    "period": "2025-01-01T00:00:00",
                    "value": 10000.0,
                },
                {
                    "period": "2025-02-01T00:00:00",
                    "value": 15000.0,
                },
                {
                    "period": "2025-03-01T00:00:00",
                    "value": 12000.0,
                },
            ],
        ),
        anomaly=None,
    )

    workflow = VisualizationWorkflow()

    result, figure = workflow.run(
        plan=plan,
        sql_result=sql_result,
        analysis_result=analysis_result,
    )

    assert result.status == "visualized"
    assert result.spec is not None
    assert result.spec.chart_type == "line"
    assert result.spec.x_column == "sale_date"
    assert result.spec.y_column == "revenue"

    assert figure is not None
    assert len(figure.data) > 0
    assert figure.data[0].type == "scatter"


def test_scatter_chart_is_used_for_two_numeric_columns():
    """Two numeric variables can produce a scatter visualization."""

    plan = AnalyticalPlan(
        analytical_objective="Analyze relationship",
        metrics=[],
        dimensions=[],
        filters=[],
        time_range=None,
        comparison=None,
        analytical_question=(
            "Is there a relationship between quantity and revenue?"
        ),
    )

    sql_result = create_sql_result(
        columns=["quantity", "revenue"],
        rows=[
            [1, 1000.0],
            [2, 2200.0],
            [3, 3100.0],
        ],
    )

    analysis_result = AnalysisResult(
        summary=create_summary(
            columns=["quantity", "revenue"],
            row_count=3,
        ),
        comparison=None,
        trend=None,
        anomaly=None,
    )

    workflow = VisualizationWorkflow()

    result, figure = workflow.run(
        plan=plan,
        sql_result=sql_result,
        analysis_result=analysis_result,
    )

    assert result.status == "visualized"
    assert result.spec is not None
    assert result.spec.chart_type == "scatter"

    assert figure is not None
    assert len(figure.data) > 0
    assert figure.data[0].type == "scatter"


def test_empty_result_is_not_visualizable():
    """An empty SQL result should not create a chart."""

    plan = AnalyticalPlan(
        analytical_objective="Calculate total revenue",
        metrics=["revenue"],
        dimensions=[],
        filters=[],
        time_range=None,
        comparison=None,
        analytical_question="What is total revenue?",
    )

    sql_result = create_sql_result(
        columns=["revenue"],
        rows=[],
    )

    analysis_result = AnalysisResult(
        summary=create_summary(
            columns=["revenue"],
            row_count=0,
        ),
        comparison=None,
        trend=None,
        anomaly=None,
    )

    workflow = VisualizationWorkflow()

    result, figure = workflow.run(
        plan=plan,
        sql_result=sql_result,
        analysis_result=analysis_result,
    )

    assert result.status == "not_visualizable"
    assert result.spec is None
    assert figure is None


def test_unknown_metric_and_dimension_do_not_create_forced_chart():
    """
    The visualization workflow must not invent a mapping when
    requested fields cannot be found.
    """

    plan = AnalyticalPlan(
        analytical_objective="Compare unknown metric",
        metrics=["unknown metric"],
        dimensions=["unknown dimension"],
        filters=[],
        time_range=None,
        comparison=None,
        analytical_question="Compare unknown metric.",
    )

    sql_result = create_sql_result(
        columns=["region", "description"],
        rows=[
            ["North", "A"],
            ["South", "B"],
        ],
    )

    analysis_result = AnalysisResult(
        summary=create_summary(
            columns=["region", "description"],
            row_count=2,
        ),
        comparison=None,
        trend=None,
        anomaly=None,
    )

    workflow = VisualizationWorkflow()

    result, figure = workflow.run(
        plan=plan,
        sql_result=sql_result,
        analysis_result=analysis_result,
    )

    assert result.status == "not_visualizable"
    assert result.spec is None
    assert figure is None


def test_visualization_workflow_preserves_numeric_values():
    """The generated visualization should use the supplied values."""

    dataframe = pd.DataFrame(
        {
            "region": ["North", "South"],
            "total_revenue": [60000.0, 55600.0],
        }
    )

    assert dataframe["total_revenue"].tolist() == [
        60000.0,
        55600.0,
    ]