from analysis import AnalysisResult, AnalysisSummary, ComparisonAnalysis
from models import AnalyticalPlan
from sql import SQLResult
from visualization import VisualizationWorkflow


def make_comparison_result() -> SQLResult:
    return SQLResult(
        columns=["region", "revenue"],
        rows=[
            ["North", 2200],
            ["South", 1700],
            ["East", 1400],
            ["West", 1900],
        ],
        row_count=4,
    )


def make_comparison_analysis() -> AnalysisResult:
    return AnalysisResult(
        summary=AnalysisSummary(
            row_count=4,
            columns=["region", "revenue"],
        ),
        comparison=ComparisonAnalysis(
            dimension="region",
            metric="revenue",
            values={
                "North": 2200.0,
                "South": 1700.0,
                "East": 1400.0,
                "West": 1900.0,
            },
        ),
    )


def test_comparison_selects_bar_chart():
    plan = AnalyticalPlan(
        analytical_objective="comparison",
        metrics=["revenue"],
        dimensions=["region"],
        comparison="North vs South",
        analytical_question="Compare revenue by region.",
    )

    status, figure = VisualizationWorkflow().run(
        plan=plan,
        sql_result=make_comparison_result(),
        analysis_result=make_comparison_analysis(),
    )

    assert status.status == "visualized"
    assert status.spec is not None
    assert status.spec.chart_type == "bar"
    assert figure is not None
    assert len(figure.data) > 0
    assert figure.data[0].type == "bar"


def test_trend_selects_line_chart():
    sql_result = SQLResult(
        columns=["sale_date", "revenue"],
        rows=[
            ["2025-01-01", 1000],
            ["2025-02-01", 1200],
            ["2025-03-01", 1500],
        ],
        row_count=3,
    )

    analysis_result = AnalysisResult(
        summary=AnalysisSummary(
            row_count=3,
            columns=["sale_date", "revenue"],
        ),
        trend={
            "time_dimension": "sale_date",
            "metric": "revenue",
            "values": [
                {
                    "period": "2025-01-01T00:00:00",
                    "value": 1000.0,
                },
                {
                    "period": "2025-02-01T00:00:00",
                    "value": 1200.0,
                },
                {
                    "period": "2025-03-01T00:00:00",
                    "value": 1500.0,
                },
            ],
        },
    )

    plan = AnalyticalPlan(
        analytical_objective="trend",
        metrics=["revenue"],
        dimensions=["sale_date"],
        time_range="2025",
        analytical_question="Show the revenue trend.",
    )

    status, figure = VisualizationWorkflow().run(
        plan=plan,
        sql_result=sql_result,
        analysis_result=analysis_result,
    )

    assert status.status == "visualized"
    assert status.spec is not None
    assert status.spec.chart_type == "line"
    assert figure is not None
    assert figure.data[0].type == "scatter"


def test_two_numeric_columns_select_scatter():
    sql_result = SQLResult(
        columns=["quantity", "revenue"],
        rows=[
            [1, 100],
            [2, 200],
            [3, 350],
        ],
        row_count=3,
    )

    analysis_result = AnalysisResult(
        summary=AnalysisSummary(
            row_count=3,
            columns=["quantity", "revenue"],
        )
    )

    plan = AnalyticalPlan(
        analytical_objective="relationship",
        metrics=["revenue"],
        dimensions=[],
        analytical_question="Explore the relationship.",
    )

    status, figure = VisualizationWorkflow().run(
        plan=plan,
        sql_result=sql_result,
        analysis_result=analysis_result,
    )

    assert status.status == "visualized"
    assert status.spec is not None
    assert status.spec.chart_type == "scatter"
    assert figure is not None
    assert figure.data[0].type == "scatter"


def test_empty_result_is_not_visualized():
    sql_result = SQLResult(
        columns=["region", "revenue"],
        rows=[],
        row_count=0,
    )

    analysis_result = AnalysisResult(
        summary=AnalysisSummary(
            row_count=0,
            columns=["region", "revenue"],
        )
    )

    plan = AnalyticalPlan(
        analytical_objective="aggregation",
        metrics=["revenue"],
        dimensions=["region"],
        analytical_question="Show revenue.",
    )

    status, figure = VisualizationWorkflow().run(
        plan=plan,
        sql_result=sql_result,
        analysis_result=analysis_result,
    )

    assert status.status == "not_visualizable"
    assert status.spec is None
    assert figure is None