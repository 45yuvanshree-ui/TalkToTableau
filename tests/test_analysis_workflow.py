from analysis import AnalysisWorkflow
from models import AnalyticalPlan
from sql import SQLResult


def make_result() -> SQLResult:
    return SQLResult(
        columns=[
            "sale_date",
            "region",
            "revenue",
        ],
        rows=[
            ["2025-01-10", "North", 1000],
            ["2025-01-11", "South", 800],
            ["2025-01-12", "North", 1200],
            ["2025-01-13", "South", 900],
        ],
        row_count=4,
    )


def test_basic_summary():
    plan = AnalyticalPlan(
        analytical_objective="aggregation",
        metrics=["revenue"],
        dimensions=[],
        analytical_question="What is the revenue?",
    )

    result = AnalysisWorkflow().run(
        plan=plan,
        sql_result=make_result(),
    )

    assert result.summary.row_count == 4
    assert "revenue" in result.summary.columns
    assert result.comparison is None
    assert result.trend is None
    assert result.anomaly is None


def test_comparison_analysis():
    plan = AnalyticalPlan(
        analytical_objective="comparison",
        metrics=["revenue"],
        dimensions=["region"],
        comparison="North vs South",
        analytical_question="Compare revenue by region.",
    )

    result = AnalysisWorkflow().run(
        plan=plan,
        sql_result=make_result(),
    )

    assert result.comparison is not None
    assert result.comparison.dimension == "region"
    assert result.comparison.metric == "revenue"
    assert result.comparison.values["North"] == 2200.0
    assert result.comparison.values["South"] == 1700.0


def test_trend_analysis():
    plan = AnalyticalPlan(
        analytical_objective="trend",
        metrics=["revenue"],
        dimensions=["sale_date"],
        time_range="2025",
        analytical_question="What is the revenue trend?",
    )

    result = AnalysisWorkflow().run(
        plan=plan,
        sql_result=make_result(),
    )

    assert result.trend is not None
    assert result.trend.time_dimension == "sale_date"
    assert result.trend.metric == "revenue"
    assert len(result.trend.values) == 4


def test_anomaly_analysis():
    plan = AnalyticalPlan(
        analytical_objective="anomaly",
        metrics=["revenue"],
        dimensions=[],
        analytical_question="Are there revenue anomalies?",
    )

    result = AnalysisWorkflow().run(
        plan=plan,
        sql_result=make_result(),
    )

    assert result.anomaly is not None
    assert result.anomaly.metric == "revenue"
    assert result.anomaly.mean == 975.0
    assert result.anomaly.standard_deviation > 0


def test_analysis_does_not_run_unrequested_operations():
    plan = AnalyticalPlan(
        analytical_objective="aggregation",
        metrics=["revenue"],
        dimensions=["region"],
        analytical_question="What is the revenue?",
    )

    result = AnalysisWorkflow().run(
        plan=plan,
        sql_result=make_result(),
    )

    assert result.comparison is None
    assert result.trend is None
    assert result.anomaly is None


def test_missing_metric_is_rejected():
    plan = AnalyticalPlan(
        analytical_objective="comparison",
        metrics=["profit"],
        dimensions=["region"],
        comparison="North vs South",
        analytical_question="Compare profit by region.",
    )

    try:
        AnalysisWorkflow().run(
            plan=plan,
            sql_result=make_result(),
        )
    except ValueError as error:
        assert "Comparison metric not found" in str(error)
    else:
        raise AssertionError("Expected ValueError")