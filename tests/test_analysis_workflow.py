from analysis import AnalysisWorkflow
from models import AnalyticalPlan
from sql import SQLResult


def create_plan(
    objective: str,
    metrics: list[str],
    dimensions: list[str],
    comparison: str | None = None,
) -> AnalyticalPlan:
    """Create a reusable analytical plan for tests."""

    return AnalyticalPlan(
        analytical_objective=objective,
        metrics=metrics,
        dimensions=dimensions,
        filters=[],
        time_range=None,
        comparison=comparison,
        analytical_question=objective,
    )


def create_sql_result(
    columns: list[str],
    rows: list[list[object]],
) -> SQLResult:
    """Create a reusable SQL result for tests."""

    return SQLResult(
        columns=columns,
        rows=rows,
        row_count=len(rows),
    )


def test_comparison_analysis():
    """Test deterministic comparison analysis."""

    plan = create_plan(
        objective="Compare revenue by region",
        metrics=["revenue"],
        dimensions=["region"],
        comparison="by region",
    )

    sql_result = create_sql_result(
        columns=["region", "revenue"],
        rows=[
            ["North", 100.0],
            ["South", 200.0],
            ["East", 50.0],
        ],
    )

    workflow = AnalysisWorkflow()

    result = workflow.run(
        plan=plan,
        sql_result=sql_result,
    )

    assert result.summary.row_count == 3

    assert result.comparison is not None
    assert result.comparison.dimension == "region"
    assert result.comparison.metric == "revenue"

    assert result.comparison.values == {
        "North": 100.0,
        "South": 200.0,
        "East": 50.0,
    }


def test_total_revenue_matches_underscored_sql_column():
    """
    The analysis workflow should resolve the natural-language
    metric "total revenue" to the SQL result column
    "total_revenue".
    """

    plan = create_plan(
        objective="Calculate total revenue by region",
        metrics=["total revenue"],
        dimensions=["region"],
    )

    sql_result = create_sql_result(
        columns=[
            "region",
            "total_revenue",
        ],
        rows=[
            ["North", 60000.0],
            ["South", 55600.0],
            ["East", 31800.0],
            ["West", 33600.0],
        ],
    )

    workflow = AnalysisWorkflow()

    result = workflow.run(
        plan=plan,
        sql_result=sql_result,
    )

    assert result.comparison is not None

    assert result.comparison.dimension == "region"
    assert result.comparison.metric == "total_revenue"

    assert result.comparison.values == {
        "North": 60000.0,
        "South": 55600.0,
        "East": 31800.0,
        "West": 33600.0,
    }


def test_comparison_is_detected_without_explicit_comparison_field():
    """
    A dimension + metric request such as "revenue by region"
    should be treated as a comparison even when comparison
    is not explicitly populated by Agent 1.
    """

    plan = create_plan(
        objective="Calculate total revenue by region",
        metrics=["revenue"],
        dimensions=["region"],
        comparison=None,
    )

    sql_result = create_sql_result(
        columns=[
            "region",
            "revenue",
        ],
        rows=[
            ["North", 100.0],
            ["South", 200.0],
        ],
    )

    workflow = AnalysisWorkflow()

    result = workflow.run(
        plan=plan,
        sql_result=sql_result,
    )

    assert result.comparison is not None
    assert result.comparison.dimension == "region"
    assert result.comparison.metric == "revenue"


def test_missing_metric_is_rejected():
    """Unknown metrics must not be silently invented."""

    plan = create_plan(
        objective="Calculate total profit by region",
        metrics=["profit"],
        dimensions=["region"],
    )

    sql_result = create_sql_result(
        columns=[
            "region",
            "revenue",
        ],
        rows=[
            ["North", 100.0],
        ],
    )

    workflow = AnalysisWorkflow()

    try:
        workflow.run(
            plan=plan,
            sql_result=sql_result,
        )
    except ValueError as error:
        assert "profit" in str(error).lower()
    else:
        raise AssertionError(
            "Expected missing metric to raise ValueError."
        )


def test_missing_dimension_is_rejected():
    """Unknown dimensions must not be silently invented."""

    plan = create_plan(
        objective="Calculate revenue by country",
        metrics=["revenue"],
        dimensions=["country"],
    )

    sql_result = create_sql_result(
        columns=[
            "region",
            "revenue",
        ],
        rows=[
            ["North", 100.0],
        ],
    )

    workflow = AnalysisWorkflow()

    try:
        workflow.run(
            plan=plan,
            sql_result=sql_result,
        )
    except ValueError as error:
        assert "country" in str(error).lower()
    else:
        raise AssertionError(
            "Expected missing dimension to raise ValueError."
        )


def test_trend_analysis():
    """Test deterministic trend analysis."""

    plan = create_plan(
        objective="Analyze revenue trend over time",
        metrics=["revenue"],
        dimensions=[],
    )

    plan.time_range = "2025"

    sql_result = create_sql_result(
        columns=[
            "sale_date",
            "revenue",
        ],
        rows=[
            ["2025-01-01", 100.0],
            ["2025-02-01", 200.0],
            ["2025-03-01", 150.0],
        ],
    )

    workflow = AnalysisWorkflow()

    result = workflow.run(
        plan=plan,
        sql_result=sql_result,
    )

    assert result.trend is not None
    assert result.trend.time_dimension == "sale_date"
    assert result.trend.metric == "revenue"
    assert len(result.trend.values) == 3


def test_anomaly_analysis():
    """Test deterministic anomaly analysis."""

    plan = create_plan(
        objective="Detect revenue anomalies",
        metrics=["revenue"],
        dimensions=[],
    )

    sql_result = create_sql_result(
        columns=["revenue"],
        rows=[
            [100.0],
            [110.0],
            [105.0],
            [1000.0],
        ],
    )

    workflow = AnalysisWorkflow()

    result = workflow.run(
        plan=plan,
        sql_result=sql_result,
    )

    assert result.anomaly is not None
    assert result.anomaly.metric == "revenue"
    assert result.anomaly.mean is not None
    assert result.anomaly.standard_deviation is not None


def test_empty_result_produces_summary():
    """An empty SQL result should still produce a valid summary."""

    plan = create_plan(
        objective="Calculate revenue",
        metrics=["revenue"],
        dimensions=[],
    )

    sql_result = create_sql_result(
        columns=["revenue"],
        rows=[],
    )

    workflow = AnalysisWorkflow()

    result = workflow.run(
        plan=plan,
        sql_result=sql_result,
    )

    assert result.summary.row_count == 0
    assert result.summary.columns == ["revenue"]
    assert result.comparison is None
    assert result.trend is None
    assert result.anomaly is None