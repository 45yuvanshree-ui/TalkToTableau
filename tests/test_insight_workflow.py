import pytest

from agent_components import InsightVerificationWorkflow
from analysis import AnalysisResult, AnalysisSummary
from models import AnalyticalPlan, Evidence, Insight, InsightAnalysis
from sql import SQLResult


def make_plan() -> AnalyticalPlan:
    return AnalyticalPlan(
        analytical_objective="comparison",
        metrics=["revenue"],
        dimensions=["region"],
        comparison="North vs South",
        analytical_question="Compare revenue between regions.",
    )


def make_sql_result() -> SQLResult:
    return SQLResult(
        columns=["region", "revenue"],
        rows=[
            ["North", 2200],
            ["South", 1700],
        ],
        row_count=2,
    )


def make_analysis_result() -> AnalysisResult:
    return AnalysisResult(
        summary=AnalysisSummary(
            row_count=2,
            columns=["region", "revenue"],
        )
    )


def test_context_contains_required_evidence_sources():
    workflow = InsightVerificationWorkflow()

    context = workflow.build_context(
        plan=make_plan(),
        sql_result=make_sql_result(),
        analysis_result=make_analysis_result(),
    )

    assert "analytical_plan" in context
    assert "sql_result" in context
    assert "analysis_result" in context

    assert context["sql_result"]["row_count"] == 2
    assert context["analytical_plan"]["metrics"] == ["revenue"]


def test_valid_verified_output_is_accepted():
    result = InsightAnalysis(
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
                            "South revenue was 1700."
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

    validated = InsightVerificationWorkflow.validate_output(
        result
    )

    assert validated.verification_status == "verified"


def test_unsupported_claim_cannot_be_marked_verified():
    result = InsightAnalysis(
        insights=[],
        unsupported_claims=["North has the highest profit."],
        verification_status="verified",
    )

    with pytest.raises(ValueError):
        InsightVerificationWorkflow.validate_output(result)


def test_unsupported_evidence_cannot_be_used_for_insight():
    result = InsightAnalysis(
        insights=[
            Insight(
                statement="North is more profitable.",
                evidence=[
                    Evidence(
                        claim="North is more profitable.",
                        supporting_data="Profit was not returned.",
                        is_supported=False,
                    )
                ],
                confidence="low",
            )
        ],
        unsupported_claims=["North is more profitable."],
        verification_status="insufficient_evidence",
    )

    with pytest.raises(ValueError):
        InsightVerificationWorkflow.validate_output(result)


def test_correct_highest_claim_is_preserved():
    result = InsightAnalysis(
        insights=[
            Insight(
                statement="North has the highest revenue.",
                evidence=[
                    Evidence(
                        claim="North has the highest revenue.",
                        supporting_data=(
                            "North revenue is 60000, greater than "
                            "all other returned regional values."
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

    sql_result = SQLResult(
        columns=["region", "revenue"],
        rows=[
            ["North", 60000],
            ["East", 31800],
            ["West", 33600],
            ["South", 55600],
        ],
        row_count=4,
    )

    validated = (
        InsightVerificationWorkflow
        .validate_against_sql_result(
            result=result,
            sql_result=sql_result,
        )
    )

    assert validated.verification_status == "verified"
    assert validated.unsupported_claims == []


def test_incorrect_second_highest_claim_is_detected():
    result = InsightAnalysis(
        insights=[
            Insight(
                statement="East has the second-highest total revenue.",
                evidence=[
                    Evidence(
                        claim=(
                            "East has the second-highest "
                            "total revenue."
                        ),
                        supporting_data=(
                            "East has total revenue of 31800."
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

    sql_result = SQLResult(
        columns=["region", "total_revenue"],
        rows=[
            ["North", 60000],
            ["East", 31800],
            ["West", 33600],
            ["South", 55600],
        ],
        row_count=4,
    )

    validated = (
        InsightVerificationWorkflow
        .validate_against_sql_result(
            result=result,
            sql_result=sql_result,
        )
    )

    assert validated.verification_status == "partially_verified"

    assert (
        "East has the second-highest total revenue."
        in validated.unsupported_claims
    )


def test_incorrect_fourth_highest_claim_is_detected():
    result = InsightAnalysis(
        insights=[
            Insight(
                statement="South has the fourth-highest total revenue.",
                evidence=[
                    Evidence(
                        claim=(
                            "South has the fourth-highest "
                            "total revenue."
                        ),
                        supporting_data=(
                            "South has total revenue of 55600."
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

    sql_result = SQLResult(
        columns=["region", "total_revenue"],
        rows=[
            ["North", 60000],
            ["East", 31800],
            ["West", 33600],
            ["South", 55600],
        ],
        row_count=4,
    )

    validated = (
        InsightVerificationWorkflow
        .validate_against_sql_result(
            result=result,
            sql_result=sql_result,
        )
    )

    assert validated.verification_status == "partially_verified"

    assert (
        "South has the fourth-highest total revenue."
        in validated.unsupported_claims
    )


def test_correct_lowest_claim_is_preserved():
    result = InsightAnalysis(
        insights=[
            Insight(
                statement="East has the lowest total revenue.",
                evidence=[
                    Evidence(
                        claim="East has the lowest total revenue.",
                        supporting_data=(
                            "East has total revenue of 31800, "
                            "the lowest returned value."
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

    sql_result = SQLResult(
        columns=["region", "total_revenue"],
        rows=[
            ["North", 60000],
            ["East", 31800],
            ["West", 33600],
            ["South", 55600],
        ],
        row_count=4,
    )

    validated = (
        InsightVerificationWorkflow
        .validate_against_sql_result(
            result=result,
            sql_result=sql_result,
        )
    )

    assert validated.verification_status == "verified"
    assert validated.unsupported_claims == []


def test_incorrect_lowest_claim_is_detected():
    result = InsightAnalysis(
        insights=[
            Insight(
                statement="South has the lowest total revenue.",
                evidence=[
                    Evidence(
                        claim="South has the lowest total revenue.",
                        supporting_data=(
                            "South has total revenue of 55600."
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

    sql_result = SQLResult(
        columns=["region", "total_revenue"],
        rows=[
            ["North", 60000],
            ["East", 31800],
            ["West", 33600],
            ["South", 55600],
        ],
        row_count=4,
    )

    validated = (
        InsightVerificationWorkflow
        .validate_against_sql_result(
            result=result,
            sql_result=sql_result,
        )
    )

    assert validated.verification_status == "partially_verified"

    assert (
        "South has the lowest total revenue."
        in validated.unsupported_claims
    )


def test_non_ranking_insight_is_not_modified():
    result = InsightAnalysis(
        insights=[
            Insight(
                statement="Revenue varies across regions.",
                evidence=[
                    Evidence(
                        claim="Revenue varies across regions.",
                        supporting_data=(
                            "The returned regional revenue "
                            "values are different."
                        ),
                        is_supported=True,
                    )
                ],
                confidence="medium",
            )
        ],
        unsupported_claims=[],
        verification_status="verified",
    )

    sql_result = SQLResult(
        columns=["region", "revenue"],
        rows=[
            ["North", 60000],
            ["East", 31800],
            ["West", 33600],
            ["South", 55600],
        ],
        row_count=4,
    )

    validated = (
        InsightVerificationWorkflow
        .validate_against_sql_result(
            result=result,
            sql_result=sql_result,
        )
    )

    assert validated.verification_status == "verified"
    assert validated.unsupported_claims == []