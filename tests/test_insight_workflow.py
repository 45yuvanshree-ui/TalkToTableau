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

    validated = InsightVerificationWorkflow.validate_output(result)

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