from agent_components import insight_verifier
from models import Evidence, Insight, InsightAnalysis


def test_insight_verifier_configuration():
    assert insight_verifier.name == "Insight and Evidence Verifier"
    assert insight_verifier.output_type is InsightAnalysis


def test_evidence_model():
    evidence = Evidence(
        claim="North generated higher revenue than South.",
        supporting_data=(
            "North revenue was 2200 and South revenue was 1700."
        ),
        is_supported=True,
    )

    assert evidence.is_supported is True
    assert "2200" in evidence.supporting_data


def test_insight_model():
    insight = Insight(
        statement="North generated higher revenue than South.",
        evidence=[
            Evidence(
                claim="North generated higher revenue than South.",
                supporting_data=(
                    "North revenue was 2200 and South revenue was 1700."
                ),
                is_supported=True,
            )
        ],
        confidence="high",
    )

    assert insight.confidence == "high"
    assert len(insight.evidence) == 1


def test_insight_analysis_model():
    result = InsightAnalysis(
        insights=[],
        unsupported_claims=[
            "North had the highest profit."
        ],
        verification_status="partially_verified",
    )

    assert result.verification_status == "partially_verified"
    assert len(result.unsupported_claims) == 1