from agents import Agent

from agent_components import (
    create_insight_verifier,
    insight_verifier,
)
from models import (
    Evidence,
    Insight,
    InsightAnalysis,
)


def test_insight_verifier_configuration():
    assert insight_verifier.name == (
        "Insight and Evidence Verifier"
    )
    assert insight_verifier.output_type is InsightAnalysis


def test_insight_verifier_can_accept_injected_model():
    agent = create_insight_verifier()

    assert isinstance(agent, Agent)
    assert agent.name == (
        "Insight and Evidence Verifier"
    )


def test_insight_analysis_structure():
    result = InsightAnalysis(
        insights=[
            Insight(
                statement="North generated higher revenue.",
                evidence=[
                    Evidence(
                        claim="North generated higher revenue.",
                        supporting_data=(
                            "North revenue was 2200."
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

    assert len(result.insights) == 1
    assert result.insights[0].confidence == "high"
    assert result.verification_status == "verified"