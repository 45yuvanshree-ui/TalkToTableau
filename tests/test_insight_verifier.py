from models import (
    Evidence,
    Insight,
    InsightAnalysis,
)


def test_valid_high_confidence_insight():
    """High confidence is accepted."""

    result = InsightAnalysis(
        insights=[
            Insight(
                statement="North has the highest revenue.",
                evidence=[
                    Evidence(
                        claim="North has the highest revenue.",
                        supporting_data=(
                            "North has total revenue of 60000."
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

    assert result.insights[0].confidence == "high"
    assert result.verification_status == "verified"


def test_valid_medium_confidence_insight():
    """Medium confidence is accepted."""

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

    assert result.insights[0].confidence == "medium"


def test_valid_low_confidence_insight():
    """Low confidence is accepted."""

    result = InsightAnalysis(
        insights=[
            Insight(
                statement="There may be a regional pattern.",
                evidence=[
                    Evidence(
                        claim="There may be a regional pattern.",
                        supporting_data=(
                            "The returned data contains "
                            "different regional values."
                        ),
                        is_supported=True,
                    )
                ],
                confidence="low",
            )
        ],
        unsupported_claims=[],
        verification_status="partially_verified",
    )

    assert result.insights[0].confidence == "low"


def test_insufficient_evidence_output():
    """Insufficient evidence is a valid verification state."""

    result = InsightAnalysis(
        insights=[],
        unsupported_claims=[
            "The data does not establish why revenue differs."
        ],
        verification_status="insufficient_evidence",
    )

    assert result.insights == []
    assert result.verification_status == "insufficient_evidence"


def test_supported_evidence_is_marked_true():
    """Evidence can explicitly indicate support."""

    evidence = Evidence(
        claim="North has the highest revenue.",
        supporting_data=(
            "North has revenue of 60000, greater than "
            "the other returned regional values."
        ),
        is_supported=True,
    )

    assert evidence.is_supported is True


def test_unsupported_evidence_can_be_represented():
    """
    The model can represent unsupported evidence so the
    guardrail layer can reject it later.
    """

    evidence = Evidence(
        claim="North is more popular with customers.",
        supporting_data=(
            "No customer preference data was supplied."
        ),
        is_supported=False,
    )

    assert evidence.is_supported is False