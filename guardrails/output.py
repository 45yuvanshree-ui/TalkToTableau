from models import BusinessStory, InsightAnalysis


ALLOWED_VERIFICATION_STATUSES = {
    "verified",
    "partially_verified",
    "insufficient_evidence",
}

ALLOWED_CONFIDENCE_LEVELS = {
    "high",
    "medium",
    "low",
}


def validate_insight_output(
    result: InsightAnalysis,
) -> tuple[bool, str | None]:
    """
    Validate Agent 2 output before it reaches Agent 3.
    """

    if result.verification_status not in ALLOWED_VERIFICATION_STATUSES:
        return (
            False,
            "Invalid insight verification status.",
        )

    for insight in result.insights:
        if insight.confidence not in ALLOWED_CONFIDENCE_LEVELS:
            return (
                False,
                "Invalid insight confidence level.",
            )

        for evidence in insight.evidence:
            if not evidence.is_supported:
                return (
                    False,
                    "Insight contains unsupported evidence.",
                )

    if (
        result.verification_status == "verified"
        and result.unsupported_claims
    ):
        return (
            False,
            "Verified output cannot contain unsupported claims.",
        )

    return True, None


def validate_story_output(
    story: BusinessStory,
) -> tuple[bool, str | None]:
    """
    Validate Agent 3 output before it reaches the user.
    """

    if not story.summary.strip():
        return False, "Story summary is empty."

    if not story.visualization_explanation.strip():
        return False, "Visualization explanation is empty."

    for section in story.sections:
        if not section.heading.strip():
            return False, "Story section heading is empty."

        if not section.content.strip():
            return False, "Story section content is empty."

    for recommendation in story.recommendations:
        if not recommendation.strip():
            return False, "Story contains an empty recommendation."

    return True, None