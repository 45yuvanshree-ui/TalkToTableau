from pydantic import BaseModel, Field


class Evidence(BaseModel):
    """Evidence supporting a specific analytical claim."""

    claim: str = Field(
        description="The factual claim being evaluated."
    )

    supporting_data: str = Field(
        description=(
            "A concise description of the returned data that "
            "supports or contradicts the claim."
        )
    )

    is_supported: bool = Field(
        description="Whether the returned data supports the claim."
    )


class Insight(BaseModel):
    """An evidence-backed analytical insight."""

    statement: str = Field(
        description="A concise insight derived from the available data."
    )

    evidence: list[Evidence] = Field(
        default_factory=list,
        description="Evidence supporting the insight."
    )

    confidence: str = Field(
        description=(
            "Confidence level based only on the available evidence: "
            "high, medium, or low."
        )
    )


class InsightAnalysis(BaseModel):
    """Structured output produced by Agent 2."""

    insights: list[Insight] = Field(
        default_factory=list,
        description="Evidence-backed insights."
    )

    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Claims that are not supported by the available data."
    )

    verification_status: str = Field(
        description=(
            "Overall verification status: verified, partially_verified, "
            "or insufficient_evidence."
        )
    )
    