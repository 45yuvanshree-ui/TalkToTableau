from models import AnalyticalPlan, InsightAnalysis
from sql import SQLResult

from analysis import AnalysisResult


class InsightVerificationWorkflow:
    """
    Prepare analytical context for Agent 2.

    This workflow keeps the analytical plan, SQL result, and
    deterministic analysis together so Agent 2 can verify
    evidence without accessing the data source directly.
    """

    def build_context(
        self,
        plan: AnalyticalPlan,
        sql_result: SQLResult,
        analysis_result: AnalysisResult,
    ) -> dict[str, object]:
        """Build the structured context supplied to Agent 2."""

        return {
            "analytical_plan": plan.model_dump(),
            "sql_result": sql_result.model_dump(),
            "analysis_result": analysis_result.model_dump(),
        }

    @staticmethod
    def validate_output(
        result: InsightAnalysis,
    ) -> InsightAnalysis:
        """
        Validate the structural boundaries of Agent 2 output.

        Pydantic has already validated the field types. This method
        additionally ensures that unsupported claims cannot be
        reported as an overall fully verified result.
        """

        if result.unsupported_claims:
            if result.verification_status == "verified":
                raise ValueError(
                    "Output cannot be fully verified when "
                    "unsupported claims are present."
                )

        for insight in result.insights:
            for evidence in insight.evidence:
                if not evidence.is_supported:
                    raise ValueError(
                        "An insight cannot contain unsupported evidence."
                    )

        return result