"""Deterministic workflow support for Agent 2 insight verification."""

import re

from models import AnalyticalPlan, InsightAnalysis
from sql.models import SQLResult


_RANK_PATTERNS = {
    "highest": 1,
    "largest": 1,
    "top": 1,
    "most": 1,
    "second-highest": 2,
    "second highest": 2,
    "third-highest": 3,
    "third highest": 3,
    "fourth-highest": 4,
    "fourth highest": 4,
    "fifth-highest": 5,
    "fifth highest": 5,
    "lowest": -1,
    "smallest": -1,
    "bottom": -1,
    "least": -1,
    "second-lowest": -2,
    "second lowest": -2,
    "third-lowest": -3,
    "third lowest": -3,
    "fourth-lowest": -4,
    "fourth lowest": -4,
    "fifth-lowest": -5,
    "fifth lowest": -5,
}


class InsightVerificationWorkflow:
    """
    Controlled verification workflow for Agent 2.

    The LLM may generate interpretations, but deterministic checks
    verify ranking claims against the actual SQL result.
    """

    @staticmethod
    def build_context(
        plan: AnalyticalPlan,
        sql_result: SQLResult,
        analysis_result,
    ) -> dict:
        """Build the complete evidence context for Agent 2."""

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
        Validate the structural correctness of Agent 2 output.

        This does not perform data-dependent validation.
        """

        if result.unsupported_claims:
            if result.verification_status == "verified":
                raise ValueError(
                    "Output cannot be fully verified when "
                    "unsupported claims are present."
                )

        if result.verification_status not in {
            "verified",
            "partially_verified",
            "insufficient_evidence",
        }:
            raise ValueError(
                "Invalid insight verification status."
            )

        for insight in result.insights:
            if insight.confidence not in {
                "high",
                "medium",
                "low",
            }:
                raise ValueError(
                    "Invalid insight confidence level."
                )

            for evidence in insight.evidence:
                if not evidence.is_supported:
                    raise ValueError(
                        "Insight contains unsupported evidence."
                    )

        return result

    @staticmethod
    def validate_against_sql_result(
        result: InsightAnalysis,
        sql_result: SQLResult,
    ) -> InsightAnalysis:
        """
        Deterministically verify ranking claims against SQL results.

        This protects Agent 2 from incorrect LLM-generated statements
        such as claiming that the fourth-highest value is actually
        the second-highest value.

        The method only performs deterministic checks when the SQL
        result has exactly one dimension column and one numeric metric
        column.
        """

        if len(sql_result.columns) != 2:
            return result

        if len(sql_result.rows) < 2:
            return result

        dimension_index = 0
        metric_index = 1

        pairs: list[tuple[str, float]] = []

        for row in sql_result.rows:
            if len(row) <= metric_index:
                continue

            dimension_value = row[dimension_index]
            metric_value = row[metric_index]

            if dimension_value is None:
                continue

            try:
                numeric_value = float(metric_value)
            except (TypeError, ValueError):
                continue

            pairs.append(
                (
                    str(dimension_value),
                    numeric_value,
                )
            )

        if len(pairs) < 2:
            return result

        descending = sorted(
            pairs,
            key=lambda item: item[1],
            reverse=True,
        )

        ascending = sorted(
            pairs,
            key=lambda item: item[1],
        )

        invalid_claims: list[str] = []

        for insight in result.insights:
            statement = insight.statement

            for rank_phrase, rank in _RANK_PATTERNS.items():
                if rank_phrase not in statement.lower():
                    continue

                expected_dimension = (
                    InsightVerificationWorkflow._dimension_for_rank(
                        descending=descending,
                        ascending=ascending,
                        rank=rank,
                    )
                )

                if expected_dimension is None:
                    continue

                claimed_dimension = (
                    InsightVerificationWorkflow._extract_dimension(
                        statement=statement,
                        known_dimensions=[
                            pair[0]
                            for pair in pairs
                        ],
                        rank_phrase=rank_phrase,
                    )
                )

                if claimed_dimension is None:
                    continue

                if (
                    claimed_dimension.lower()
                    != expected_dimension.lower()
                ):
                    invalid_claims.append(statement)

        if not invalid_claims:
            return result

        updated_unsupported_claims = list(
            result.unsupported_claims
        )

        for claim in invalid_claims:
            if claim not in updated_unsupported_claims:
                updated_unsupported_claims.append(claim)

        return result.model_copy(
            update={
                "unsupported_claims": updated_unsupported_claims,
                "verification_status": (
                    "partially_verified"
                    if result.insights
                    else "insufficient_evidence"
                ),
            }
        )

    @staticmethod
    def _dimension_for_rank(
        descending: list[tuple[str, float]],
        ascending: list[tuple[str, float]],
        rank: int,
    ) -> str | None:
        """Return the dimension value expected at a requested rank."""

        if rank > 0:
            index = rank - 1

            if index >= len(descending):
                return None

            return descending[index][0]

        index = abs(rank) - 1

        if index >= len(ascending):
            return None

        return ascending[index][0]

    @staticmethod
    def _extract_dimension(
        statement: str,
        known_dimensions: list[str],
        rank_phrase: str,
    ) -> str | None:
        """
        Extract the dimension value associated with a ranking phrase.

        The extraction is deliberately conservative. It only returns
        a value when it can match one of the actual SQL-result
        dimension values.
        """

        statement_lower = statement.lower()
        phrase_lower = rank_phrase.lower()

        phrase_index = statement_lower.find(
            phrase_lower
        )

        if phrase_index < 0:
            return None

        before_phrase = statement[:phrase_index].strip()

        # Prefer a known dimension appearing immediately before the
        # ranking phrase.
        for dimension in sorted(
            known_dimensions,
            key=len,
            reverse=True,
        ):
            dimension_lower = dimension.lower()

            if before_phrase.lower().endswith(
                dimension_lower
            ):
                return dimension

        # Also check common grammatical forms such as:
        # "North has the highest total revenue."
        match = re.search(
            r"(.+?)\s+has\s+(?:the\s+)?"
            + re.escape(rank_phrase)
            + r"\b",
            statement,
            re.IGNORECASE,
        )

        if match:
            candidate = match.group(1).strip()

            for dimension in sorted(
                known_dimensions,
                key=len,
                reverse=True,
            ):
                if candidate.lower().endswith(
                    dimension.lower()
                ):
                    return dimension

        return None