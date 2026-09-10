"""Deterministic workflow support for Agent 2 insight verification."""

import re
import math

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
    verify claims against the actual SQL result.

    Validation covers:
    - structural validity
    - scalar numerical evidence
    - ranking claims
    - unsupported claims
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

        This checks:
        - verification status
        - confidence levels
        - unsupported evidence markers

        Data-dependent validation is performed separately by
        validate_against_sql_result().
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
        Deterministically verify Agent 2 claims against SQL results.

        Validation includes:

        1. Scalar numerical evidence
           When SQL returns exactly one row, numerical values cited
           as evidence must be present in the returned SQL result.

        2. Ranking claims
           When SQL returns one dimension and one numeric metric,
           highest/lowest/ranking claims are checked against the
           actual ordered result.

        The SQL result is authoritative.
        """

        validated_result = (
            InsightVerificationWorkflow
            ._validate_scalar_numerical_evidence(
                result=result,
                sql_result=sql_result,
            )
        )

        validated_result = (
            InsightVerificationWorkflow
            ._validate_ranking_claims(
                result=validated_result,
                sql_result=sql_result,
            )
        )

        return validated_result

    @staticmethod
    def _validate_scalar_numerical_evidence(
        result: InsightAnalysis,
        sql_result: SQLResult,
    ) -> InsightAnalysis:
        """
        Validate numerical evidence for scalar SQL results.

        For a one-row SQL result, only numerical values actually
        returned by SQL may be presented as directly supported
        evidence.

        Example:

            SQL result:
                ratio = 0.0657

        Supported:
            "The ratio is 0.0657."

        Unsupported:
            "EUR customers = 3206."

        because 3206 was not returned by the SQL query.

        Row-index numbers such as row[0][0] are ignored.
        """

        if len(sql_result.rows) != 1:
            return result

        if not sql_result.rows:
            return result

        sql_values = sql_result.rows[0]

        numeric_sql_values: list[float] = []

        for value in sql_values:
            numeric_value = (
                InsightVerificationWorkflow
                ._to_numeric(value)
            )

            if numeric_value is not None:
                numeric_sql_values.append(
                    numeric_value
                )

        if not numeric_sql_values:
            return result

        invalid_claims: list[str] = []

        for insight in result.insights:
            for evidence in insight.evidence:
                supporting_data = (
                    evidence.supporting_data or ""
                )

                claimed_numbers = (
                    InsightVerificationWorkflow
                    ._extract_numeric_claims(
                        supporting_data
                    )
                )

                if not claimed_numbers:
                    continue

                for claimed_number in claimed_numbers:
                    if not InsightVerificationWorkflow._matches_sql_value(
                        claimed_number,
                        numeric_sql_values,
                    ):
                        invalid_claims.append(
                            insight.statement
                        )
                        break

        if not invalid_claims:
            return result

        return (
            InsightVerificationWorkflow
            ._mark_claims_unsupported(
                result=result,
                invalid_claims=invalid_claims,
            )
        )

    @staticmethod
    def _extract_numeric_claims(
        text: str,
    ) -> list[float]:
        """
        Extract numerical claims from evidence text.

        Numbers used only as row/column indexes, such as:

            row[0][1]

        are ignored.

        Commas in numbers are supported:

            3,206 -> 3206
        """

        if not text:
            return []

        cleaned = re.sub(
            r"row\s*\[\s*\d+\s*\]"
            r"(?:\s*\[\s*\d+\s*\])*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        matches = re.findall(
            r"(?<![A-Za-z0-9_])"
            r"[-+]?"
            r"(?:\d{1,3}(?:,\d{3})+|\d+)"
            r"(?:\.\d+)?"
            r"(?:[eE][-+]?\d+)?"
            r"(?![A-Za-z0-9_])",
            cleaned,
        )

        values: list[float] = []

        for match in matches:
            normalized = match.replace(
                ",",
                "",
            )

            try:
                values.append(
                    float(normalized)
                )
            except ValueError:
                continue

        return values

    @staticmethod
    def _to_numeric(
        value,
    ) -> float | None:
        """Convert a value to a finite float when possible."""

        if isinstance(value, bool):
            return None

        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return None

        if not math.isfinite(numeric_value):
            return None

        return numeric_value

    @staticmethod
    def _matches_sql_value(
        claimed_value: float,
        sql_values: list[float],
    ) -> bool:
        """
        Determine whether a claimed number matches a SQL result value.

        Relative/absolute tolerance handles normal floating-point
        representation differences.
        """

        for sql_value in sql_values:
            if math.isclose(
                claimed_value,
                sql_value,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                return True

        return False

    @staticmethod
    def _mark_claims_unsupported(
        result: InsightAnalysis,
        invalid_claims: list[str],
    ) -> InsightAnalysis:
        """Mark deterministic unsupported claims."""

        updated_unsupported_claims = list(
            result.unsupported_claims
        )

        for claim in invalid_claims:
            if claim not in updated_unsupported_claims:
                updated_unsupported_claims.append(
                    claim
                )

        return result.model_copy(
            update={
                "unsupported_claims":
                    updated_unsupported_claims,
                "verification_status": (
                    "partially_verified"
                    if result.insights
                    else "insufficient_evidence"
                ),
            }
        )

    @staticmethod
    def _validate_ranking_claims(
        result: InsightAnalysis,
        sql_result: SQLResult,
    ) -> InsightAnalysis:
        """
        Deterministically verify ranking claims against SQL results.

        This protects Agent 2 from incorrect LLM-generated statements
        such as claiming that the fourth-highest value is actually
        the second-highest value.

        The method only performs deterministic ranking checks when
        the SQL result has exactly one dimension column and one
        numeric metric column.
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
                numeric_value = float(
                    metric_value
                )
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
                    InsightVerificationWorkflow
                    ._dimension_for_rank(
                        descending=descending,
                        ascending=ascending,
                        rank=rank,
                    )
                )

                if expected_dimension is None:
                    continue

                claimed_dimension = (
                    InsightVerificationWorkflow
                    ._extract_dimension(
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
                    invalid_claims.append(
                        statement
                    )

        if not invalid_claims:
            return result

        return (
            InsightVerificationWorkflow
            ._mark_claims_unsupported(
                result=result,
                invalid_claims=invalid_claims,
            )
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

        before_phrase = (
            statement[:phrase_index]
            .strip()
        )

        # Prefer a known dimension appearing immediately before the
        # ranking phrase.
        for dimension in sorted(
            known_dimensions,
            key=len,
            reverse=True,
        ):
            dimension_lower = (
                dimension.lower()
            )

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
            candidate = (
                match.group(1)
                .strip()
            )

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