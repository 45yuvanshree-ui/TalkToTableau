"""Output guardrails and deterministic factual validation."""

import re

from models import BusinessStory, InsightAnalysis
from sql.models import SQLResult


def validate_insight_output(
    result: InsightAnalysis,
) -> tuple[bool, str | None]:
    """Validate the structural integrity of Agent 2 output."""

    if result.verification_status not in {
        "verified",
        "partially_verified",
        "insufficient_evidence",
    }:
        return False, "Invalid insight verification status."

    for insight in result.insights:
        if insight.confidence not in {
            "high",
            "medium",
            "low",
        }:
            return False, "Invalid insight confidence level."

        for evidence in insight.evidence:
            if not evidence.is_supported:
                return False, "Insight contains unsupported evidence."

    if (
        result.unsupported_claims
        and result.verification_status == "verified"
    ):
        return (
            False,
            "Output cannot be fully verified when unsupported "
            "claims are present.",
        )

    return True, None


def validate_story_output(
    story: BusinessStory,
    insight_analysis: InsightAnalysis | None = None,
) -> tuple[bool, str | None]:
    """Validate the structural and evidence-related story output."""

    if not story.summary.strip():
        return False, "Story summary is empty."

    if not story.sections:
        return False, "Story must contain at least one section."

    for section in story.sections:
        if not section.heading.strip():
            return False, "Story section heading is empty."

        if not section.content.strip():
            return False, "Story section content is empty."

    if not story.visualization_explanation.strip():
        return False, "Visualization explanation must not be empty."

    if (
        insight_analysis is not None
        and insight_analysis.verification_status
        == "insufficient_evidence"
        and story.recommendations
    ):
        return (
            False,
            "Recommendations are not allowed when there is "
            "insufficient evidence.",
        )

    return True, None


def _extract_story_numbers(text: str) -> list[float]:
    """Extract explicit numeric values from story text."""

    text = re.sub(r"(?<=\d),(?=\d)", "", text)

    number_pattern = re.compile(
        r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])"
    )

    return [
        float(match.group())
        for match in number_pattern.finditer(text)
    ]


def _numeric_values_from_sql(
    sql_result: SQLResult,
) -> list[float]:
    """
    Return all authoritative numeric values associated with
    the SQL result.

    Numeric values may come from:
    1. Numeric values contained in SQL result rows.
    2. The SQL result row_count metadata.

    row_count is included because it is an authoritative property
    of the executed result set and may legitimately be referenced
    by a business story, especially for retrieval/list queries
    whose returned values are identifiers represented as strings.
    """

    values: list[float] = []

    for row in sql_result.rows:
        for value in row:
            if isinstance(value, bool):
                continue

            if isinstance(value, (int, float)):
                values.append(float(value))

    if sql_result.row_count is not None:
        values.append(
            float(sql_result.row_count)
        )

    return values


def _normalise_text(value: str) -> str:
    """Normalize text for deterministic comparison."""

    value = value.strip().lower()

    value = re.sub(
        r"[^a-z0-9_]+",
        " ",
        value,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value.strip()


def _extract_ranking_claims(
    text: str,
    dimensions: list[str],
) -> list[tuple[str, int, str]]:
    """
    Extract ranking claims from story text.

    Returns:
        (normalized_dimension, rank, direction)

    Examples:
        "North has the highest revenue"
            -> ("north", 1, "highest")

        "South has the third-highest revenue"
            -> ("south", 3, "highest")

        "West has the second-lowest revenue"
            -> ("west", 2, "lowest")
    """

    claims: list[tuple[str, int, str]] = []

    if not dimensions:
        return claims

    normalized_dimensions = {
        _normalise_text(dimension): dimension
        for dimension in dimensions
    }

    rank_words = {
        "first": 1,
        "second": 2,
        "third": 3,
        "fourth": 4,
        "fifth": 5,
    }

    # Match the common factual-story structure:
    #
    #   North has the highest revenue
    #   West has the lowest revenue
    #   South has the third-highest revenue
    #
    # The dimension is captured directly before "has", making
    # the validation deterministic rather than dependent on
    # sentence parsing heuristics.

    dimension_pattern = "|".join(
        re.escape(name)
        for name in sorted(
            normalized_dimensions.keys(),
            key=len,
            reverse=True,
        )
    )

    if not dimension_pattern:
        return claims

    ranking_pattern = re.compile(
        rf"""
        (?P<dimension>\b(?:{dimension_pattern})\b)
        \s+
        (?:has|is|was)
        \s+
        (?:the\s+)?
        (?:
            (?P<ordinal>first|second|third|fourth|fifth)
            [-\s]+
            (?P<ordinal_direction>highest|lowest)
            |
            (?P<extreme>highest|lowest|largest|smallest|top|bottom)
        )
        \b
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    for match in ranking_pattern.finditer(text):
        claimed_dimension = _normalise_text(
            match.group("dimension")
        )

        ordinal = match.group("ordinal")
        ordinal_direction = match.group(
            "ordinal_direction"
        )
        extreme = match.group("extreme")

        if ordinal is not None:
            rank = rank_words[ordinal.lower()]
            direction = ordinal_direction.lower()
        else:
            rank = 1

            if extreme.lower() in {
                "highest",
                "largest",
                "top",
            }:
                direction = "highest"
            else:
                direction = "lowest"

        claims.append(
            (
                claimed_dimension,
                rank,
                direction,
            )
        )

    return claims


def _validate_extreme_claims(
    text: str,
    sql_result: SQLResult,
) -> tuple[bool, str | None]:
    """
    Validate highest/lowest and ordinal ranking claims
    against SQL results.
    """

    if len(sql_result.columns) != 2:
        return True, None

    dimension_index = 0
    metric_index = 1

    pairs: list[tuple[str, float]] = []

    for row in sql_result.rows:
        if len(row) <= max(
            dimension_index,
            metric_index,
        ):
            continue

        dimension_value = row[dimension_index]
        metric_value = row[metric_index]

        if dimension_value is None:
            continue

        if isinstance(metric_value, bool):
            continue

        if not isinstance(metric_value, (int, float)):
            continue

        pairs.append(
            (
                str(dimension_value),
                float(metric_value),
            )
        )

    if not pairs:
        return True, None

    descending = sorted(
        pairs,
        key=lambda item: item[1],
        reverse=True,
    )

    ascending = sorted(
        pairs,
        key=lambda item: item[1],
    )

    dimensions = [
        name
        for name, _ in pairs
    ]

    ranking_claims = _extract_ranking_claims(
        text=text,
        dimensions=dimensions,
    )

    for claimed_dimension, rank, direction in ranking_claims:
        ordered_pairs = (
            descending
            if direction == "highest"
            else ascending
        )

        if rank > len(ordered_pairs):
            continue

        expected_value = ordered_pairs[rank - 1][1]

        expected_dimensions = {
            _normalise_text(name)
            for name, value in ordered_pairs
            if value == expected_value
        }

        if claimed_dimension not in expected_dimensions:
            expected_names = [
                name
                for name, value in ordered_pairs
                if value == expected_value
            ]

            rank_labels = {
                ("highest", 1): "highest",
                ("highest", 2): "second-highest",
                ("highest", 3): "third-highest",
                ("highest", 4): "fourth-highest",
                ("highest", 5): "fifth-highest",
                ("lowest", 1): "lowest",
                ("lowest", 2): "second-lowest",
                ("lowest", 3): "third-lowest",
                ("lowest", 4): "fourth-lowest",
                ("lowest", 5): "fifth-lowest",
            }

            rank_label = rank_labels.get(
                (direction, rank),
                f"rank {rank}",
            )

            return (
                False,
                "Story incorrectly identifies "
                f"'{claimed_dimension}' as the {rank_label} value. "
                f"Expected: {expected_names}.",
            )

    return True, None


def validate_story_factual_claims(
    story: BusinessStory,
    sql_result: SQLResult,
    insight_analysis: InsightAnalysis | None = None,
) -> tuple[bool, str | None]:
    """
    Validate factual numerical and ranking claims in a business story.

    Numerical claims are considered supported when they are present in:
    1. the authoritative SQL result, or
    2. explicitly supported evidence produced by Agent 2.

    Agent 2 evidence is only trusted when:
        evidence.is_supported == True

    Ranking claims remain deterministically validated against the
    SQL result itself.
    """

    story_text_parts = [
        story.summary,
        story.visualization_explanation,
    ]

    story_text_parts.extend(
        section.content
        for section in story.sections
    )

    story_text_parts.extend(
        story.recommendations
    )

    story_text = "\n".join(
        story_text_parts
    )

    # --------------------------------------------------------------
    # SQL NUMERICAL EVIDENCE
    # --------------------------------------------------------------

    sql_numbers = _numeric_values_from_sql(
        sql_result
    )

    supported_numbers = {
        round(value, 10)
        for value in sql_numbers
    }

    # --------------------------------------------------------------
    # AGENT 2 VERIFIED NUMERICAL EVIDENCE
    # --------------------------------------------------------------

    if insight_analysis is not None:
        for insight in insight_analysis.insights:
            for evidence in insight.evidence:

                if not evidence.is_supported:
                    continue

                evidence_text_parts = [
                    evidence.claim,
                    evidence.supporting_data,
                ]

                evidence_text = "\n".join(
                    part
                    for part in evidence_text_parts
                    if part
                )

                for number in _extract_story_numbers(
                    evidence_text
                ):
                    supported_numbers.add(
                        round(number, 10)
                    )

    # --------------------------------------------------------------
    # STORY NUMERICAL VALIDATION
    # --------------------------------------------------------------

    story_numbers = _extract_story_numbers(
        story_text
    )

    for number in story_numbers:

        if round(number, 10) not in supported_numbers:

            return (
                False,
                "Story contains unsupported numerical value: "
                f"{number:g}.",
            )

    # --------------------------------------------------------------
    # RANKING VALIDATION
    # --------------------------------------------------------------

    is_valid, reason = _validate_extreme_claims(
        text=story_text,
        sql_result=sql_result,
    )

    if not is_valid:
        return False, reason

    return True, None