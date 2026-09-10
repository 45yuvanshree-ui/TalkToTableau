import re

from models import AnalyticalPlan

from .models import SQLValidationResult


FORBIDDEN_STATEMENTS = {
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "ALTER",
    "TRUNCATE",
    "CREATE",
    "GRANT",
    "REVOKE",
    "COMMENT",
    "VACUUM",
    "ANALYZE",
    "COPY",
}

AGGREGATE_FUNCTIONS = {
    "SUM",
    "COUNT",
    "AVG",
    "MIN",
    "MAX",
}


def _find_nested_aggregate(query: str) -> str | None:
    """
    Detect directly nested aggregate functions such as:

        SUM(COUNT(...))
        COUNT(SUM(...))
        AVG(SUM(...))

    These are invalid in the scalar aggregation context used by
    TalkToTableau and commonly indicate an incorrect Text-to-SQL
    generation.

    Valid expressions such as:

        SUM(IIF(...))
        COUNT(CASE WHEN ... END)

    are not rejected because their inner expression is not itself
    an aggregate function.
    """

    aggregate_pattern = "|".join(
        sorted(
            AGGREGATE_FUNCTIONS,
            key=len,
            reverse=True,
        )
    )

    pattern = (
        rf"\b(?:{aggregate_pattern})"
        rf"\s*\(\s*"
        rf"(?:DISTINCT\s+)?"
        rf"(?:{aggregate_pattern})"
        rf"\s*\("
    )

    match = re.search(
        pattern,
        query,
        re.IGNORECASE,
    )

    if not match:
        return None

    return match.group(0)


def validate_sql(query: str) -> SQLValidationResult:
    """Validate that SQL is a single read-only SELECT statement."""

    if not query or not query.strip():
        return SQLValidationResult(
            is_valid=False,
            normalized_query="",
            reason="SQL query is empty.",
        )

    normalized = query.strip().rstrip(";").strip()

    if not normalized:
        return SQLValidationResult(
            is_valid=False,
            normalized_query="",
            reason="SQL query is empty.",
        )

    if ";" in normalized:
        return SQLValidationResult(
            is_valid=False,
            normalized_query=normalized,
            reason="Multiple SQL statements are not allowed.",
        )

    without_comments = re.sub(
        r"--.*?$",
        "",
        normalized,
        flags=re.MULTILINE,
    )

    without_comments = re.sub(
        r"/\*.*?\*/",
        "",
        without_comments,
        flags=re.DOTALL,
    ).strip()

    if not without_comments:
        return SQLValidationResult(
            is_valid=False,
            normalized_query=normalized,
            reason="SQL query contains no executable statement.",
        )

    if not re.match(
        r"^SELECT\b",
        without_comments,
        re.IGNORECASE,
    ):
        return SQLValidationResult(
            is_valid=False,
            normalized_query=normalized,
            reason="Only SELECT statements are allowed.",
        )

    for statement in FORBIDDEN_STATEMENTS:
        pattern = rf"\b{statement}\b"

        if re.search(
            pattern,
            without_comments,
            re.IGNORECASE,
        ):
            return SQLValidationResult(
                is_valid=False,
                normalized_query=normalized,
                reason=(
                    f"Forbidden SQL statement detected: "
                    f"{statement}."
                ),
            )

    # TalkToTableau does not currently provide SQL parameter bindings.
    # Therefore generated SQL must contain concrete values rather than
    # placeholders such as ?, :region, $1, or {{region}}.
    if "?" in without_comments:
        return SQLValidationResult(
            is_valid=False,
            normalized_query=normalized,
            reason=(
                "SQL contains an unbound '?' parameter placeholder. "
                "TalkToTableau does not provide SQL parameter bindings. "
                "Generate concrete SQL values instead. Do not use '?'."
            ),
        )

    if re.search(
        r"(?<!:):[A-Za-z_][A-Za-z0-9_]*",
        without_comments,
    ):
        return SQLValidationResult(
            is_valid=False,
            normalized_query=normalized,
            reason=(
                "SQL contains an unbound named parameter placeholder. "
                "TalkToTableau does not provide SQL parameter bindings. "
                "Generate concrete SQL values instead. "
                "Do not use placeholders such as :region or :value."
            ),
        )

    if re.search(
        r"\$\d+",
        without_comments,
    ):
        return SQLValidationResult(
            is_valid=False,
            normalized_query=normalized,
            reason=(
                "SQL contains an unbound positional parameter placeholder. "
                "TalkToTableau does not provide SQL parameter bindings. "
                "Generate concrete SQL values instead. "
                "Do not use placeholders such as $1 or $2."
            ),
        )

    if re.search(
        r"\$\{[^}]+\}",
        without_comments,
    ):
        return SQLValidationResult(
            is_valid=False,
            normalized_query=normalized,
            reason=(
                "SQL contains a template variable placeholder. "
                "Generate concrete SQL values instead. "
                "Do not use ${...}."
            ),
        )

    if re.search(
        r"\{\{[^}]+\}\}",
        without_comments,
    ):
        return SQLValidationResult(
            is_valid=False,
            normalized_query=normalized,
            reason=(
                "SQL contains a template variable placeholder. "
                "Generate concrete SQL values instead. "
                "Do not use {{...}}."
            ),
        )

    # Detect invalid nested aggregate functions.
    nested_aggregate = _find_nested_aggregate(
        without_comments
    )

    if nested_aggregate:
        return SQLValidationResult(
            is_valid=False,
            normalized_query=normalized,
            reason=(
                "SQL contains nested aggregate functions: "
                f"{nested_aggregate}. "
                "Nested aggregates such as SUM(COUNT(...)), "
                "COUNT(SUM(...)), or AVG(SUM(...)) are not valid "
                "for this analytical workflow. "
                "For scalar ratios or conditional counts, use "
                "conditional aggregation directly, for example "
                "SUM(IIF(condition, 1, 0)) or "
                "COUNT(CASE WHEN condition THEN 1 END). "
                "Do not wrap one aggregate function inside another."
            ),
        )

    return SQLValidationResult(
        is_valid=True,
        normalized_query=normalized,
    )


def validate_sql_against_plan(
    query: str,
    plan: AnalyticalPlan,
) -> SQLValidationResult:
    """
    Validate generated SQL against the analytical plan.

    A WHERE clause is permitted only when the analytical plan
    explicitly contains filters.
    """

    validation = validate_sql(query)

    if not validation.is_valid:
        return validation

    normalized_query = validation.normalized_query

    has_where_clause = bool(
        re.search(
            r"\bWHERE\b",
            normalized_query,
            re.IGNORECASE,
        )
    )

    filters = plan.filters or []

    if not filters and has_where_clause:
        return SQLValidationResult(
            is_valid=False,
            normalized_query=normalized_query,
            reason=(
                "SQL validation failed: the analytical plan "
                "contains NO filters, but the generated SQL "
                "contains a WHERE clause. Remove the WHERE "
                "clause completely. Do not invent filtering "
                "conditions. For a scalar metric or ratio, "
                "calculate the requested metric directly at "
                "the requested aggregation level."
            ),
        )

    return SQLValidationResult(
        is_valid=True,
        normalized_query=normalized_query,
    )