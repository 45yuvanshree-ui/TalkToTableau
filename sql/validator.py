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


def validate_sql(query: str) -> SQLValidationResult:
    """
    Validate SQL before execution.

    V1 allows read-only SELECT queries only.
    """

    if not query or not query.strip():
        return SQLValidationResult(
            is_valid=False,
            normalized_query="",
            reason="SQL query is empty.",
        )

    normalized = query.strip()

    normalized = normalized.rstrip(";").strip()

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
                    "Forbidden SQL statement detected: "
                    f"{statement}."
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

    This prevents the SQL generator from introducing analytical
    constraints that were not requested.

    V1 specifically protects against hallucinated WHERE filters.
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
                "Generated SQL contains a WHERE clause, "
                "but the analytical plan contains no filters."
            ),
        )

    return SQLValidationResult(
        is_valid=True,
        normalized_query=normalized_query,
    )