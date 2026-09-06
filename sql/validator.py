import re

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

    # Remove a trailing semicolon for consistent processing.
    normalized = normalized.rstrip(";").strip()

    if not normalized:
        return SQLValidationResult(
            is_valid=False,
            normalized_query="",
            reason="SQL query is empty.",
        )

    # Reject multiple statements.
    if ";" in normalized:
        return SQLValidationResult(
            is_valid=False,
            normalized_query=normalized,
            reason="Multiple SQL statements are not allowed.",
        )

    # Remove SQL comments before checking the statement.
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

    # V1 only permits SELECT statements.
    if not re.match(r"^SELECT\b", without_comments, re.IGNORECASE):
        return SQLValidationResult(
            is_valid=False,
            normalized_query=normalized,
            reason="Only SELECT statements are allowed.",
        )

    # Reject dangerous SQL keywords anywhere in the query.
    for statement in FORBIDDEN_STATEMENTS:
        pattern = rf"\b{statement}\b"

        if re.search(pattern, without_comments, re.IGNORECASE):
            return SQLValidationResult(
                is_valid=False,
                normalized_query=normalized,
                reason=f"Forbidden SQL statement detected: {statement}.",
            )

    return SQLValidationResult(
        is_valid=True,
        normalized_query=normalized,
    )