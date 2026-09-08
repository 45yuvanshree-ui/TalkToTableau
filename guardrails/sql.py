import re

from schema import DataSchema
from sql import validate_sql


SQL_IDENTIFIER_PATTERN = re.compile(
    r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_.]*)",
    re.IGNORECASE,
)


def validate_sql_against_schema(
    query: str,
    schema: DataSchema,
) -> tuple[bool, str | None]:
    """
    Validate read-only SQL and ensure referenced tables exist
    in the supplied schema.
    """

    base_validation = validate_sql(query)

    if not base_validation.is_valid:
        return False, base_validation.reason

    available_tables = {
        table.name.lower()
        for table in schema.tables
    }

    referenced_tables = {
        match.lower()
        for match in SQL_IDENTIFIER_PATTERN.findall(
            base_validation.normalized_query
        )
    }

    for table in referenced_tables:
        if table not in available_tables:
            return (
                False,
                f"Referenced table is not present in schema: {table}",
            )

    return True, None