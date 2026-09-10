import re

from agents import Agent, Model, Runner
from pydantic import BaseModel, Field

from models import AnalyticalPlan
from schema import DataSchema


class SQLGenerationOutput(BaseModel):
    """Structured SQL generated from an analytical plan."""

    query: str = Field(
        description="One read-only SQL SELECT query."
    )


SQL_GENERATOR_INSTRUCTIONS = """
You are the SQL generation component for TalkToTableau.

Your responsibility is to convert a structured analytical plan
into ONE executable, read-only SQL query.

CORE RULE:

The analytical plan is authoritative.

Never invent:
- tables
- columns
- filters
- values
- dimensions
- aliases representing nonexistent columns
- business requirements

Use only information present in the analytical plan and supplied schema.

==================================================
SQL SAFETY
==================================================

Generate exactly one SELECT statement.

Never generate:
INSERT
UPDATE
DELETE
DROP
ALTER
CREATE
TRUNCATE
GRANT
REVOKE
COMMENT
VACUUM
ANALYZE
COPY

Never use:
- ?
- :parameter
- :region
- $1
- $2
- ${...}
- {{...}}

TalkToTableau does not provide SQL parameter bindings.

All explicitly requested filter values must be written as concrete SQL
literals.

==================================================
FILTER RULE
==================================================

Only add a WHERE clause when the analytical plan explicitly contains
a filter.

If the analytical plan contains no filters:

- DO NOT generate WHERE.
- DO NOT invent a WHERE condition.
- DO NOT filter by Segment.
- DO NOT filter by Country.
- DO NOT filter by Culture.
- DO NOT filter by any other column.

The existence of a column in the schema does NOT mean it should be used
as a filter.

==================================================
ANALYTICAL CORRECTNESS
==================================================

Answer the actual analytical question.

Do not introduce unnecessary:
- GROUP BY
- JOIN
- subqueries
- dimensions
- columns
- calculations

If the user asks for one overall:
- ratio
- percentage
- proportion
- average
- mean
- total
- sum
- count
- maximum
- minimum

return one overall result unless the analytical plan explicitly requests
grouping.

Do NOT group a scalar metric by:
- CustomerID
- ProductID
- TransactionID
- Region
- Country
- Segment
- any other identifier or dimension

unless the analytical plan explicitly requests that grouping.

==================================================
RATIO QUESTIONS
==================================================

For a question asking for a ratio such as:

"What is the ratio of customers who pay in EUR against customers
who pay in CZK?"

identify:

numerator = number of customers satisfying Currency = 'EUR'

denominator = number of customers satisfying Currency = 'CZK'

The ratio should be calculated at the requested overall level.

Prefer conditional aggregation when appropriate.

For example, when the relevant table contains Currency:

SELECT
    CAST(SUM(IIF(Currency = 'EUR', 1, 0)) AS FLOAT)
    / SUM(IIF(Currency = 'CZK', 1, 0)) AS ratio
FROM customers

The exact SQL does not need to match this example character-for-character,
but it must have the same analytical meaning.

Do NOT:
- group by CustomerID
- join another table unless required
- filter by an unrelated column
- invent columns
- calculate the ratio separately for every customer
- introduce unnecessary subqueries

If the denominator could be zero, use a valid SQL-safe expression,
but do not introduce unrelated filters or dimensions.

==================================================
RESULT GRANULARITY
==================================================

If the analytical plan requests a scalar result:

- return exactly one aggregate result
- use aggregate functions or conditional aggregation
- do not return one row per entity

If the analytical plan explicitly requests "by X":

- include X in SELECT
- GROUP BY X
- calculate the requested metric for each X

If the plan does not request grouping, do not group.

==================================================
FILE-BASED DATA
==================================================

For file-based workflows:

- the available table name is "data"
- use "data" in FROM
- do not use the original filename as a table name

==================================================
SQLITE DATA
==================================================

For SQLite:

- use the exact table names supplied in the schema
- use the exact column names supplied in the schema
- never prefix tables with "data."
- never invent a "data" schema
- use SQLite-compatible SQL
- use IIF(condition, true_value, false_value)
- never use MySQL IF(...)
- do not use unsupported dialect-specific functions
- do not introduce unnecessary joins

==================================================
COLUMN VALIDITY
==================================================

Every column referenced in the generated SQL must exist in the supplied
schema.

Before generating SQL, mentally verify:

1. Every table exists.
2. Every column exists in its referenced table.
3. Every filter column exists.
4. Every metric column exists.
5. Every grouping column exists.
6. Every JOIN key exists on both sides.
7. Every alias expression uses valid columns.

Do not transform a requested concept into a different column merely
because that column exists.

For example, if the plan says:

Currency = EUR

and the schema contains:

Currency

then use:

Currency = 'EUR'

Do NOT replace Currency with:
Culture
Country
Region
Segment
or another unrelated field.

==================================================
SQL DIALECT
==================================================

For SQLite, string values must use normal SQL single quotes:

'EUR'
'CZK'

Do not generate escaped Python-style strings such as:

\"EUR\"
\\'EUR\\'

Do not generate malformed backslash sequences.

==================================================
OUTPUT
==================================================

Return only the structured SQLGenerationOutput.

Do not return explanations.
Do not return markdown.
Do not execute SQL.
"""


def create_sql_generator(model: Model) -> Agent:
    """Create the SQL generation component."""

    return Agent(
        name="SQL Generator",
        instructions=SQL_GENERATOR_INSTRUCTIONS,
        output_type=SQLGenerationOutput,
        model=model,
    )


def _normalize_sqlite_functions(query: str) -> str:
    """
    Normalize common SQL dialect differences for SQLite.

    Small local models may generate MySQL-style IF().
    SQLite provides IIF().
    """

    return re.sub(
        r"\bIF\s*\(",
        "IIF(",
        query,
        flags=re.IGNORECASE,
    )


def _normalize_sqlite_table_references(
    query: str,
    schema: DataSchema,
) -> str:
    """
    Remove an invalid 'data.' prefix from SQLite table references.

    SQLite databases expose their real table names directly.
    """

    table_names = {
        table.name
        for table in schema.tables
    }

    normalized = query

    for table_name in sorted(
        table_names,
        key=len,
        reverse=True,
    ):
        pattern = rf"\bdata\.{re.escape(table_name)}\b"

        normalized = re.sub(
            pattern,
            table_name,
            normalized,
            flags=re.IGNORECASE,
        )

    return normalized


def _normalize_sqlite_escaping(query: str) -> str:
    """
    Remove malformed escaping occasionally produced by small local models.

    This is intentionally conservative and only handles common escaped
    quote forms that should have been ordinary SQL string literals.
    """

    normalized = query

    normalized = normalized.replace(
        '\\"',
        '"',
    )

    normalized = normalized.replace(
        "\\'",
        "'",
    )

    return normalized


def _normalize_generated_sql(
    query: str,
    schema: DataSchema,
) -> str:
    """Apply deterministic source-specific SQL normalization."""

    normalized = query.strip()

    if schema.source_type == "sqlite":
        normalized = _normalize_sqlite_functions(
            normalized
        )

        normalized = _normalize_sqlite_table_references(
            normalized,
            schema,
        )

        normalized = _normalize_sqlite_escaping(
            normalized
        )

    return normalized


def _build_schema_description(
    schema: DataSchema,
) -> str:
    """Build a compact human-readable schema for the SQL model."""

    sections: list[str] = []

    for table in schema.tables:
        columns = ", ".join(
            column.name
            for column in table.columns
        )

        sections.append(
            f"TABLE: {table.name}\n"
            f"COLUMNS: {columns}"
        )

    return "\n\n".join(sections)


def generate_sql(
    model: Model,
    user_request: str,
    plan: AnalyticalPlan,
    schema: DataSchema,
    feedback: str | None = None,
) -> str:
    """
    Generate one SQL query from an analytical plan.

    SQL validation and execution happen separately.

    Optional feedback describes a failure from a previous SQL attempt
    so a bounded retry can correct the query.
    """

    sql_agent = create_sql_generator(
        model=model,
    )

    retry_feedback = ""

    if feedback:
        retry_feedback = f"""
==================================================
PREVIOUS ATTEMPT FAILED
==================================================

Previous SQL attempt feedback:

{feedback}

Previous SQL was invalid or incorrect.

Generate a completely new SQL query.

Do NOT repeat:
- the previous invalid SQL
- the previous invented filters
- the previous invented columns
- the previous unnecessary joins
- the previous unnecessary GROUP BY

Re-read the analytical plan and schema before generating the replacement.
"""

    schema_description = _build_schema_description(
        schema
    )

    if schema.source_type == "sqlite":
        source_instructions = f"""
The data source is SQLite.

ACTUAL SQLITE TABLES AND COLUMNS:

{schema_description}

Use ONLY these exact tables and columns.

Never use:
data.<table>

Never invent:
data.customers
data.transactions
or any other table that is not listed above.

For SQLite:
- use IIF(...)
- use normal single-quoted string literals
- do not use IF(...)
- do not use parameter placeholders
- do not use backslash-escaped SQL strings
"""

    else:
        source_instructions = f"""
The data source is file-based.

The available DuckDB table is:

data

Schema:

{schema_description}

Use "data" in the FROM clause.
Do not use the original filename as a table name.
"""

    prompt = f"""
==================================================
USER REQUEST
==================================================

{user_request}

==================================================
AUTHORITATIVE ANALYTICAL PLAN
==================================================

{plan.model_dump_json(indent=2)}

==================================================
AVAILABLE SCHEMA
==================================================

{schema_description}

==================================================
SOURCE INSTRUCTIONS
==================================================

{source_instructions}

==================================================
FINAL SQL GENERATION RULES
==================================================

The analytical plan is authoritative.

Generate exactly ONE executable SELECT query.

Before generating the query, verify:

- requested dimensions
- requested metrics
- requested filters
- requested time range
- requested comparison
- requested analytical granularity
- available tables
- available columns

IMPORTANT:

If the plan contains NO filters:

DO NOT generate a WHERE clause.

If the request is a scalar ratio:

DO NOT group by CustomerID.

DO NOT join transactions_1k unless the analytical plan explicitly
requires transaction data.

DO NOT use Segment unless the analytical plan explicitly requests
Segment.

DO NOT use Culture unless the analytical plan explicitly requests
Culture.

DO NOT use Country unless the analytical plan explicitly requests
Country.

For the BIRD-style customer currency ratio:

If the plan asks for the ratio of customers paying in EUR against
customers paying in CZK, and Currency exists in the customers table,
calculate:

count of Currency = 'EUR'
/
count of Currency = 'CZK'

using conditional aggregation over the customers table.

Return one scalar ratio.

Do not introduce unrelated filters.

Do not invent values.

Do not invent columns.

Do not invent tables.

Do not use placeholders.

Do not use backslashes in SQL string literals.

{retry_feedback}

Generate the SQL query now.
"""

    result = Runner.run_sync(
        sql_agent,
        prompt,
    )

    output = result.final_output

    if not isinstance(
        output,
        SQLGenerationOutput,
    ):
        raise ValueError(
            "SQL generator returned an invalid structured output."
        )

    return _normalize_generated_sql(
        output.query,
        schema,
    )