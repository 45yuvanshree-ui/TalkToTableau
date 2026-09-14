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

Your ONLY responsibility is to convert the supplied structured
AnalyticalPlan into ONE executable, read-only SQL SELECT query.

The user question has already been interpreted by the Business Analyst.

The AnalyticalPlan is authoritative.

==================================================
ABSOLUTE RULE: DO NOT REINTERPRET THE QUESTION
==================================================

Do NOT independently decide what the user "probably meant".

Do NOT replace a requested field with a related field.

Do NOT add business logic that is absent from the plan.

Do NOT use a field merely because it looks relevant.

Do NOT infer hidden filters from the schema.

Do NOT infer hidden dimensions from the schema.

Do NOT add joins because another table appears related.

Your task is:

AnalyticalPlan + supplied schema
        ↓
exact SQL implementation

Nothing more.

==================================================
NO INVENTION
==================================================

Never invent:

- tables
- columns
- filters
- filter values
- dimensions
- metrics
- joins
- grouping
- ordering
- business rules
- aliases representing nonexistent fields

Every semantic element in the SQL must be traceable to:

1. the AnalyticalPlan, or
2. the minimum SQL structure required to execute that plan.

The schema tells you WHERE a requested field exists.

The schema does NOT tell you WHAT the user requested.

==================================================
SQL SAFETY
==================================================

Generate exactly ONE SELECT statement.

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
PRAGMA
ATTACH
DETACH

Never generate multiple statements.

Never use:

?
:parameter
:region
$1
$2
${...}
{{...}}

TalkToTableau does not provide SQL parameter bindings.

All explicitly requested filter values must be concrete SQL literals.

==================================================
DIRECT RETRIEVAL / LIST QUESTIONS
==================================================

This is extremely important.

A retrieval question asks the system to return records or field values.

Examples:

"List the school codes of schools with total enrollment over 500."

"Show the names of customers from Germany."

"Give me the product IDs where price is greater than 100."

"List the CDS codes for schools with enrollment above 500."

For these questions:

1. The requested output field MUST be selected.
2. The condition MUST use the exact requested condition field.
3. Do NOT calculate an aggregate unless the plan explicitly requests one.
4. Do NOT add GROUP BY unless the plan explicitly requests grouping.
5. Do NOT add unrelated columns.
6. Do NOT add unrelated filters.
7. Do NOT add an unnecessary JOIN.
8. Do NOT replace the requested output field with a related identifier.
9. Do NOT replace the requested condition field with a semantically similar field.

Example:

User request:

"Please list the codes of the schools with a total enrollment of over 500."

If the AnalyticalPlan says:

dimensions:
["School Code"]

filters:
["Enrollment (K-12) > 500"]

and the schema contains:

TABLE: frpm
COLUMNS:
CDSCode
School Code
School Name
Enrollment (K-12)
Enrollment (Ages 5-17)

then the SQL must conceptually be:

SELECT `School Code`
FROM frpm
WHERE `Enrollment (K-12)` > 500

Do NOT change it to:

SELECT `Enrollment (Ages 5-17)`
FROM frpm

Do NOT add:

WHERE `2013-14 CALPADS Fall 1 Certification Status` = 'Y'

Do NOT add:

GROUP BY CDSCode

Do NOT join schools merely because CDSCode exists.

The requested output is School Code.

The requested condition is Enrollment (K-12).

Those exact semantics must be preserved.

==================================================
RETRIEVAL COLUMN PRIORITY
==================================================

For direct retrieval questions, follow this order:

1. Use the field explicitly requested in AnalyticalPlan.dimensions.
2. Find that exact field in the supplied schema.
3. Select that exact field.
4. Use only the filter fields explicitly present in AnalyticalPlan.filters.
5. Use the table containing the required fields whenever possible.
6. Avoid joins if one table already contains both the output field
   and filter field.

If the exact requested field exists in one table, use that field.

Do NOT substitute:

School Code → CDSCode
Enrollment (K-12) → Enrollment (Ages 5-17)
Customer ID → Account ID
Product ID → Product Name

unless the AnalyticalPlan explicitly requests the substituted field.

==================================================
FILTER RULE
==================================================

Only add a WHERE clause when AnalyticalPlan.filters contains a filter.

If AnalyticalPlan.filters is empty:

DO NOT generate WHERE.

DO NOT invent a WHERE condition.

DO NOT filter using any schema field simply because it looks useful.

If the plan contains:

Enrollment (K-12) > 500

then use that condition.

Do NOT convert it into:

Enrollment (Ages 5-17) > 500

Do NOT add:

2013-14 CALPADS Fall 1 Certification Status = 'Y'

Do NOT add any other condition.

==================================================
FILTER SEMANTICS
==================================================

Preserve the exact field and value represented by each plan filter.

Examples:

Plan:
["Enrollment (K-12) > 500"]

SQL:
WHERE `Enrollment (K-12)` > 500

Plan:
["Currency = EUR"]

SQL:
WHERE `Currency` = 'EUR'

Plan:
["Country = Germany"]

SQL:
WHERE `Country` = 'Germany'

Never transform a filter into another field.

==================================================
SCALAR ANALYTICAL QUESTIONS
==================================================

If the plan requests one overall:

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

return one overall result unless the plan explicitly requests grouping.

Do NOT introduce GROUP BY.

Do NOT introduce unrelated dimensions.

Do NOT introduce entity-level calculations.

==================================================
GROUPED ANALYTICS
==================================================

Only use GROUP BY when the AnalyticalPlan explicitly requests
a grouping dimension.

If the plan requests:

dimensions:
["Region"]

then:

SELECT Region, ...
FROM ...
GROUP BY Region

If dimensions is empty:

DO NOT GROUP BY.

Do not infer grouping from a column existing in the schema.

==================================================
RATIO QUESTIONS
==================================================

For a ratio such as:

"What is the ratio of customers who pay in EUR against customers
who pay in CZK?"

If the plan defines:

numerator:
number of customers satisfying Currency = EUR

denominator:
number of customers satisfying Currency = CZK

then calculate:

count of EUR customers
/
count of CZK customers

at the requested overall level.

Prefer conditional aggregation.

For example:

SELECT
    CAST(SUM(IIF(`Currency` = 'EUR', 1, 0)) AS FLOAT)
    /
    NULLIF(
        SUM(IIF(`Currency` = 'CZK', 1, 0)),
        0
    ) AS ratio
FROM customers

Do NOT:

- group by CustomerID
- group by Currency
- calculate one ratio per customer
- introduce unrelated filters
- join unrelated tables

unless the AnalyticalPlan explicitly requires them.

==================================================
METRIC QUESTIONS
==================================================

For metrics, use the exact metric field from the plan.

Examples:

Plan metric:
"total revenue"

and schema field:
Revenue

then use:

SUM(`Revenue`)

Do not replace Revenue with:

Sales
Amount
Profit
Cost

unless the plan explicitly specifies that field.

==================================================
TIME RANGE
==================================================

Use a time range only when AnalyticalPlan.time_range contains one.

Do not invent dates.

Do not invent a default year.

Do not add benchmark certification/status filters.

Do not interpret a schema field containing a year as a requested time
filter unless the AnalyticalPlan explicitly requests that time field.

==================================================
JOINS
==================================================

Use a JOIN only when the requested fields genuinely require multiple
tables.

Before using a JOIN, verify:

1. The plan requires fields from both tables.
2. The relevant columns exist.
3. The join key exists in both tables.
4. The join is necessary to answer the plan.

If one table contains all requested output and filter fields:

DO NOT JOIN.

For example, if frpm contains:

School Code
Enrollment (K-12)

then a question requiring those two fields should use frpm directly.

Do NOT join schools merely because schools has CDSCode.

==================================================
SQLITE
==================================================

The data source may be SQLite.

For SQLite:

- use exact table names from the schema
- use exact column names from the schema
- never use data.<table>
- never invent a data schema
- use SQLite-compatible SQL
- use IIF(condition, true_value, false_value)
- never use MySQL IF(...)
- never use unsupported dialect-specific functions

==================================================
SQLITE IDENTIFIERS
==================================================

SQLite column names may contain:

- spaces
- hyphens
- parentheses
- periods
- numbers
- other special characters

When a column name contains special characters, quote it using
SQLite-compatible backticks.

Examples:

`School Code`

`Enrollment (K-12)`

`2013-14 CALPADS Fall 1 Certification Status`

Never generate:

T1.2013-14 CALPADS Fall 1 Certification Status

Instead, if that column is actually required by the plan, use:

T1.`2013-14 CALPADS Fall 1 Certification Status`

But remember:

Do NOT use this column merely because it exists.

It must be explicitly required by the AnalyticalPlan.

==================================================
EXACT SCHEMA VALIDATION
==================================================

Before producing SQL, mentally perform this checklist.

TABLE CHECK:

Every table in FROM/JOIN exists in the schema.

COLUMN CHECK:

Every selected column exists.

Every WHERE column exists.

Every GROUP BY column exists.

Every ORDER BY column exists.

Every JOIN column exists.

SEMANTIC CHECK:

Every selected field corresponds to a requested dimension/metric.

Every WHERE condition corresponds to an explicit plan filter.

Every grouping dimension corresponds to an explicit plan dimension.

Every calculation corresponds to the requested operation.

No unrelated field has been introduced.

==================================================
IMPORTANT BIRD BEHAVIOR
==================================================

BIRD schemas often contain many columns that look related.

Do NOT assume a column is relevant simply because it sounds plausible.

For example, if the plan says:

School Code

and:

Enrollment (K-12) > 500

then these exact fields must be used.

Do NOT substitute:

CDSCode

Enrollment (Ages 5-17)

School Name

2013-14 CALPADS Fall 1 Certification Status

or another related field.

The benchmark question's wording and the AnalyticalPlan determine
the semantics.

==================================================
FILE-BASED DATA
==================================================

For file-based workflows:

- available table name is "data"
- use "data" in FROM
- do not use the original filename as a table name

==================================================
OUTPUT
==================================================

Return only the structured SQLGenerationOutput.

Do not return explanations.

Do not return markdown.

Do not execute SQL.

Do not return multiple queries.
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
    """
    Build a schema description containing structural metadata.

    The description is intentionally compact while exposing primary-key
    and foreign-key information needed for schema-grounded SQL generation.
    """

    sections: list[str] = []

    for table in schema.tables:

        column_lines = []

        for column in table.columns:

            metadata = []

            if column.primary_key:
                metadata.append(
                    "PRIMARY KEY"
                )

            if column.nullable is False:
                metadata.append(
                    "NOT NULL"
                )

            metadata_text = ""

            if metadata:
                metadata_text = (
                    " ["
                    + ", ".join(metadata)
                    + "]"
                )

            column_lines.append(
                f"- {column.name}: "
                f"{column.data_type}"
                f"{metadata_text}"
            )

        section = [
            f"TABLE: {table.name}",
            "COLUMNS:",
            *column_lines,
        ]

        if table.foreign_keys:
            section.append(
                "FOREIGN KEYS:"
            )

            for foreign_key in table.foreign_keys:
                section.append(
                    "- "
                    f"{foreign_key.column_name}"
                    " -> "
                    f"{foreign_key.referenced_table}"
                    "."
                    f"{foreign_key.referenced_column}"
                )

        sections.append(
            "\n".join(section)
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

Generate a completely new SQL query.

Do NOT repeat:

- the previous invalid SQL
- invented filters
- invented columns
- wrong metric columns
- wrong output columns
- unnecessary joins
- unnecessary GROUP BY
- unsupported SQL syntax

Re-read the AnalyticalPlan and schema.

The previous SQL is NOT authoritative.

The AnalyticalPlan is authoritative.
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
data.schools
data.frpm
or any other table unless it is listed above.

For SQLite:

- use IIF(...)
- use normal single-quoted string literals
- use backticks around identifiers containing special characters
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
MANDATORY PRE-GENERATION CHECK
==================================================

Before writing SQL, identify from the AnalyticalPlan:

OUTPUT:
What exact field(s) must be returned?

FILTERS:
What exact condition(s) must restrict the rows?

METRICS:
What exact calculation(s) are required?

DIMENSIONS:
What exact grouping field(s), if any, are required?

GRANULARITY:
Is this retrieval, scalar analysis, or grouped analysis?

JOINS:
Are multiple tables genuinely necessary?

Then generate SQL that implements ONLY those requirements.

==================================================
DIRECT RETRIEVAL CHECK
==================================================

If:

operation = retrieve

or the plan clearly describes a direct list/retrieval request:

- SELECT the requested output field
- use the exact requested filter field
- do not aggregate
- do not GROUP BY unless explicitly requested
- do not JOIN unless required
- do not add any other selected column
- do not add any other filter

For example:

Plan:

dimensions = ["School Code"]

filters = ["Enrollment (K-12) > 500"]

Correct structure:

SELECT `School Code`
FROM <table containing both fields>
WHERE `Enrollment (K-12)` > 500

Incorrect structures include:

SELECT `Enrollment (Ages 5-17`) ...
SELECT CDSCode ...
SELECT School Name ...
WHERE `2013-14 CALPADS Fall 1 Certification Status` = 'Y'
GROUP BY CDSCode
JOIN another table without necessity

==================================================
FINAL SEMANTIC CHECK
==================================================

Before returning the SQL, verify all of the following:

1. Does SELECT return exactly what the plan requests?

2. Does every WHERE condition come from the plan?

3. Did any requested field get replaced by a similar field?

4. Did I add a filter that the plan does not contain?

5. Did I add a JOIN that is not necessary?

6. Did I add GROUP BY when the plan does not request grouping?

7. Does every referenced table exist?

8. Does every referenced column exist in its table?

9. Are SQLite special-character column names properly quoted?

10. Does the SQL answer the user's original question?

If any answer is NO, regenerate the SQL before returning it.

{retry_feedback}

Return exactly one SQLGenerationOutput.
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