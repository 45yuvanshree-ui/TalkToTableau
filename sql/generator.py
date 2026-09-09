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
into one executable, read-only SQL query.

You must:

- use only tables and columns present in the supplied schema
- follow the analytical objective
- use the requested metrics
- use the requested dimensions
- apply only filters explicitly requested by the user
- apply the requested time range when present
- apply the requested comparison when present
- generate exactly one executable SELECT query
- use real SQL values and expressions
- use the table name "data" for the file-based workflow

IMPORTANT:
- Every requested analytical dimension MUST appear in the SELECT
  result when the analytical plan requires that dimension.
- Every requested analytical metric MUST appear in the SELECT
  result.
- If a dimension is used for a "by X" analysis, it must be selected
  and normally included in GROUP BY.
- Do not remove a requested dimension merely because the metric can
  be calculated without it.
- The SQL result must contain the fields required for the
  downstream analytical plan.

IMPORTANT:
- Never use placeholders.
- Never use template variables.
- Never use ${...}.
- Never use {{...}}.
- Never use parameter names such as :region or <region>.
- If no filter value was explicitly requested, do not add a WHERE
  clause for that dimension.
- For "total revenue by region", generate a query equivalent to:

  SELECT region, SUM(revenue) AS total_revenue
  FROM data
  GROUP BY region

You must never:

- generate INSERT, UPDATE, DELETE, DROP, ALTER, CREATE,
  TRUNCATE, GRANT, REVOKE, COPY, VACUUM, or ANALYZE
- invent tables
- invent columns
- invent filter values
- invent business requirements
- execute SQL
- return explanations

If a previous SQL attempt failed because required result columns
were missing, correct that problem in the new query.

Return only the structured SQLGenerationOutput.
"""


def create_sql_generator(
    model: Model,
) -> Agent:
    """Create the SQL generation component."""

    return Agent(
        name="SQL Generator",
        instructions=SQL_GENERATOR_INSTRUCTIONS,
        output_type=SQLGenerationOutput,
        model=model,
    )


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

    Optional feedback describes a failure from a previous SQL
    attempt so a bounded retry can correct the query.
    """

    sql_agent = create_sql_generator(model)

    retry_feedback = ""

    if feedback:
        retry_feedback = f"""
Previous SQL attempt feedback:
{feedback}

You MUST correct the previous problem in this attempt.
"""

    prompt = f"""
User request:
{user_request}

Analytical plan:
{plan.model_dump_json(indent=2)}

Available schema:
{schema.model_dump_json(indent=2)}

For this file-based workflow, the data is available to DuckDB
using the table name "data".

Therefore:
- Use "data" in the FROM clause.
- Do not use the original filename as the table name.
- Use only columns present in the supplied schema.
- Include every requested analytical dimension in the SELECT
  result when the plan requires it.
- Include every requested analytical metric in the SELECT result.

{retry_feedback}

Generate the SQL query required to answer the user's request.
"""

    result = Runner.run_sync(
        sql_agent,
        prompt,
    )

    output = result.final_output

    if not isinstance(output, SQLGenerationOutput):
        raise ValueError(
            "SQL generator returned an invalid structured output."
        )

    return output.query.strip()