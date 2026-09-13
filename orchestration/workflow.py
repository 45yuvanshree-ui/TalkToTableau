"""End-to-end TalkToTableau orchestration workflow."""

from pathlib import Path
from uuid import uuid4

import re

from agents import Runner

from agent_components import (
    InsightVerificationWorkflow,
    create_business_analyst,
    create_data_storyteller,
    create_insight_verifier,
)
from analysis import AnalysisWorkflow
from context import SessionContext
from data_sources import DataSource, create_data_source
from guardrails import (
    validate_input,
    validate_insight_output,
    validate_sql,
    validate_story_factual_claims,
    validate_story_output,
)
from models import BusinessStory, InsightAnalysis
from routing import (
    AdaptiveRouter,
    AgentsModelFactory,
    RoutingRequest,
)
from schema import SchemaRetriever
from sql import SQLExecutor, generate_sql
from sql.validator import validate_sql_against_plan
from tracing import ApplicationTracer
from tracing.models import TraceEventType
from visualization import VisualizationWorkflow


MAX_SQL_RETRIES = 2
MAX_INSIGHT_RETRIES = 1
MAX_STORY_RETRIES = 1

SQLITE_EXTENSIONS = {
    ".sqlite",
    ".sqlite3",
    ".db",
}


class TalkToTableauWorkflow:
    def __init__(
        self,
        data_file: str | Path | None = None,
        model_factory: AgentsModelFactory | None = None,
        router: AdaptiveRouter | None = None,
        data_source: DataSource | None = None,
    ):
        if data_source is None and data_file is None:
            raise ValueError("Either data_file or data_source must be provided.")

        self.data_file = (
            Path(data_file)
            if data_file is not None
            else None
        )

        self.data_source = (
            data_source
            if data_source is not None
            else create_data_source(self.data_file)
        )

        self.model_factory = (
            model_factory
            if model_factory is not None
            else AgentsModelFactory()
        )

        self.router = (
            router
            if router is not None
            else AdaptiveRouter()
        )

        self.schema_retriever = SchemaRetriever()
        self.sql_executor = SQLExecutor()
        self.analysis_workflow = AnalysisWorkflow()
        self.insight_workflow = InsightVerificationWorkflow()
        self.visualization_workflow = VisualizationWorkflow()

        self.last_trace = None

    # ==============================================================
    # SOURCE HELPERS
    # ==============================================================

    def _is_sqlite_source(self) -> bool:
        """Return True when the configured source is SQLite."""

        return self.data_source.__class__.__name__ == "SQLiteDataSource"

    def _retrieve_schema(self):
        """
        Retrieve the schema through the DataSource abstraction.

        FileDataSource uses the conventional DuckDB table name ``data``.
        Other DataSource implementations use their native schema metadata.
        """

        if self.data_source.__class__.__name__ == "FileDataSource":
            return self.schema_retriever.from_source(
                source=self.data_source,
                table_name="data",
            )

        return self.schema_retriever.from_source(
            source=self.data_source,
        )

    def _execute_sql(
        self,
        query: str,
    ):
        """Execute SQL through SQLExecutor and the configured DataSource."""

        return self.sql_executor.execute_source(
            source=self.data_source,
            query=query,
        )

    # ==============================================================
    # MODEL ROUTING
    # ==============================================================

    def _route_model(
        self,
        user_request: str,
        sql_complexity: int = 1,
        ambiguity: int = 1,
        reasoning_required: bool = False,
        output_complexity: int = 1,
        schema_size: int = 1,
    ):
        routing_request = RoutingRequest(
            user_request=user_request,
            sql_complexity=sql_complexity,
            ambiguity=ambiguity,
            reasoning_required=reasoning_required,
            output_complexity=output_complexity,
            schema_size=schema_size,
        )

        decision = self.router.route(
            routing_request
        )

        model = self.model_factory.create_model(
            decision.tier
        )

        return model, decision

    # ==============================================================
    # RESULT COLUMN MATCHING
    # ==============================================================

    @staticmethod
    def _normalize_field_name(
        value: str,
    ) -> str:
        """
        Normalize field names for deterministic matching.

        Handles:
        - spaces
        - underscores
        - hyphens
        - camelCase
        - PascalCase
        - acronym boundaries
        """

        normalized = value.strip()

        normalized = re.sub(
            r"([a-z0-9])([A-Z])",
            r"\1_\2",
            normalized,
        )

        normalized = re.sub(
            r"([A-Z]+)([A-Z][a-z])",
            r"\1_\2",
            normalized,
        )

        normalized = normalized.lower()

        normalized = re.sub(
            r"[^a-z0-9]+",
            "_",
            normalized,
        )

        normalized = re.sub(
            r"_+",
            "_",
            normalized,
        )

        return normalized.strip("_")

    @classmethod
    def _find_result_column(
        cls,
        requested_name: str,
        result_columns: list[str],
    ) -> str | None:
        """
        Find a requested field in the SQL result.

        Supports aggregation aliases such as:

            revenue -> totalRevenue
            revenue -> total_revenue
            revenue -> sum_revenue
            revenue -> averageRevenue
        """

        requested_normalized = (
            cls._normalize_field_name(
                requested_name
            )
        )

        for column in result_columns:
            if (
                cls._normalize_field_name(column)
                == requested_normalized
            ):
                return column

        aggregation_words = {
            "total",
            "sum",
            "average",
            "avg",
            "mean",
            "count",
            "maximum",
            "max",
            "minimum",
            "min",
        }

        requested_tokens = set(
            requested_normalized.split("_")
        )

        requested_base_tokens = (
            requested_tokens - aggregation_words
        )

        if requested_base_tokens:
            for column in result_columns:

                column_normalized = (
                    cls._normalize_field_name(
                        column
                    )
                )

                column_tokens = set(
                    column_normalized.split("_")
                )

                column_base_tokens = (
                    column_tokens - aggregation_words
                )

                if (
                    requested_base_tokens
                    == column_base_tokens
                ):
                    return column

        return None

    # ==============================================================
    # SQL RESULT VALIDATION
    # ==============================================================

    @classmethod
    def _validate_sql_result_against_plan(
        cls,
        result_columns: list[str],
        plan,
    ) -> tuple[bool, str | None]:
        """
        Validate that the executed SQL result contains the
        analytical outputs required by the Business Analyst plan.

        Semantic distinction:
        - output_fields -> fields explicitly requested for direct retrieval
        - dimensions -> fields explicitly requested for grouping/breakdown
        - metrics -> calculated measures
        """

        missing_output_fields: list[str] = []
        missing_dimensions: list[str] = []
        missing_metrics: list[str] = []

        output_fields = getattr(
            plan,
            "output_fields",
            None,
        ) or []

        dimensions = getattr(
            plan,
            "dimensions",
            None,
        ) or []

        metrics = getattr(
            plan,
            "metrics",
            None,
        ) or []

        if not result_columns:
            return (
                False,
                "SQL result validation failed: "
                "the SQL query produced no result columns.",
            )

        # ----------------------------------------------------------
        # DIRECT RETRIEVAL OUTPUT FIELDS
        # ----------------------------------------------------------
        #
        # Direct list/retrieval requests are represented by
        # output_fields. These fields must actually appear in
        # the SQL result.
        #
        for output_field in output_fields:

            if (
                cls._find_result_column(
                    requested_name=output_field,
                    result_columns=result_columns,
                )
                is None
            ):
                missing_output_fields.append(
                    output_field
                )

        # ----------------------------------------------------------
        # GROUPING DIMENSIONS
        # ----------------------------------------------------------
        #
        # dimensions represent actual grouping/breakdown fields.
        #
        for dimension in dimensions:

            if (
                cls._find_result_column(
                    requested_name=dimension,
                    result_columns=result_columns,
                )
                is None
            ):
                missing_dimensions.append(
                    dimension
                )

        # ----------------------------------------------------------
        # GROUPED METRICS
        # ----------------------------------------------------------
        #
        # For grouped analytical requests, requested metrics must
        # appear in the result.
        #
        # Scalar/derived metrics may legitimately have a final
        # derived alias such as "ratio", so they are not required
        # to match the internal metric description verbatim.
        #
        if dimensions:

            for metric in metrics:

                if (
                    cls._find_result_column(
                        requested_name=metric,
                        result_columns=result_columns,
                    )
                    is None
                ):
                    missing_metrics.append(
                        metric
                    )

        if (
            missing_output_fields
            or missing_dimensions
            or missing_metrics
        ):

            problems: list[str] = []

            if missing_output_fields:
                problems.append(
                    "Missing required output fields "
                    f"in SQL result: {missing_output_fields}."
                )

            if missing_dimensions:
                problems.append(
                    "Missing required analytical dimensions "
                    f"in SQL result: {missing_dimensions}."
                )

            if missing_metrics:
                problems.append(
                    "Missing required analytical metrics "
                    f"in SQL result: {missing_metrics}."
                )

            return (
                False,
                " ".join(problems),
            )

        return True, None

    # ==============================================================
    # GENERATED SQL STRUCTURAL VALIDATION
    # ==============================================================

    @staticmethod
    def _validate_generated_sql_structure(
        query: str,
        plan,
    ) -> tuple[bool, str | None]:
        """
        Perform deterministic structural checks before SQL execution.

        Semantic distinction:
        - output_fields -> direct retrieval
        - dimensions -> grouping
        - metrics -> calculations
        """

        normalized = query.strip()

        if not normalized:
            return (
                False,
                "Generated SQL is empty.",
            )

        if not re.match(
            r"^\s*SELECT\b",
            normalized,
            flags=re.IGNORECASE,
        ):
            return (
                False,
                "Generated SQL must begin with SELECT.",
            )

        if not re.search(
            r"\bFROM\b",
            normalized,
            flags=re.IGNORECASE,
        ):
            return (
                False,
                "Generated SQL is missing a FROM clause. "
                "For file-based data use FROM data. "
                "Generate a complete executable SELECT query.",
            )

        dimensions = getattr(
            plan,
            "dimensions",
            None,
        ) or []

        # ONLY actual grouping dimensions require GROUP BY.
        #
        # output_fields do NOT require GROUP BY.
        #
        # Therefore:
        #
        # output_fields = ["School Code"]
        # dimensions = []
        #
        # is a direct retrieval query and must not be forced
        # into GROUP BY semantics.
        #
        if dimensions:

            if not re.search(
                r"\bGROUP\s+BY\b",
                normalized,
                flags=re.IGNORECASE,
            ):
                return (
                    False,
                    "The analytical plan requests grouped results, "
                    f"but the generated SQL has no GROUP BY clause. "
                    f"Group the result by the requested dimension(s): "
                    f"{dimensions}.",
                )

        return True, None

    # ==============================================================
    # SQL FEEDBACK
    # ==============================================================

    @staticmethod
    def _build_sql_retry_feedback(
        error: Exception,
        previous_sql: str | None,
        plan,
        schema,
    ) -> str:
        """
        Build precise feedback for a bounded SQL regeneration attempt.
        """

        output_fields = getattr(
            plan,
            "output_fields",
            None,
        ) or []

        dimensions = getattr(
            plan,
            "dimensions",
            None,
        ) or []

        metrics = getattr(
            plan,
            "metrics",
            None,
        ) or []

        filters = getattr(
            plan,
            "filters",
            None,
        ) or []

        # ----------------------------------------------------------
        # DIRECT RETRIEVAL
        # ----------------------------------------------------------

        if output_fields:

            retrieval_instruction = f"""
The analytical plan requests DIRECT RETRIEVAL of specific fields.

Requested output fields:
{output_fields}

The SQL MUST:
- SELECT every requested output field.
- Return the requested fields exactly.
- Apply only the filters explicitly present in the analytical plan.
- NOT replace a requested output field with a related field.
- NOT invent additional output fields.
- NOT invent metrics.
- NOT add unrelated joins.
- NOT add unrelated filters.
- NOT add GROUP BY unless dimensions are also explicitly requested.

IMPORTANT:

output_fields are NOT grouping dimensions.

For example:

User request:
"List the school codes of schools with total enrollment over 500."

Analytical plan:

output_fields:
["School Code"]

dimensions:
[]

filters:
["Enrollment (K-12) > 500"]

The SQL should have the equivalent semantic structure:

SELECT "School Code"
FROM <table containing the requested fields>
WHERE "Enrollment (K-12)" > 500

Use the actual table and exact column identifiers from the supplied schema.

Do NOT:
- SELECT Enrollment instead of School Code
- GROUP BY School Code
- GROUP BY Age
- add County filters
- add School Type filters
- add certification-status filters
- add unrelated joins
"""

        else:

            retrieval_instruction = """
The analytical plan does not request direct retrieval output fields.
"""

        # ----------------------------------------------------------
        # GROUPED ANALYTICS
        # ----------------------------------------------------------

        if dimensions:

            granularity_instruction = f"""
The analytical plan requests GROUPED results.

Requested grouping dimensions:
{dimensions}

Requested metrics:
{metrics}

Therefore the SQL MUST:
- select the requested dimension(s)
- calculate the requested metric(s)
- include FROM
- include GROUP BY for the requested dimension(s)
- return one row per requested dimension value

Example:

"total revenue by region"

Expected semantic structure:

SELECT
    region,
    SUM(revenue) AS total_revenue
FROM data
GROUP BY region

Do NOT select raw unaggregated metric values alongside the grouped
result unless that field is itself correctly aggregated.
"""

        else:

            granularity_instruction = f"""
The analytical plan does NOT request grouping dimensions.

Requested metrics:
{metrics}

If this is a scalar analytical request, return the requested scalar
calculation directly.

If this is a direct retrieval request, return the requested
output_fields directly.

Do NOT add GROUP BY unless dimensions are explicitly present in
the analytical plan.
"""

        # ----------------------------------------------------------
        # FILTERS
        # ----------------------------------------------------------

        if filters:

            filter_instruction = f"""
The analytical plan explicitly contains these global filters:

{filters}

Apply ONLY these filters.

Do not invent additional WHERE conditions.
"""

        else:

            filter_instruction = """
The analytical plan contains NO global filters.

Therefore:
- DO NOT add WHERE.
- DO NOT add IS NOT NULL conditions.
- DO NOT add arbitrary region filters.
- DO NOT add arbitrary country filters.
- DO NOT add arbitrary segment filters.
- DO NOT add certification-status filters.
"""

        # ----------------------------------------------------------
        # SOURCE
        # ----------------------------------------------------------

        source_instruction = ""

        if schema.source_type == "file":

            source_instruction = """
The data source is file-based.

The available DuckDB table is exactly:

data

Therefore the SQL must contain:

FROM data

Do not use the original filename as a table name.
"""

        elif schema.source_type == "sqlite":

            source_instruction = """
The data source is SQLite.

Use the exact SQLite table names from the supplied schema.

Do not use:

data.<table>

Do not invent a data schema.

Use exact schema column names.
"""

        return f"""
The previous SQL generation attempt failed.

ERROR:
{error}

PREVIOUS SQL:
{previous_sql}

==================================================
REGENERATE THE SQL COMPLETELY
==================================================

The analytical plan is authoritative.

{retrieval_instruction}

{granularity_instruction}

{filter_instruction}

{source_instruction}

==================================================
MANDATORY SQL REQUIREMENTS
==================================================

Generate exactly ONE complete executable SELECT statement.

The query MUST contain:

1. SELECT
2. FROM
3. valid table references
4. valid columns
5. correct analytical operation
6. GROUP BY only when dimensions are requested

Never return a partial SQL fragment.

Never invent:
- tables
- columns
- filters
- dimensions
- metrics
- joins

Never add unrelated conditions.

For direct retrieval requests:
- requested fields belong in SELECT
- requested conditions belong in WHERE
- output_fields do NOT imply GROUP BY

For grouped requests:
- dimensions belong in SELECT
- metrics must be aggregated
- dimensions require GROUP BY

For scalar requests:
- return the scalar analytical result
- do not introduce arbitrary grouping

For file-based data:

FROM data

For SQLite:
- use SQLite-compatible SQL
- use exact table names
- use exact column names
- use IIF(...) instead of IF(...)
- do not use data.<table>

Return only the structured SQLGenerationOutput.
"""

    # ==============================================================
    # STORY HELPERS
    # ==============================================================

    @staticmethod
    def _build_visualization_explanation(
        visualization_result,
        analytical_plan,
    ) -> str:
        """
        Build a deterministic visualization explanation when the
        local storytelling model returns an empty explanation.

        This does not introduce new numerical claims. It only describes
        the visualization metadata already produced by the visualization
        workflow.
        """

        if visualization_result is None:
            return (
                "No visualization was generated for this analysis."
            )

        visualization_data = (
            visualization_result.model_dump()
            if hasattr(
                visualization_result,
                "model_dump",
            )
            else {}
        )

        status = visualization_data.get(
            "status"
        )

        if status:
            status_text = str(status).replace(
                "_",
                " ",
            )
        else:
            status_text = "generated"

        chart_type = (
            visualization_data.get(
                "chart_type"
            )
            or visualization_data.get(
                "type"
            )
            or visualization_data.get(
                "chart"
            )
        )

        x_axis = (
            visualization_data.get(
                "x_axis"
            )
            or visualization_data.get(
                "x"
            )
        )

        y_axis = (
            visualization_data.get(
                "y_axis"
            )
            or visualization_data.get(
                "y"
            )
        )

        title = visualization_data.get(
            "title"
        )

        parts: list[str] = []

        if title:
            parts.append(
                f'The visualization "{title}" presents the '
                "analytical result."
            )
        else:
            parts.append(
                "The visualization presents the analytical result."
            )

        if chart_type:
            parts.append(
                f"It uses a {chart_type} chart."
            )

        if x_axis and y_axis:
            parts.append(
                f"The horizontal axis represents {x_axis}, "
                f"while the vertical axis represents {y_axis}."
            )
        elif x_axis:
            parts.append(
                f"The visualization uses {x_axis} as its main axis."
            )
        elif y_axis:
            parts.append(
                f"The visualization uses {y_axis} as its main measure."
            )

        if status_text:
            parts.append(
                f"The visualization workflow status is {status_text}."
            )

        return " ".join(parts)

    @classmethod
    def _repair_empty_visualization_explanation(
        cls,
        story: BusinessStory,
        visualization_result,
        analytical_plan,
    ) -> BusinessStory:
        """
        Deterministically repair only an empty visualization explanation.

        The repair uses visualization metadata already generated by the
        visualization workflow and does not modify the analytical story,
        SQL result, insights, or recommendations.
        """

        explanation = (
            story.visualization_explanation
            if story.visualization_explanation
            else ""
        )

        if explanation.strip():
            return story

        fallback_explanation = (
            cls._build_visualization_explanation(
                visualization_result=visualization_result,
                analytical_plan=analytical_plan,
            )
        )

        story.visualization_explanation = (
            fallback_explanation
        )

        return story

    # ==============================================================
    # MAIN WORKFLOW
    # ==============================================================

    def run(
        self,
        user_request: str,
        session: SessionContext | None = None,
    ) -> SessionContext:

        tracer = ApplicationTracer()

        context = (
            session
            if session is not None
            else SessionContext(
                session_id=str(uuid4())
            )
        )

        context.user_request = user_request
        context.data_source = (
            str(self.data_file)
            if self.data_file is not None
            else self.data_source.__class__.__name__
        )

        # ==========================================================
        # REQUEST
        # ==========================================================

        tracer.add_event(
            event_type=TraceEventType.REQUEST,
            message="User request received.",
            metadata={
                "user_request": user_request,
            },
        )

        # ==========================================================
        # INPUT GUARDRAIL
        # ==========================================================

        input_validation = validate_input(
            user_request
        )

        if not input_validation.is_valid:

            tracer.add_event(
                event_type=TraceEventType.INPUT_GUARDRAIL,
                message="Input rejected.",
                metadata={
                    "reason": input_validation.reason,
                },
            )

            raise ValueError(
                input_validation.reason
            )

        tracer.add_event(
            event_type=TraceEventType.INPUT_GUARDRAIL,
            message="Input accepted.",
            metadata={},
        )

        # ==========================================================
        # SCHEMA RETRIEVAL
        # ==========================================================

        tracer.add_event(
            event_type=TraceEventType.SCHEMA,
            message="Retrieving data schema.",
            metadata={
                "source": (
                    str(self.data_file)
                    if self.data_file is not None
                    else self.data_source.__class__.__name__
                ),
            },
        )

        data_schema = self._retrieve_schema()

        context.data_schema = data_schema

        schema_size = sum(
            len(table.columns)
            for table in data_schema.tables
        )

        tracer.add_event(
            event_type=TraceEventType.SCHEMA,
            message="Data schema retrieved.",
            metadata={
                "schema": data_schema.model_dump(),
                "schema_size": schema_size,
                "source_type": data_schema.source_type,
            },
        )

        # ==========================================================
        # AGENT 1 — BUSINESS ANALYST
        # ==========================================================

        (
            business_analyst_model,
            business_analysis_routing,
        ) = self._route_model(
            user_request=user_request,
            sql_complexity=2,
            ambiguity=2,
            reasoning_required=True,
            output_complexity=2,
            schema_size=max(
                schema_size,
                1,
            ),
        )

        tracer.add_event(
            event_type=TraceEventType.ROUTING,
            message="Model routed for Business Analyst.",
            metadata={
                "tier":
                    business_analysis_routing.tier.value,
                "reason":
                    business_analysis_routing.reason,
                "complexity_score":
                    business_analysis_routing.complexity_score,
            },
        )

        business_analyst = create_business_analyst(
            model=business_analyst_model
        )

        tracer.add_event(
            event_type=TraceEventType.AGENT,
            message="Business Analyst started.",
            metadata={
                "agent": "Business Analyst",
            },
        )

        agent_1_prompt = f"""
Convert the following user request into a structured
AnalyticalPlan.

User request:
{user_request}

Available data schema:
{data_schema.model_dump_json(indent=2)}

Use the Business Analyst instructions as the authoritative
guidance for interpreting the user's analytical intent.

Use only:
1. the user's request
2. the supplied schema
3. the Business Analyst instructions

The AnalyticalPlan must correctly distinguish:

- analytical objective
- primary analytical operation
- base metric definition
- requested output fields
- requested metrics
- actual grouping dimensions
- global filters
- metric-specific conditions
- numerator and denominator when applicable
- time range
- comparison requirements
- analytical question

IMPORTANT:
For direct retrieval or list requests, populate
`output_fields` with the exact schema fields that the user
explicitly asks to see.

For example:

User request:
"Please list the codes of the schools with a total enrollment
of over 500."

Correct interpretation:

- analytical_objective: retrieval
- operation: retrieve
- output_fields: ["School Code"]
- metrics: []
- dimensions: []
- filters: ["Enrollment (K-12) + Enrollment (Ages 5-17) > 500"]
- metric_conditions: []

Do NOT treat the requested output field as a grouping dimension.

Do NOT replace the requested output field with a field used only
to evaluate the filter.

Do NOT invent additional filters, dimensions, metrics, joins,
or output fields.

IMPORTANT SEMANTIC RULES:

A field used inside a metric calculation is NOT automatically
a grouping dimension.

A condition used to define a numerator, denominator,
conditional count, conditional sum, percentage, ratio,
proportion, share, or similar derived metric is NOT
automatically a global filter.

Only include a field in dimensions when the user explicitly
requests a grouping or breakdown by that field.

Only include a condition in filters when it restricts the
overall population being analyzed.

Metric-specific conditions belong in metric_conditions.

For ratio questions:

"What is the ratio of customers who pay in EUR against
customers who pay in CZK?"

must be interpreted as:

- analytical objective: ratio
- operation: ratio
- numerator: customers whose Currency is EUR
- denominator: customers whose Currency is CZK
- metric conditions: Currency = EUR and Currency = CZK
- dimensions: []
- filters: []

Currency MUST NOT be added to dimensions unless the user
explicitly asks for the result broken down by Currency.

"What percentage of customers pay in EUR?"

is a scalar metric request.

Currency = EUR is a metric condition, not a grouping dimension.

"What is the ratio of EUR customers to CZK customers by region?"

has region as a grouping dimension because the user explicitly
requested the breakdown by region.

Do not infer a grouping dimension merely because a field appears
inside a condition, CASE expression, conditional aggregation,
numerator, denominator, or metric definition.

A time range by itself does NOT imply trend analysis.

"What was revenue in 2025?"

is not automatically a trend request.

"How has revenue changed over time?"

is a trend request.

If the user requests a scalar result, do not invent dimensions.

If the user requests a grouped result, preserve every explicitly
requested grouping dimension.

Preserve the distinction between:

1. grouping dimensions
2. global filters
3. metric-specific conditions
4. numerator/denominator definitions

Do not:
- generate SQL
- execute SQL
- invent columns
- invent tables
- invent filters
- invent dimensions
- invent metric conditions
- invent data
- create insights
- create visualizations
- create recommendations
- silently add assumptions unsupported by the request or schema

If the request is ambiguous, preserve the ambiguity rather than
inventing an interpretation.

Return only the structured AnalyticalPlan.
"""

        agent_1_result = Runner.run_sync(
            business_analyst,
            agent_1_prompt,
        )

        analytical_plan = (
            agent_1_result.final_output
        )

        context.analytical_plan = (
            analytical_plan
        )

        tracer.add_event(
            event_type=TraceEventType.AGENT,
            message="Business Analyst completed.",
            metadata={
                "agent": "Business Analyst",
                "plan":
                    analytical_plan.model_dump(),
            },
        )

        # ==========================================================
        # SQL GENERATION + VALIDATION + EXECUTION
        # ==========================================================

        sql_model, sql_routing = self._route_model(
            user_request=user_request,
            sql_complexity=3,
            ambiguity=2,
            reasoning_required=True,
            output_complexity=2,
            schema_size=max(
                schema_size,
                1,
            ),
        )

        tracer.add_event(
            event_type=TraceEventType.ROUTING,
            message="Model routed for SQL generation.",
            metadata={
                "tier":
                    sql_routing.tier.value,
                "reason":
                    sql_routing.reason,
                "complexity_score":
                    sql_routing.complexity_score,
            },
        )

        last_sql_error = None
        sql_feedback = None

        for attempt in range(
            MAX_SQL_RETRIES + 1
        ):

            try:

                tracer.add_event(
                    event_type=TraceEventType.SQL,
                    message="Generating SQL.",
                    metadata={
                        "attempt":
                            attempt + 1,
                        "feedback":
                            sql_feedback,
                    },
                )

                generated_sql = generate_sql(
                    model=sql_model,
                    user_request=user_request,
                    plan=analytical_plan,
                    schema=data_schema,
                    feedback=sql_feedback,
                )

                context.generated_sql = (
                    generated_sql
                )

                tracer.add_event(
                    event_type=TraceEventType.SQL,
                    message="SQL generated.",
                    metadata={
                        "query":
                            generated_sql,
                        "attempt":
                            attempt + 1,
                    },
                )

                # --------------------------------------------------
                # STRUCTURAL VALIDATION
                # --------------------------------------------------

                (
                    structure_valid,
                    structure_reason,
                ) = self._validate_generated_sql_structure(
                    query=generated_sql,
                    plan=analytical_plan,
                )

                if not structure_valid:

                    tracer.add_event(
                        event_type=TraceEventType.SQL,
                        message="SQL structural validation failed.",
                        metadata={
                            "reason":
                                structure_reason,
                            "attempt":
                                attempt + 1,
                        },
                    )

                    raise ValueError(
                        structure_reason
                    )

                # --------------------------------------------------
                # SQL GUARDRAIL
                # --------------------------------------------------

                sql_guardrail = validate_sql(
                    generated_sql
                )

                if not sql_guardrail.is_valid:

                    raise ValueError(
                        sql_guardrail.reason
                    )

                # --------------------------------------------------
                # PLAN VALIDATION
                # --------------------------------------------------

                sql_validation = (
                    validate_sql_against_plan(
                        query=generated_sql,
                        plan=analytical_plan,
                    )
                )

                if not sql_validation.is_valid:

                    tracer.add_event(
                        event_type=TraceEventType.SQL,
                        message="SQL validation failed.",
                        metadata={
                            "reason":
                                sql_validation.reason,
                            "attempt":
                                attempt + 1,
                        },
                    )

                    raise ValueError(
                        sql_validation.reason
                    )

                tracer.add_event(
                    event_type=TraceEventType.SQL,
                    message="SQL validation passed.",
                    metadata={
                        "attempt":
                            attempt + 1,
                    },
                )

                # --------------------------------------------------
                # EXECUTION
                # --------------------------------------------------

                try:

                    sql_result = self._execute_sql(
                        query=sql_validation.normalized_query
                    )

                except Exception as execution_error:

                    raise ValueError(
                        "SQL execution failed: "
                        f"{execution_error}"
                    ) from execution_error

                context.sql_result = (
                    sql_result
                )

                tracer.add_event(
                    event_type=TraceEventType.SQL,
                    message="SQL execution completed.",
                    metadata={
                        "row_count":
                            sql_result.row_count,
                        "columns":
                            sql_result.columns,
                    },
                )

                # --------------------------------------------------
                # RESULT VALIDATION
                # --------------------------------------------------

                (
                    result_valid,
                    result_reason,
                ) = self._validate_sql_result_against_plan(
                    result_columns=
                        sql_result.columns,
                    plan=
                        analytical_plan,
                )

                if not result_valid:

                    tracer.add_event(
                        event_type=TraceEventType.SQL,
                        message="SQL result validation failed.",
                        metadata={
                            "reason":
                                result_reason,
                            "columns":
                                sql_result.columns,
                            "attempt":
                                attempt + 1,
                        },
                    )

                    raise ValueError(
                        result_reason
                    )

                tracer.add_event(
                    event_type=TraceEventType.SQL,
                    message="SQL result validation passed.",
                    metadata={
                        "columns":
                            sql_result.columns,
                        "attempt":
                            attempt + 1,
                    },
                )

                sql_feedback = None

                break

            except Exception as error:

                last_sql_error = error

                sql_feedback = (
                    self._build_sql_retry_feedback(
                        error=error,
                        previous_sql=
                            context.generated_sql,
                        plan=
                            analytical_plan,
                        schema=
                            data_schema,
                    )
                )

                tracer.add_event(
                    event_type=TraceEventType.ERROR,
                    message="SQL attempt failed.",
                    metadata={
                        "attempt":
                            attempt + 1,
                        "error":
                            str(error),
                    },
                )

                if attempt >= MAX_SQL_RETRIES:

                    raise RuntimeError(
                        f"SQL workflow failed after "
                        f"{MAX_SQL_RETRIES + 1} attempt(s)."
                    ) from last_sql_error

        if context.sql_result is None:

            raise RuntimeError(
                "SQL workflow completed without a result."
            )

        # ==========================================================
        # DETERMINISTIC ANALYSIS
        # ==========================================================

        tracer.add_event(
            event_type=TraceEventType.ANALYSIS,
            message="Analytical result processing started.",
            metadata={},
        )

        analysis_result = (
            self.analysis_workflow.run(
                plan=analytical_plan,
                sql_result=context.sql_result,
            )
        )

        context.analysis_result = (
            analysis_result
        )

        tracer.add_event(
            event_type=TraceEventType.ANALYSIS,
            message="Analytical result processing completed.",
            metadata={
                "analysis":
                    analysis_result.model_dump(),
            },
        )

        # ==========================================================
        # AGENT 2 — INSIGHT + EVIDENCE VERIFICATION
        # ==========================================================

        insight_model, insight_routing = self._route_model(
            user_request=user_request,
            sql_complexity=2,
            ambiguity=1,
            reasoning_required=True,
            output_complexity=3,
            schema_size=max(
                schema_size,
                1,
            ),
        )

        tracer.add_event(
            event_type=TraceEventType.ROUTING,
            message="Model routed for Insight Verification.",
            metadata={
                "tier":
                    insight_routing.tier.value,
                "reason":
                    insight_routing.reason,
                "complexity_score":
                    insight_routing.complexity_score,
            },
        )

        insight_context = (
            self.insight_workflow.build_context(
                plan=analytical_plan,
                sql_result=context.sql_result,
                analysis_result=analysis_result,
            )
        )

        insight_analysis = None
        last_insight_error = None

        for attempt in range(
            MAX_INSIGHT_RETRIES + 1
        ):

            try:

                tracer.add_event(
                    event_type=TraceEventType.EVIDENCE,
                    message="Insight verification started.",
                    metadata={
                        "agent":
                            "Insight and Evidence Verification",
                        "attempt":
                            attempt + 1,
                    },
                )

                insight_agent = (
                    create_insight_verifier(
                        model=insight_model
                    )
                )

                agent_2_prompt = f"""
Verify the analytical result using only the supplied evidence.

Analytical context:
{insight_context}

You must:
- verify numerical claims
- verify comparisons
- verify rankings
- verify highest/lowest claims
- verify second-highest/second-lowest claims
- verify third-highest/third-lowest claims
- identify unsupported claims
- never invent missing information

The SQL result is authoritative.

Return only the structured InsightAnalysis.
"""

                agent_2_result = Runner.run_sync(
                    insight_agent,
                    agent_2_prompt,
                )

                insight_analysis = (
                    agent_2_result.final_output
                )

                if not isinstance(
                    insight_analysis,
                    InsightAnalysis,
                ):
                    raise ValueError(
                        "Agent 2 returned an invalid "
                        "structured output."
                    )

                insight_analysis = (
                    self.insight_workflow.validate_output(
                        insight_analysis
                    )
                )

                insight_analysis = (
                    self.insight_workflow
                    .validate_against_sql_result(
                        result=
                            insight_analysis,
                        sql_result=
                            context.sql_result,
                    )
                )

                insight_analysis = (
                    self.insight_workflow.validate_output(
                        insight_analysis
                    )
                )

                insight_valid, insight_reason = (
                    validate_insight_output(
                        insight_analysis
                    )
                )

                if not insight_valid:

                    raise ValueError(
                        insight_reason
                    )

                context.insights = (
                    insight_analysis
                )

                tracer.add_event(
                    event_type=TraceEventType.EVIDENCE,
                    message="Insight verification completed.",
                    metadata={
                        "insight_analysis":
                            insight_analysis.model_dump(),
                        "attempt":
                            attempt + 1,
                    },
                )

                break

            except (
                ValueError,
                RuntimeError,
            ) as error:

                last_insight_error = error

                tracer.add_event(
                    event_type=TraceEventType.ERROR,
                    message="Insight verification attempt failed.",
                    metadata={
                        "attempt":
                            attempt + 1,
                        "error":
                            str(error),
                    },
                )

                if attempt >= MAX_INSIGHT_RETRIES:

                    raise RuntimeError(
                        f"Agent 2 failed after "
                        f"{MAX_INSIGHT_RETRIES + 1} attempt(s)."
                    ) from last_insight_error

        if context.insights is None:

            raise RuntimeError(
                "Agent 2 completed without an insight analysis."
            )

        # ==========================================================
        # VISUALIZATION
        # ==========================================================

        tracer.add_event(
            event_type=TraceEventType.VISUALIZATION,
            message="Visualization workflow started.",
            metadata={},
        )

        (
            visualization_result,
            _figure,
        ) = self.visualization_workflow.run(
            plan=analytical_plan,
            sql_result=context.sql_result,
            analysis_result=analysis_result,
        )

        context.visualization = (
            visualization_result
        )

        tracer.add_event(
            event_type=TraceEventType.VISUALIZATION,
            message="Visualization workflow completed.",
            metadata={
                "visualization":
                    visualization_result.model_dump(),
            },
        )

        # ==========================================================
        # AGENT 3 — DATA STORYTELLING
        # ==========================================================

        story_model, story_routing = self._route_model(
            user_request=user_request,
            sql_complexity=1,
            ambiguity=1,
            reasoning_required=True,
            output_complexity=4,
            schema_size=max(
                schema_size,
                1,
            ),
        )

        tracer.add_event(
            event_type=TraceEventType.ROUTING,
            message="Model routed for Data Storytelling.",
            metadata={
                "tier":
                    story_routing.tier.value,
                "reason":
                    story_routing.reason,
                "complexity_score":
                    story_routing.complexity_score,
            },
        )

        story_context = {
            "analytical_plan":
                analytical_plan.model_dump(),

            "sql_result":
                context.sql_result.model_dump(),

            "analysis_result":
                analysis_result.model_dump(),

            "insight_analysis":
                context.insights.model_dump(),

            "visualization_result":
                visualization_result.model_dump(),
        }

        # ----------------------------------------------------------
        # Strong deterministic visualization instructions
        # ----------------------------------------------------------

        visualization_explanation_instruction = (
            self._build_visualization_explanation(
                visualization_result=visualization_result,
                analytical_plan=analytical_plan,
            )
        )

        base_story_prompt = f"""
Create the final business story from the verified
analytical evidence below.

Context:
{story_context}

==================================================
STRICT DATA RULES
==================================================

- Use only the supplied evidence.
- Never invent numbers.
- Never change SQL result values.
- Never invent dimensions.
- Never invent categories.
- Never invent causes.
- Never contradict verified insights.
- Never create unsupported recommendations.
- If evidence is insufficient, recommendations MUST be empty.
- The SQL result is the authoritative source for numerical values.

==================================================
REQUIRED BUSINESS STORY
==================================================

The structured BusinessStory MUST contain:

1. A non-empty summary.
2. At least one section.
3. Every section must have:
   - a non-empty heading
   - non-empty content
4. A NON-EMPTY visualization_explanation.
5. Recommendations only when directly supported by evidence.

==================================================
VISUALIZATION EXPLANATION
==================================================

The visualization explanation is REQUIRED.

You MUST populate:

visualization_explanation

Do NOT leave it blank.

The visualization metadata is:

{visualization_result.model_dump()}

A safe explanation based only on the supplied visualization metadata is:

{visualization_explanation_instruction}

You may use this wording or a clearer equivalent.

Do not invent additional chart properties.

==================================================
FACTUAL VALIDATION
==================================================

- Any numerical value mentioned must exactly match a value
  present in the SQL result.
- Highest claims must match the actual highest value.
- Lowest claims must match the actual lowest value.
- Do not claim a category is highest or lowest unless the
  supplied evidence supports it.
- Do not mention unsupported numbers.
- Do not create unsupported recommendations.

Return ONLY the structured BusinessStory.
"""

        story = None
        last_story_error = None

        for attempt in range(
            MAX_STORY_RETRIES + 1
        ):

            try:

                tracer.add_event(
                    event_type=TraceEventType.STORY,
                    message="Data Storyteller started.",
                    metadata={
                        "agent":
                            "Data Storyteller",
                        "attempt":
                            attempt + 1,
                    },
                )

                storyteller = (
                    create_data_storyteller(
                        model=story_model
                    )
                )

                if attempt == 0:

                    agent_3_prompt = (
                        base_story_prompt
                    )

                else:

                    agent_3_prompt = f"""
The previous Data Storyteller output failed
deterministic validation.

Validation failure:
{last_story_error}

Create the BusinessStory again.

Analytical context:
{story_context}

==================================================
MANDATORY CORRECTIONS
==================================================

- summary MUST be non-empty.
- sections MUST contain at least one section.
- every section heading MUST be non-empty.
- every section content MUST be non-empty.
- visualization_explanation MUST be non-empty.
- recommendations MUST contain only evidence-supported
  recommendations.
- if verification status is insufficient_evidence,
  recommendations MUST be an empty list.

==================================================
VISUALIZATION
==================================================

The visualization metadata is:

{visualization_result.model_dump()}

You MUST fill visualization_explanation.

Never leave:

visualization_explanation = ""

Instead, explain what the supplied visualization shows.

A safe explanation is:

{visualization_explanation_instruction}

==================================================
FACTUAL RULES
==================================================

- Use only the supplied evidence.
- Never invent numbers.
- Every numerical value must exactly match the SQL result.
- Never invent categories.
- Never invent dimensions.
- Never invent causes.
- Never contradict Agent 2.
- Highest claims must match the actual highest value.
- Lowest claims must match the actual lowest value.
- If evidence is insufficient, recommendations MUST be empty.

Return ONLY the structured BusinessStory.
"""

                agent_3_result = Runner.run_sync(
                    storyteller,
                    agent_3_prompt,
                )

                candidate_story = (
                    agent_3_result.final_output
                )

                if not isinstance(
                    candidate_story,
                    BusinessStory,
                ):
                    raise ValueError(
                        "Agent 3 returned an invalid "
                        "structured output."
                    )

                # --------------------------------------------------
                # Deterministic repair of ONLY empty visualization
                # explanation.
                # --------------------------------------------------

                candidate_story = (
                    self._repair_empty_visualization_explanation(
                        story=candidate_story,
                        visualization_result=
                            visualization_result,
                        analytical_plan=
                            analytical_plan,
                    )
                )

                # --------------------------------------------------
                # Structural story validation
                # --------------------------------------------------

                (
                    valid_story,
                    story_reason,
                ) = validate_story_output(
                    story=candidate_story,
                    insight_analysis=context.insights,
                )

                if not valid_story:

                    last_story_error = (
                        story_reason
                    )

                    tracer.add_event(
                        event_type=TraceEventType.ERROR,
                        message=(
                            "Data Storyteller output failed "
                            "structural validation."
                        ),
                        metadata={
                            "attempt":
                                attempt + 1,
                            "reason":
                                story_reason,
                        },
                    )

                    if attempt >= MAX_STORY_RETRIES:

                        raise RuntimeError(
                            f"Agent 3 failed after "
                            f"{MAX_STORY_RETRIES + 1} attempt(s)."
                        )

                    continue

                # --------------------------------------------------
                # Factual story validation
                # --------------------------------------------------

                (
                    factual_story_valid,
                    factual_reason,
                ) = validate_story_factual_claims(
                    story=candidate_story,
                    sql_result=context.sql_result,
                )

                if not factual_story_valid:

                    last_story_error = (
                        factual_reason
                    )

                    tracer.add_event(
                        event_type=TraceEventType.ERROR,
                        message=(
                            "Data Storyteller output failed "
                            "factual validation."
                        ),
                        metadata={
                            "attempt":
                                attempt + 1,
                            "reason":
                                factual_reason,
                        },
                    )

                    if attempt >= MAX_STORY_RETRIES:

                        raise RuntimeError(
                            f"Agent 3 failed after "
                            f"{MAX_STORY_RETRIES + 1} attempt(s)."
                        )

                    continue

                story = candidate_story
                context.story = story

                tracer.add_event(
                    event_type=TraceEventType.STORY,
                    message="Data Storyteller completed.",
                    metadata={
                        "story":
                            story.model_dump(),
                        "attempt":
                            attempt + 1,
                    },
                )

                break

            except (
                ValueError,
                RuntimeError,
            ) as error:

                last_story_error = error

                tracer.add_event(
                    event_type=TraceEventType.ERROR,
                    message="Data Storyteller attempt failed.",
                    metadata={
                        "attempt":
                            attempt + 1,
                        "error":
                            str(error),
                    },
                )

                if attempt >= MAX_STORY_RETRIES:

                    raise RuntimeError(
                        f"Agent 3 failed after "
                        f"{MAX_STORY_RETRIES + 1} attempt(s)."
                    ) from last_story_error

        if context.story is None:

            raise RuntimeError(
                "Agent 3 completed without a story."
            )

        # ==========================================================
        # FINAL OUTPUT
        # ==========================================================

        tracer.add_event(
            event_type=TraceEventType.OUTPUT,
            message="TalkToTableau workflow completed.",
            metadata={
                "status":
                    "completed",
                "session_id":
                    context.session_id,
            },
        )

        self.last_trace = tracer.complete()

        return context