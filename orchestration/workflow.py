"""End-to-end TalkToTableau orchestration workflow."""

from pathlib import Path
from uuid import uuid4

from agents import Runner

from agent_components import (
    InsightVerificationWorkflow,
    create_business_analyst,
    create_data_storyteller,
    create_insight_verifier,
)
from analysis import AnalysisWorkflow
from context import SessionContext
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


class TalkToTableauWorkflow:
    def __init__(
        self,
        data_file: str | Path,
        model_factory: AgentsModelFactory | None = None,
        router: AdaptiveRouter | None = None,
    ):
        self.data_file = Path(data_file)

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

        decision = self.router.route(routing_request)

        model = self.model_factory.create_model(
            decision.tier
        )

        return model, decision

    @staticmethod
    def _normalize_field_name(value: str) -> str:
        """Normalize field names for deterministic matching."""

        import re

        normalized = value.strip().lower()

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
        """Find a requested field in the SQL result."""

        requested_normalized = cls._normalize_field_name(
            requested_name
        )

        for column in result_columns:
            if (
                cls._normalize_field_name(column)
                == requested_normalized
            ):
                return column

        return None

    @classmethod
    def _validate_sql_result_against_plan(
        cls,
        result_columns: list[str],
        plan,
    ) -> tuple[bool, str | None]:
        """
        Ensure the executed SQL result contains every analytical
        field required by the Business Analyst plan.
        """

        missing_dimensions: list[str] = []
        missing_metrics: list[str] = []

        for dimension in plan.dimensions:
            if (
                cls._find_result_column(
                    requested_name=dimension,
                    result_columns=result_columns,
                )
                is None
            ):
                missing_dimensions.append(dimension)

        for metric in plan.metrics:
            if (
                cls._find_result_column(
                    requested_name=metric,
                    result_columns=result_columns,
                )
                is None
            ):
                missing_metrics.append(metric)

        if missing_dimensions or missing_metrics:
            problems: list[str] = []

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

            return False, " ".join(problems)

        return True, None

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
        context.data_source = str(self.data_file)

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

        input_validation = validate_input(user_request)

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
                "source": str(self.data_file),
            },
        )

        data_schema = self.schema_retriever.from_file(
            file_path=self.data_file,
            table_name="data",
        )

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
            },
        )

        # ==========================================================
        # AGENT 1 — BUSINESS ANALYST
        # ==========================================================

        business_analyst_model, business_analysis_routing = (
            self._route_model(
                user_request=user_request,
                sql_complexity=2,
                ambiguity=2,
                reasoning_required=True,
                output_complexity=2,
                schema_size=max(schema_size, 1),
            )
        )

        tracer.add_event(
            event_type=TraceEventType.ROUTING,
            message="Model routed for Business Analyst.",
            metadata={
                "tier": business_analysis_routing.tier.value,
                "reason": business_analysis_routing.reason,
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
analytical plan.

User request:
{user_request}

Available data schema:
{data_schema.model_dump_json(indent=2)}

Use only the supplied request and schema.

Do not:
- generate SQL
- invent columns
- invent filters
- invent data
- create insights
- create visualizations
- create recommendations

Return only the structured AnalyticalPlan.
"""

        agent_1_result = Runner.run_sync(
            business_analyst,
            agent_1_prompt,
        )

        analytical_plan = agent_1_result.final_output

        context.analytical_plan = analytical_plan

        tracer.add_event(
            event_type=TraceEventType.AGENT,
            message="Business Analyst completed.",
            metadata={
                "agent": "Business Analyst",
                "plan": analytical_plan.model_dump(),
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
            schema_size=max(schema_size, 1),
        )

        tracer.add_event(
            event_type=TraceEventType.ROUTING,
            message="Model routed for SQL generation.",
            metadata={
                "tier": sql_routing.tier.value,
                "reason": sql_routing.reason,
                "complexity_score":
                    sql_routing.complexity_score,
            },
        )

        last_sql_error = None
        sql_feedback = None

        for attempt in range(MAX_SQL_RETRIES + 1):

            try:

                tracer.add_event(
                    event_type=TraceEventType.SQL,
                    message="Generating SQL.",
                    metadata={
                        "attempt": attempt + 1,
                        "feedback": sql_feedback,
                    },
                )

                generated_sql = generate_sql(
                    model=sql_model,
                    user_request=user_request,
                    plan=analytical_plan,
                    schema=data_schema,
                    feedback=sql_feedback,
                )

                context.generated_sql = generated_sql

                tracer.add_event(
                    event_type=TraceEventType.SQL,
                    message="SQL generated.",
                    metadata={
                        "query": generated_sql,
                        "attempt": attempt + 1,
                    },
                )

                sql_guardrail = validate_sql(
                    generated_sql
                )

                if not sql_guardrail.is_valid:
                    raise ValueError(
                        sql_guardrail.reason
                    )

                sql_validation = validate_sql_against_plan(
                    query=generated_sql,
                    plan=analytical_plan,
                )

                if not sql_validation.is_valid:

                    tracer.add_event(
                        event_type=TraceEventType.SQL,
                        message="SQL validation failed.",
                        metadata={
                            "reason": sql_validation.reason,
                            "attempt": attempt + 1,
                        },
                    )

                    raise ValueError(
                        sql_validation.reason
                    )

                tracer.add_event(
                    event_type=TraceEventType.SQL,
                    message="SQL validation passed.",
                    metadata={
                        "attempt": attempt + 1,
                    },
                )

                sql_result = self.sql_executor.execute_file(
                    file_path=self.data_file,
                    query=sql_validation.normalized_query,
                )

                context.sql_result = sql_result

                tracer.add_event(
                    event_type=TraceEventType.SQL,
                    message="SQL execution completed.",
                    metadata={
                        "row_count": sql_result.row_count,
                        "columns": sql_result.columns,
                    },
                )

                result_valid, result_reason = (
                    self._validate_sql_result_against_plan(
                        result_columns=sql_result.columns,
                        plan=analytical_plan,
                    )
                )

                if not result_valid:

                    tracer.add_event(
                        event_type=TraceEventType.SQL,
                        message="SQL result validation failed.",
                        metadata={
                            "reason": result_reason,
                            "columns": sql_result.columns,
                            "attempt": attempt + 1,
                        },
                    )

                    raise ValueError(
                        result_reason
                    )

                tracer.add_event(
                    event_type=TraceEventType.SQL,
                    message="SQL result validation passed.",
                    metadata={
                        "columns": sql_result.columns,
                        "attempt": attempt + 1,
                    },
                )

                sql_feedback = None

                break

            except (ValueError, RuntimeError) as error:

                last_sql_error = error
                sql_feedback = str(error)

                tracer.add_event(
                    event_type=TraceEventType.ERROR,
                    message="SQL attempt failed.",
                    metadata={
                        "attempt": attempt + 1,
                        "error": str(error),
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

        analysis_result = self.analysis_workflow.run(
            plan=analytical_plan,
            sql_result=context.sql_result,
        )

        context.analysis_result = analysis_result

        tracer.add_event(
            event_type=TraceEventType.ANALYSIS,
            message="Analytical result processing completed.",
            metadata={
                "analysis": analysis_result.model_dump(),
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
            schema_size=max(schema_size, 1),
        )

        tracer.add_event(
            event_type=TraceEventType.ROUTING,
            message="Model routed for Insight Verification.",
            metadata={
                "tier": insight_routing.tier.value,
                "reason": insight_routing.reason,
                "complexity_score":
                    insight_routing.complexity_score,
            },
        )

        insight_context = self.insight_workflow.build_context(
            plan=analytical_plan,
            sql_result=context.sql_result,
            analysis_result=analysis_result,
        )

        insight_analysis = None
        last_insight_error = None

        for attempt in range(MAX_INSIGHT_RETRIES + 1):

            try:

                tracer.add_event(
                    event_type=TraceEventType.EVIDENCE,
                    message="Insight verification started.",
                    metadata={
                        "agent":
                            "Insight and Evidence Verification",
                        "attempt": attempt + 1,
                    },
                )

                insight_agent = create_insight_verifier(
                    model=insight_model
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
                        result=insight_analysis,
                        sql_result=context.sql_result,
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

                context.insights = insight_analysis

                tracer.add_event(
                    event_type=TraceEventType.EVIDENCE,
                    message="Insight verification completed.",
                    metadata={
                        "insight_analysis":
                            insight_analysis.model_dump(),
                        "attempt": attempt + 1,
                    },
                )

                break

            except (ValueError, RuntimeError) as error:

                last_insight_error = error

                tracer.add_event(
                    event_type=TraceEventType.ERROR,
                    message="Insight verification attempt failed.",
                    metadata={
                        "attempt": attempt + 1,
                        "error": str(error),
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

        visualization_result, _figure = (
            self.visualization_workflow.run(
                plan=analytical_plan,
                sql_result=context.sql_result,
                analysis_result=analysis_result,
            )
        )

        context.visualization = visualization_result

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
            schema_size=max(schema_size, 1),
        )

        tracer.add_event(
            event_type=TraceEventType.ROUTING,
            message="Model routed for Data Storytelling.",
            metadata={
                "tier": story_routing.tier.value,
                "reason": story_routing.reason,
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

        story = None
        last_story_error = None

        for attempt in range(MAX_STORY_RETRIES + 1):

            try:

                tracer.add_event(
                    event_type=TraceEventType.STORY,
                    message="Data Storyteller started.",
                    metadata={
                        "agent": "Data Storyteller",
                        "attempt": attempt + 1,
                    },
                )

                storyteller = create_data_storyteller(
                    model=story_model
                )

                agent_3_prompt = f"""
Create the final business story from the verified
analytical evidence below.

Context:
{story_context}

STRICT RULES:
- Use only the supplied evidence.
- Never invent numbers.
- Never change SQL result values.
- Never invent dimensions or categories.
- Never contradict verified insights.
- Never invent causes.
- Do not create unsupported recommendations.
- If evidence is insufficient, recommendations must be empty.
- Explain the supplied visualization accurately.
- Return only the structured BusinessStory.

The SQL result is the authoritative source for numerical values.
"""

                agent_3_result = Runner.run_sync(
                    storyteller,
                    agent_3_prompt,
                )

                story = agent_3_result.final_output

                if not isinstance(
                    story,
                    BusinessStory,
                ):
                    raise ValueError(
                        "Agent 3 returned an invalid "
                        "structured output."
                    )

                valid_story, story_reason = (
                    validate_story_output(
                        story,
                        context.insights,
                    )
                )

                if not valid_story:
                    raise ValueError(
                        story_reason
                    )

                factual_story_valid, factual_reason = (
                    validate_story_factual_claims(
                        story,
                        context.sql_result,
                    )
                )

                if not factual_story_valid:
                    raise ValueError(
                        factual_reason
                    )

                context.story = story

                tracer.add_event(
                    event_type=TraceEventType.STORY,
                    message="Data Storyteller completed.",
                    metadata={
                        "story": story.model_dump(),
                        "attempt": attempt + 1,
                    },
                )

                break

            except (ValueError, RuntimeError) as error:

                last_story_error = error

                tracer.add_event(
                    event_type=TraceEventType.ERROR,
                    message="Data Storyteller attempt failed.",
                    metadata={
                        "attempt": attempt + 1,
                        "error": str(error),
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
                "status": "completed",
                "session_id": context.session_id,
            },
        )

        self.last_trace = tracer.complete()

        return context