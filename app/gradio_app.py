import os
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd

from data_sources.postgres_source import PostgresDataSource
from orchestration.workflow import TalkToTableauWorkflow


APP_TITLE = "TalkToTableau"

APP_DESCRIPTION = """
### TalkToTableau

An Agentic Business Intelligence framework for:

- Natural-language business questions
- Text-to-SQL
- Evidence-based analytics
- Visual analytics
- Data storytelling
"""

DEFAULT_DATA_FILE = "database/golden_sales.csv"

SUPPORTED_EXTENSIONS = {".csv", ".xlsx"}


def _format_value(value: Any) -> str:
    """Convert a Python value into readable UI text."""

    if value is None:
        return ""

    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))

    return str(value)


def _format_sql_result(sql_result: Any) -> str:
    """Format SQLResult as a Markdown table."""

    if sql_result is None:
        return "No SQL result available."

    columns = getattr(sql_result, "columns", [])
    rows = getattr(sql_result, "rows", [])

    if not columns:
        return "No result columns available."

    header = "| " + " | ".join(str(column) for column in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"

    table_rows = []

    for row in rows:
        values = [_format_value(value) for value in row]
        table_rows.append("| " + " | ".join(values) + " |")

    if table_rows:
        return "\n".join([header, separator, *table_rows])

    return "\n".join(
        [
            header,
            separator,
            "| " + " | ".join("" for _ in columns) + " |",
        ]
    )


def _format_plan(plan: Any) -> str:
    """Format the analytical plan for display."""

    if plan is None:
        return "No analytical plan available."

    lines = [
        f"**Objective:** {plan.analytical_objective}",
        f"**Operation:** {plan.operation}",
        f"**Metric definition:** {plan.metric_definition}",
        f"**Dimensions:** {', '.join(plan.dimensions) if plan.dimensions else 'None'}",
        f"**Metrics:** {', '.join(plan.metrics) if plan.metrics else 'None'}",
        f"**Filters:** {', '.join(plan.filters) if plan.filters else 'None'}",
        f"**Comparison:** {plan.comparison or 'None'}",
    ]

    return "\n\n".join(lines)


def _format_insights(insights: Any) -> str:
    """Format verified insights for display."""

    if insights is None:
        return "No insights available."

    insight_items = getattr(insights, "insights", [])

    if not insight_items:
        return "No insights were generated."

    lines = []

    for index, insight in enumerate(insight_items, start=1):
        statement = getattr(insight, "statement", "")
        confidence = getattr(insight, "confidence", "")

        lines.append(
            f"**{index}. {statement}**  \n"
            f"Confidence: `{confidence}`"
        )

    verification_status = getattr(
        insights,
        "verification_status",
        "unknown",
    )

    lines.append(
        f"\n**Verification status:** `{verification_status}`"
    )

    unsupported_claims = getattr(
        insights,
        "unsupported_claims",
        [],
    )

    if unsupported_claims:
        lines.append("\n**Unsupported claims:**")

        for claim in unsupported_claims:
            lines.append(f"- {claim}")

    return "\n\n".join(lines)


def _format_story(story: Any) -> str:
    """Format the generated business story."""

    if story is None:
        return "No business story available."

    lines = []

    summary = getattr(story, "summary", "")

    if summary:
        lines.append(f"**Summary**\n\n{summary}")

    sections = getattr(story, "sections", [])

    for section in sections:
        heading = getattr(section, "heading", "")
        content = getattr(section, "content", "")

        if heading or content:
            lines.append(
                f"### {heading}\n\n{content}"
            )

    recommendations = getattr(
        story,
        "recommendations",
        [],
    )

    if recommendations:
        lines.append("### Recommendations")

        for recommendation in recommendations:
            lines.append(f"- {recommendation}")

    visualization_explanation = getattr(
        story,
        "visualization_explanation",
        "",
    )

    if visualization_explanation:
        lines.append(
            "### Visualization Explanation\n\n"
            + visualization_explanation
        )

    return "\n\n".join(lines)


def _format_visualization(visualization: Any) -> str:
    """Format visualization metadata for display."""

    if visualization is None:
        return "No visualization available."

    status = getattr(
        visualization,
        "status",
        "unknown",
    )

    spec = getattr(
        visualization,
        "spec",
        None,
    )

    if spec is None:
        return f"**Status:** `{status}`"

    chart_type = getattr(
        spec,
        "chart_type",
        "unknown",
    )

    x_column = getattr(
        spec,
        "x_column",
        "unknown",
    )

    y_column = getattr(
        spec,
        "y_column",
        "unknown",
    )

    title = getattr(
        spec,
        "title",
        "unknown",
    )

    reason = getattr(
        spec,
        "reason",
        "",
    )

    return (
        f"**Status:** `{status}`\n\n"
        f"**Chart type:** `{chart_type}`\n\n"
        f"**X-axis:** `{x_column}`\n\n"
        f"**Y-axis:** `{y_column}`\n\n"
        f"**Title:** `{title}`\n\n"
        f"**Reason:** {reason}"
    )


def _build_postgres_connection_string(
    host: str,
    port: str,
    database: str,
    username: str,
    password: str,
) -> str:
    """Build a PostgreSQL connection URI safely."""

    from urllib.parse import quote

    encoded_username = quote(username or "", safe="")
    encoded_password = quote(password or "", safe="")
    encoded_database = quote(database or "", safe="")

    return (
        f"postgresql://{encoded_username}:{encoded_password}"
        f"@{host.strip()}:{port.strip()}/{encoded_database}"
    )


def _test_postgres_connection(
    host: str,
    port: str,
    database: str,
    username: str,
    password: str,
):
    """Test PostgreSQL connectivity and display accessible schema metadata."""

    fields = {
        "Host": host,
        "Port": port,
        "Database": database,
        "Username": username,
    }

    missing = [
        name
        for name, value in fields.items()
        if not str(value or "").strip()
    ]

    if missing:
        return (
            "### PostgreSQL connection failed\n\n"
            "Missing: " + ", ".join(missing),
            pd.DataFrame(),
        )

    try:
        connection_string = _build_postgres_connection_string(
            host=host,
            port=port,
            database=database,
            username=username,
            password=password,
        )

        source = PostgresDataSource(
            connection_string=connection_string,
        )

        if not source.test_connection():
            return (
                "### PostgreSQL connection failed\n\n"
                "The database could not be reached with the "
                "provided connection details.",
                pd.DataFrame(),
            )

        schema = source.get_schema()
        schema_dataframe = pd.DataFrame(schema)

        return (
            "### PostgreSQL connection successful\n\n"
            f"Found **{len(schema_dataframe)} column definitions** "
            "in the accessible database.",
            schema_dataframe,
        )

    except Exception as exc:
        return (
            "### PostgreSQL connection failed\n\n"
            f"`{type(exc).__name__}: {exc}`",
            pd.DataFrame(),
        )


def _load_dataset_preview(file_path: str | None):
    """
    Load the uploaded CSV/XLSX file and return its first five rows.

    The complete set of columns is preserved in the preview DataFrame.
    """

    if not file_path:
        return (
            "No dataset uploaded yet.",
            pd.DataFrame(),
        )

    path = Path(file_path)
    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        return (
            "Unsupported file type. Please upload a CSV or XLSX file.",
            pd.DataFrame(),
        )

    try:
        if extension == ".csv":
            try:
                dataframe = pd.read_csv(
                    path,
                    encoding="utf-8",
                )
            except UnicodeDecodeError:
                dataframe = pd.read_csv(
                    path,
                    encoding="cp1252",
                )
        else:
            dataframe = pd.read_excel(path)

        preview = dataframe.head(5)

        info = (
            f"### Dataset loaded successfully\n\n"
            f"**File:** `{path.name}`  \n"
            f"**Rows:** {len(dataframe):,}  \n"
            f"**Columns:** {len(dataframe.columns):,}"
        )

        return info, preview

    except Exception as exc:
        return (
            "### Dataset preview failed\n\n"
            f"`{type(exc).__name__}: {exc}`",
            pd.DataFrame(),
        )


def analyze_question(
    question: str,
    source_type: str,
    uploaded_file: str | None,
    postgres_host: str,
    postgres_port: str,
    postgres_database: str,
    postgres_username: str,
    postgres_password: str,
) -> tuple[str, str, str, str, str, Any, str]:
    """
    Execute the existing TalkToTableau workflow.

    CSV/XLSX uses the existing uploaded-file path. PostgreSQL uses
    the existing PostgresDataSource implementation. Both paths then
    enter the same TalkToTableau workflow.

    Returns:
        status,
        analytical_plan,
        generated_sql,
        sql_result,
        insights,
        figure,
        story
    """

    if not question or not question.strip():
        message = "Please enter a business question."

        return (
            message,
            "",
            "",
            "",
            "",
            None,
            "",
        )

    try:
        if source_type == "PostgreSQL":
            fields = {
                "Host": postgres_host,
                "Port": postgres_port,
                "Database": postgres_database,
                "Username": postgres_username,
            }

            missing = [
                name
                for name, value in fields.items()
                if not str(value or "").strip()
            ]

            if missing:
                return (
                    "### Analysis failed\n\n"
                    "Missing PostgreSQL fields: "
                    + ", ".join(missing),
                    "",
                    "",
                    "",
                    "",
                    None,
                    "",
                )

            connection_string = _build_postgres_connection_string(
                host=postgres_host,
                port=postgres_port,
                database=postgres_database,
                username=postgres_username,
                password=postgres_password,
            )

            data_source = PostgresDataSource(
                connection_string=connection_string,
            )

            if not data_source.test_connection():
                return (
                    "### Analysis failed\n\n"
                    "PostgreSQL connection could not be established.",
                    "",
                    "",
                    "",
                    "",
                    None,
                    "",
                )

            workflow = TalkToTableauWorkflow(
                data_source=data_source,
            )
            dataset_label = f"PostgreSQL: {postgres_database}"

        else:
            data_file = uploaded_file or DEFAULT_DATA_FILE

            workflow = TalkToTableauWorkflow(
                data_file=data_file,
            )
            dataset_label = Path(data_file).name

        session = workflow.run(
            question.strip(),
        )

        status = (
            "### Analysis completed successfully.\n\n"
            f"Session ID: `{session.session_id}`\n\n"
            f"Data source: `{dataset_label}`"
        )

        analytical_plan = _format_plan(
            session.analytical_plan,
        )

        generated_sql = session.generated_sql or (
            "No SQL was generated."
        )

        sql_result = _format_sql_result(
            session.sql_result,
        )

        insights = _format_insights(
            session.insights,
        )

        visualization = _format_visualization(
            session.visualization,
        )

        story = _format_story(
            session.story,
        )

        figure = workflow.last_figure

        return (
            status,
            analytical_plan,
            generated_sql,
            sql_result,
            insights,
            figure,
            story,
        )

    except Exception as exc:
        status = (
            "### Analysis failed\n\n"
            f"`{type(exc).__name__}: {exc}`"
        )

        return (
            status,
            "",
            "",
            "",
            "",
            None,
            "",
        )


def create_app() -> gr.Blocks:
    """Create the TalkToTableau Gradio application."""

    with gr.Blocks(
        title=APP_TITLE,
    ) as app:

        gr.Markdown(
            f"# {APP_TITLE}"
        )

        gr.Markdown(
            APP_DESCRIPTION
        )

        # ==========================================================
        # 1. DATA SOURCE
        # ==========================================================

        gr.Markdown(
            "## 1. Select Data Source"
        )

        source_type = gr.Radio(
            choices=["CSV / Excel", "PostgreSQL"],
            value="CSV / Excel",
            label="Data Source",
        )

        with gr.Group(visible=True) as file_source_group:
            dataset_file = gr.File(
                label="Upload CSV or Excel Dataset",
                file_types=[".csv", ".xlsx"],
                type="filepath",
            )

            dataset_status = gr.Markdown(
                value=(
                    "Upload a **CSV** or **XLSX** file to preview "
                    "the dataset before asking questions."
                )
            )

            dataset_preview = gr.Dataframe(
                label="Dataset Preview — First 5 Rows",
                headers=None,
                datatype=None,
                interactive=False,
                wrap=False,
            )

        with gr.Group(visible=False) as postgres_source_group:
            gr.Markdown("### PostgreSQL Connection")

            postgres_host = gr.Textbox(
                label="Host",
                placeholder="localhost",
            )

            postgres_port = gr.Textbox(
                label="Port",
                value="5432",
            )

            postgres_database = gr.Textbox(
                label="Database",
                placeholder="my_database",
            )

            postgres_username = gr.Textbox(
                label="Username",
                placeholder="postgres",
            )

            postgres_password = gr.Textbox(
                label="Password",
                type="password",
            )

            test_postgres_button = gr.Button(
                "Test PostgreSQL Connection",
            )

            postgres_status = gr.Markdown(
                value=(
                    "Enter the PostgreSQL connection details and "
                    "click **Test PostgreSQL Connection**."
                )
            )

            postgres_schema_preview = gr.Dataframe(
                label="PostgreSQL Schema",
                interactive=False,
                wrap=False,
            )

        def switch_source(selected_source: str):
            is_postgres = selected_source == "PostgreSQL"

            return (
                gr.update(visible=not is_postgres),
                gr.update(visible=is_postgres),
            )

        source_type.change(
            fn=switch_source,
            inputs=source_type,
            outputs=[
                file_source_group,
                postgres_source_group,
            ],
        )

        dataset_file.change(
            fn=_load_dataset_preview,
            inputs=dataset_file,
            outputs=[
                dataset_status,
                dataset_preview,
            ],
        )

        test_postgres_button.click(
            fn=_test_postgres_connection,
            inputs=[
                postgres_host,
                postgres_port,
                postgres_database,
                postgres_username,
                postgres_password,
            ],
            outputs=[
                postgres_status,
                postgres_schema_preview,
            ],
        )

        # ==========================================================
        # 2. BUSINESS QUESTION
        # ==========================================================

        gr.Markdown(
            "## 2. Ask a Business Question"
        )

        user_question = gr.Textbox(
            label="Business Question",
            placeholder=(
                "Example: What was the total revenue "
                "by region?"
            ),
            lines=3,
        )

        gr.Markdown(
            "**Example questions:** "
            "Total revenue by region · "
            "Which product generated the highest profit? · "
            "How many units were sold by customer type?"
        )

        run_button = gr.Button(
            "Run Analysis",
            variant="primary",
        )

        status_output = gr.Markdown(
            value=(
                "Select a data source, provide the dataset or "
                "PostgreSQL connection, enter a business question, "
                "and click **Run Analysis**."
            )
        )

        # ==========================================================
        # 3. ANALYSIS RESULTS
        # ==========================================================

        gr.Markdown(
            "## 3. Analysis Results"
        )

        with gr.Accordion("Analytical Plan", open=False):
            analytical_plan_output = gr.Markdown()

        with gr.Accordion("Generated SQL", open=False):
            generated_sql_output = gr.Code(
                language="sql"
            )

        with gr.Accordion("SQL Result", open=True):
            sql_result_output = gr.Markdown()

        with gr.Accordion("Verified Insights", open=True):
            insights_output = gr.Markdown()

        with gr.Accordion("Visualization", open=True):
            visualization_plot = gr.Plot(
                label="Generated Visualization",
            )

        with gr.Accordion("Business Story", open=True):
            story_output = gr.Markdown()

        run_button.click(
            fn=analyze_question,
            inputs=[
                user_question,
                source_type,
                dataset_file,
                postgres_host,
                postgres_port,
                postgres_database,
                postgres_username,
                postgres_password,
            ],
            outputs=[
                status_output,
                analytical_plan_output,
                generated_sql_output,
                sql_result_output,
                insights_output,
                visualization_plot,
                story_output,
            ],
        )

    return app


if __name__ == "__main__":
    application = create_app()

    application.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
        inbrowser=False,
    )
