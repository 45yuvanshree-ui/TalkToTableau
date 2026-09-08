import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from analysis import AnalysisResult
from models import AnalyticalPlan
from sql import SQLResult

from .models import VisualizationResult, VisualizationSpec


class VisualizationWorkflow:
    """
    Select, prepare, generate, and validate Plotly visualizations.

    Visualization decisions are deterministic and based on the
    analytical plan and returned data.
    """

    def run(
        self,
        plan: AnalyticalPlan,
        sql_result: SQLResult,
        analysis_result: AnalysisResult,
    ) -> tuple[VisualizationResult, go.Figure | None]:
        """Generate an appropriate visualization when possible."""

        dataframe = self._to_dataframe(sql_result)

        if dataframe.empty:
            return (
                VisualizationResult(
                    status="not_visualizable",
                ),
                None,
            )

        spec = self._select_visualization(
            plan=plan,
            dataframe=dataframe,
            analysis_result=analysis_result,
        )

        if spec is None:
            return (
                VisualizationResult(
                    status="not_visualizable",
                ),
                None,
            )

        figure = self._create_figure(
            dataframe=dataframe,
            spec=spec,
        )

        self._validate_figure(
            figure=figure,
            spec=spec,
        )

        return (
            VisualizationResult(
                status="visualized",
                spec=spec,
            ),
            figure,
        )

    @staticmethod
    def _to_dataframe(sql_result: SQLResult) -> pd.DataFrame:
        """Convert SQLResult into a DataFrame."""

        return pd.DataFrame(
            sql_result.rows,
            columns=sql_result.columns,
        )

    @staticmethod
    def _select_visualization(
        plan: AnalyticalPlan,
        dataframe: pd.DataFrame,
        analysis_result: AnalysisResult,
    ) -> VisualizationSpec | None:
        """Select a chart based on analytical semantics."""

        metric = plan.metrics[0] if plan.metrics else None
        dimension = plan.dimensions[0] if plan.dimensions else None

        if (
            analysis_result.trend is not None
            and analysis_result.trend.time_dimension in dataframe.columns
            and metric in dataframe.columns
        ):
            return VisualizationSpec(
                chart_type="line",
                x_column=analysis_result.trend.time_dimension,
                y_column=metric,
                title=f"{metric} trend",
                reason="A time-based trend analysis is available.",
            )

        if (
            analysis_result.comparison is not None
            and dimension in dataframe.columns
            and metric in dataframe.columns
        ):
            return VisualizationSpec(
                chart_type="bar",
                x_column=dimension,
                y_column=metric,
                title=f"{metric} by {dimension}",
                reason="A categorical comparison is available.",
            )

        numeric_columns = dataframe.select_dtypes(
            include="number"
        ).columns.tolist()

        if len(numeric_columns) >= 2:
            return VisualizationSpec(
                chart_type="scatter",
                x_column=numeric_columns[0],
                y_column=numeric_columns[1],
                title=(
                    f"{numeric_columns[1]} vs "
                    f"{numeric_columns[0]}"
                ),
                reason=(
                    "The result contains at least two numeric "
                    "variables suitable for relationship analysis."
                ),
            )

        return None

    @staticmethod
    def _create_figure(
        dataframe: pd.DataFrame,
        spec: VisualizationSpec,
    ) -> go.Figure:
        """Create the Plotly figure described by the specification."""

        if spec.chart_type == "line":
            working = dataframe.copy()

            working[spec.x_column] = pd.to_datetime(
                working[spec.x_column],
                errors="coerce",
            )

            working[spec.y_column] = pd.to_numeric(
                working[spec.y_column],
                errors="coerce",
            )

            working = working.dropna(
                subset=[spec.x_column, spec.y_column]
            ).sort_values(spec.x_column)

            return px.line(
                working,
                x=spec.x_column,
                y=spec.y_column,
                title=spec.title,
            )

        if spec.chart_type == "bar":
            working = dataframe.copy()

            working[spec.y_column] = pd.to_numeric(
                working[spec.y_column],
                errors="coerce",
            )

            working = working.dropna(
                subset=[spec.x_column, spec.y_column]
            )

            return px.bar(
                working,
                x=spec.x_column,
                y=spec.y_column,
                title=spec.title,
            )

        if spec.chart_type == "scatter":
            working = dataframe.copy()

            working[spec.x_column] = pd.to_numeric(
                working[spec.x_column],
                errors="coerce",
            )

            working[spec.y_column] = pd.to_numeric(
                working[spec.y_column],
                errors="coerce",
            )

            working = working.dropna(
                subset=[spec.x_column, spec.y_column]
            )

            return px.scatter(
                working,
                x=spec.x_column,
                y=spec.y_column,
                title=spec.title,
            )

        raise ValueError(
            f"Unsupported chart type: {spec.chart_type}"
        )

    @staticmethod
    def _validate_figure(
        figure: go.Figure,
        spec: VisualizationSpec,
    ) -> None:
        """Validate that the generated figure matches its specification."""

        if not isinstance(figure, go.Figure):
            raise ValueError(
                "Visualization did not produce a Plotly Figure."
            )

        if len(figure.data) == 0:
            raise ValueError(
                "Visualization contains no data traces."
            )

        if spec.chart_type == "line":
            if not any(
                trace.type == "scatter"
                and getattr(trace, "mode", None)
                in {"lines", "lines+markers"}
                for trace in figure.data
            ):
                raise ValueError(
                    "Generated figure does not contain a valid line trace."
                )
            return

        if spec.chart_type == "bar":
            if not any(
                trace.type == "bar"
                for trace in figure.data
            ):
                raise ValueError(
                    "Generated figure does not contain a valid bar trace."
                )
            return

        if spec.chart_type == "scatter":
            if not any(
                trace.type == "scatter"
                for trace in figure.data
            ):
                raise ValueError(
                    "Generated figure does not contain a valid scatter trace."
                )
            return

        raise ValueError(
            f"Unsupported chart type: {spec.chart_type}."
        )