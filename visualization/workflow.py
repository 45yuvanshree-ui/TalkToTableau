import re

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
    def _to_dataframe(
        sql_result: SQLResult,
    ) -> pd.DataFrame:
        """Convert SQLResult into a DataFrame."""

        return pd.DataFrame(
            sql_result.rows,
            columns=sql_result.columns,
        )

    @staticmethod
    def _normalize_name(value: str) -> str:
        """
        Normalize a metric or column name so semantically equivalent
        names can be compared deterministically.
        """

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
    def _find_matching_column(
        cls,
        requested_name: str | None,
        dataframe: pd.DataFrame,
    ) -> str | None:
        """
        Find the actual result column corresponding to a requested
        analytical field.

        Handles common differences such as:

        total revenue -> total_revenue
        Total Revenue -> total_revenue
        revenue total -> revenue_total
        """

        if not requested_name:
            return None

        requested_normalized = cls._normalize_name(
            requested_name
        )

        # Exact normalized match.
        for column in dataframe.columns:
            if (
                cls._normalize_name(str(column))
                == requested_normalized
            ):
                return str(column)

        # Handle common aggregation wording.
        requested_tokens = set(
            requested_normalized.split("_")
        )

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

        requested_base_tokens = (
            requested_tokens - aggregation_words
        )

        for column in dataframe.columns:
            column_normalized = cls._normalize_name(
                str(column)
            )

            column_tokens = set(
                column_normalized.split("_")
            )

            column_base_tokens = (
                column_tokens - aggregation_words
            )

            if (
                requested_base_tokens
                and requested_base_tokens
                == column_base_tokens
            ):
                return str(column)

        return None

    @classmethod
    def _resolve_metric_column(
        cls,
        plan: AnalyticalPlan,
        dataframe: pd.DataFrame,
    ) -> str | None:
        """Resolve the plan metric to an actual SQL result column."""

        if not plan.metrics:
            return None

        requested_metric = plan.metrics[0]

        direct_match = cls._find_matching_column(
            requested_name=requested_metric,
            dataframe=dataframe,
        )

        if direct_match is not None:
            return direct_match

        # If the plan says "total revenue", look for a result
        # column based on "revenue".
        requested_normalized = cls._normalize_name(
            requested_metric
        )

        requested_tokens = set(
            requested_normalized.split("_")
        )

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

        base_tokens = requested_tokens - aggregation_words

        if base_tokens:
            for column in dataframe.columns:
                column_normalized = cls._normalize_name(
                    str(column)
                )

                column_tokens = set(
                    column_normalized.split("_")
                )

                if base_tokens.issubset(column_tokens):
                    return str(column)

        return None

    @classmethod
    def _resolve_dimension_column(
        cls,
        plan: AnalyticalPlan,
        dataframe: pd.DataFrame,
    ) -> str | None:
        """Resolve the plan dimension to an actual result column."""

        if not plan.dimensions:
            return None

        return cls._find_matching_column(
            requested_name=plan.dimensions[0],
            dataframe=dataframe,
        )

    @classmethod
    def _select_visualization(
        cls,
        plan: AnalyticalPlan,
        dataframe: pd.DataFrame,
        analysis_result: AnalysisResult,
    ) -> VisualizationSpec | None:
        """
        Select a chart based on analytical semantics and actual
        SQL result columns.
        """

        metric_column = cls._resolve_metric_column(
            plan=plan,
            dataframe=dataframe,
        )

        dimension_column = cls._resolve_dimension_column(
            plan=plan,
            dataframe=dataframe,
        )

        # --------------------------------------------------
        # 1. Time-series / trend visualization
        # --------------------------------------------------

        if (
            analysis_result.trend is not None
            and analysis_result.trend.time_dimension
            in dataframe.columns
        ):
            trend_metric = cls._find_matching_column(
                requested_name=analysis_result.trend.metric,
                dataframe=dataframe,
            )

            if trend_metric is not None:
                return VisualizationSpec(
                    chart_type="line",
                    x_column=(
                        analysis_result.trend.time_dimension
                    ),
                    y_column=trend_metric,
                    title=f"{trend_metric} trend",
                    reason=(
                        "A time-based trend analysis is available."
                    ),
                )

        # --------------------------------------------------
        # 2. Categorical comparison visualization
        # --------------------------------------------------

        if (
            dimension_column is not None
            and metric_column is not None
        ):
            return VisualizationSpec(
                chart_type="bar",
                x_column=dimension_column,
                y_column=metric_column,
                title=(
                    f"{metric_column} by "
                    f"{dimension_column}"
                ),
                reason=(
                    "The analytical plan contains a dimension "
                    "and metric suitable for categorical "
                    "comparison."
                ),
            )

        # --------------------------------------------------
        # 3. Deterministic fallback for two numeric variables
        # --------------------------------------------------

        numeric_columns = dataframe.select_dtypes(
            include="number"
        ).columns.tolist()

        if len(numeric_columns) >= 2:
            return VisualizationSpec(
                chart_type="scatter",
                x_column=str(numeric_columns[0]),
                y_column=str(numeric_columns[1]),
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
                subset=[
                    spec.x_column,
                    spec.y_column,
                ]
            ).sort_values(
                spec.x_column
            )

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
                subset=[
                    spec.x_column,
                    spec.y_column,
                ]
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
                subset=[
                    spec.x_column,
                    spec.y_column,
                ]
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
                    "Generated figure does not contain "
                    "a valid line trace."
                )
            return

        if spec.chart_type == "bar":
            if not any(
                trace.type == "bar"
                for trace in figure.data
            ):
                raise ValueError(
                    "Generated figure does not contain "
                    "a valid bar trace."
                )
            return

        if spec.chart_type == "scatter":
            if not any(
                trace.type == "scatter"
                for trace in figure.data
            ):
                raise ValueError(
                    "Generated figure does not contain "
                    "a valid scatter trace."
                )
            return

        raise ValueError(
            f"Unsupported chart type: {spec.chart_type}."
        )