from typing import Any

import pandas as pd

from models import AnalyticalPlan
from sql import SQLResult

from .models import (
    AnalysisResult,
    AnalysisSummary,
    AnomalyAnalysis,
    ComparisonAnalysis,
    TrendAnalysis,
)


class AnalysisWorkflow:
    """
    Perform deterministic analysis on validated SQL results.

    This workflow does not generate business insights.
    It prepares structured analytical evidence for Agent 2.
    """

    def run(
        self,
        plan: AnalyticalPlan,
        sql_result: SQLResult,
    ) -> AnalysisResult:
        """Run only the analysis operations relevant to the plan."""

        dataframe = self._to_dataframe(sql_result)

        summary = AnalysisSummary(
            row_count=len(dataframe),
            columns=[str(column) for column in dataframe.columns],
        )

        comparison = None
        trend = None
        anomaly = None

        if plan.comparison and plan.dimensions and plan.metrics:
            comparison = self._comparison(
                dataframe=dataframe,
                dimension=plan.dimensions[0],
                metric=plan.metrics[0],
            )

        if self._requires_trend(plan) and plan.metrics:
            trend = self._trend(
                dataframe=dataframe,
                metric=plan.metrics[0],
            )

        if self._requires_anomaly(plan) and plan.metrics:
            anomaly = self._anomaly(
                dataframe=dataframe,
                metric=plan.metrics[0],
            )

        return AnalysisResult(
            summary=summary,
            comparison=comparison,
            trend=trend,
            anomaly=anomaly,
        )

    @staticmethod
    def _to_dataframe(sql_result: SQLResult) -> pd.DataFrame:
        """Convert SQLResult into a DataFrame."""

        return pd.DataFrame(
            sql_result.rows,
            columns=sql_result.columns,
        )

    @staticmethod
    def _requires_trend(plan: AnalyticalPlan) -> bool:
        """Determine whether trend analysis was requested."""

        objective = plan.analytical_objective.lower()

        return (
            "trend" in objective
            or "time" in objective
            or plan.time_range is not None
        )

    @staticmethod
    def _requires_anomaly(plan: AnalyticalPlan) -> bool:
        """Determine whether anomaly analysis was requested."""

        objective = plan.analytical_objective.lower()

        return "anomal" in objective

    @staticmethod
    def _comparison(
        dataframe: pd.DataFrame,
        dimension: str,
        metric: str,
    ) -> ComparisonAnalysis:
        """Aggregate a metric by a requested dimension."""

        if dimension not in dataframe.columns:
            raise ValueError(
                f"Comparison dimension not found: {dimension}"
            )

        if metric not in dataframe.columns:
            raise ValueError(
                f"Comparison metric not found: {metric}"
            )

        numeric_metric = pd.to_numeric(
            dataframe[metric],
            errors="coerce",
        )

        working = dataframe.copy()
        working["_analysis_metric"] = numeric_metric

        grouped = (
            working.groupby(dimension)["_analysis_metric"]
            .sum()
            .dropna()
        )

        values = {
            str(key): float(value)
            for key, value in grouped.items()
        }

        return ComparisonAnalysis(
            dimension=dimension,
            metric=metric,
            values=values,
        )

    @staticmethod
    def _trend(
        dataframe: pd.DataFrame,
        metric: str,
    ) -> TrendAnalysis:
        """
        Analyze a metric across the first suitable time-like column.
        """

        if metric not in dataframe.columns:
            raise ValueError(
                f"Trend metric not found: {metric}"
            )

        time_column = AnalysisWorkflow._find_time_column(
            dataframe
        )

        if time_column is None:
            raise ValueError(
                "No suitable time dimension found for trend analysis."
            )

        working = dataframe.copy()

        working["_analysis_time"] = pd.to_datetime(
            working[time_column],
            errors="coerce",
        )

        working["_analysis_metric"] = pd.to_numeric(
            working[metric],
            errors="coerce",
        )

        working = working.dropna(
            subset=["_analysis_time", "_analysis_metric"]
        )

        grouped = (
            working.groupby("_analysis_time")["_analysis_metric"]
            .sum()
            .sort_index()
        )

        values = [
            {
                "period": timestamp.isoformat(),
                "value": float(value),
            }
            for timestamp, value in grouped.items()
        ]

        return TrendAnalysis(
            time_dimension=time_column,
            metric=metric,
            values=values,
        )

    @staticmethod
    def _anomaly(
        dataframe: pd.DataFrame,
        metric: str,
    ) -> AnomalyAnalysis:
        """Detect statistical outliers using a three-standard-deviation rule."""

        if metric not in dataframe.columns:
            raise ValueError(
                f"Anomaly metric not found: {metric}"
            )

        numeric_values = pd.to_numeric(
            dataframe[metric],
            errors="coerce",
        )

        valid_values = numeric_values.dropna()

        if valid_values.empty:
            return AnomalyAnalysis(metric=metric)

        mean = float(valid_values.mean())
        standard_deviation = float(valid_values.std(ddof=0))

        if standard_deviation == 0:
            anomalous_rows: list[dict[str, object]] = []
        else:
            mask = (
                (numeric_values - mean).abs()
                > 3 * standard_deviation
            )

            anomalous_rows = (
                dataframe.loc[mask.fillna(False)]
                .astype(object)
                .where(dataframe.loc[mask.fillna(False)].notna(), None)
                .to_dict(orient="records")
            )

        return AnomalyAnalysis(
            metric=metric,
            mean=mean,
            standard_deviation=standard_deviation,
            anomalous_rows=anomalous_rows,
        )

    @staticmethod
    def _find_time_column(
        dataframe: pd.DataFrame,
    ) -> str | None:
        """Find the first column that can reasonably represent time."""

        preferred_names = [
            "date",
            "sale_date",
            "order_date",
            "transaction_date",
            "datetime",
            "timestamp",
            "time",
        ]

        lowered_columns = {
            str(column).lower(): str(column)
            for column in dataframe.columns
        }

        for name in preferred_names:
            if name in lowered_columns:
                return lowered_columns[name]

        for column in dataframe.columns:
            converted = pd.to_datetime(
                dataframe[column],
                errors="coerce",
            )

            if converted.notna().mean() >= 0.8:
                return str(column)

        return None