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

        if self._requires_comparison(plan):
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
        Normalize a field name for deterministic matching.

        Examples:

        total revenue -> total_revenue
        Total Revenue -> total_revenue
        total-revenue -> total_revenue
        """

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
    def _find_matching_column(
        cls,
        requested_name: str,
        dataframe: pd.DataFrame,
    ) -> str | None:
        """
        Resolve a requested analytical field to an actual
        SQL result column.
        """

        requested_normalized = cls._normalize_name(
            requested_name
        )

        # --------------------------------------------------
        # 1. Exact normalized match
        # --------------------------------------------------

        for column in dataframe.columns:
            if (
                cls._normalize_name(str(column))
                == requested_normalized
            ):
                return str(column)

        # --------------------------------------------------
        # 2. Aggregation-aware matching
        # --------------------------------------------------

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
                    == column_base_tokens
                ):
                    return str(column)

        return None

    @classmethod
    def _requires_comparison(
        cls,
        plan: AnalyticalPlan,
    ) -> bool:
        """
        Determine whether the analytical plan represents a
        categorical comparison.

        A request such as:

        "total revenue by region"

        is a comparison even if the LLM does not explicitly
        populate the comparison field.
        """

        if not plan.dimensions or not plan.metrics:
            return False

        objective = plan.analytical_objective.lower()

        comparison_keywords = {
            "comparison",
            "compare",
            "comparison of",
            "by",
            "ranking",
            "rank",
            "highest",
            "lowest",
            "top",
            "bottom",
            "distribution",
        }

        return (
            plan.comparison is not None
            or any(
                keyword in objective
                for keyword in comparison_keywords
            )
        )

    @classmethod
    def _resolve_metric(
        cls,
        dataframe: pd.DataFrame,
        metric: str,
    ) -> str:
        """
        Resolve the metric requested by Agent 1 to an actual
        SQL result column.
        """

        resolved = cls._find_matching_column(
            requested_name=metric,
            dataframe=dataframe,
        )

        if resolved is not None:
            return resolved

        raise ValueError(
            f"Analysis metric not found in SQL result: {metric}"
        )

    @classmethod
    def _resolve_dimension(
        cls,
        dataframe: pd.DataFrame,
        dimension: str,
    ) -> str:
        """
        Resolve the requested dimension to an actual SQL
        result column.
        """

        resolved = cls._find_matching_column(
            requested_name=dimension,
            dataframe=dataframe,
        )

        if resolved is not None:
            return resolved

        raise ValueError(
            f"Analysis dimension not found in SQL result: {dimension}"
        )

    @classmethod
    def _comparison(
        cls,
        dataframe: pd.DataFrame,
        dimension: str,
        metric: str,
    ) -> ComparisonAnalysis:
        """Aggregate a metric by a requested dimension."""

        resolved_dimension = cls._resolve_dimension(
            dataframe=dataframe,
            dimension=dimension,
        )

        resolved_metric = cls._resolve_metric(
            dataframe=dataframe,
            metric=metric,
        )

        numeric_metric = pd.to_numeric(
            dataframe[resolved_metric],
            errors="coerce",
        )

        working = dataframe.copy()

        working["_analysis_metric"] = numeric_metric

        grouped = (
            working.groupby(
                resolved_dimension
            )["_analysis_metric"]
            .sum()
            .dropna()
        )

        values = {
            str(key): float(value)
            for key, value in grouped.items()
        }

        return ComparisonAnalysis(
            dimension=resolved_dimension,
            metric=resolved_metric,
            values=values,
        )

    @classmethod
    def _trend(
        cls,
        dataframe: pd.DataFrame,
        metric: str,
    ) -> TrendAnalysis:
        """
        Analyze a metric across the first suitable time-like column.
        """

        resolved_metric = cls._resolve_metric(
            dataframe=dataframe,
            metric=metric,
        )

        time_column = cls._find_time_column(
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
            working[resolved_metric],
            errors="coerce",
        )

        working = working.dropna(
            subset=[
                "_analysis_time",
                "_analysis_metric",
            ]
        )

        grouped = (
            working.groupby(
                "_analysis_time"
            )["_analysis_metric"]
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
            metric=resolved_metric,
            values=values,
        )

    @classmethod
    def _anomaly(
        cls,
        dataframe: pd.DataFrame,
        metric: str,
    ) -> AnomalyAnalysis:
        """Detect statistical outliers using a three-standard-deviation rule."""

        resolved_metric = cls._resolve_metric(
            dataframe=dataframe,
            metric=metric,
        )

        numeric_values = pd.to_numeric(
            dataframe[resolved_metric],
            errors="coerce",
        )

        valid_values = numeric_values.dropna()

        if valid_values.empty:
            return AnomalyAnalysis(
                metric=resolved_metric
            )

        mean = float(valid_values.mean())

        standard_deviation = float(
            valid_values.std(
                ddof=0
            )
        )

        if standard_deviation == 0:
            anomalous_rows: list[
                dict[str, object]
            ] = []

        else:
            mask = (
                (
                    numeric_values
                    - mean
                ).abs()
                > 3 * standard_deviation
            )

            selected_rows = dataframe.loc[
                mask.fillna(False)
            ]

            anomalous_rows = (
                selected_rows
                .astype(object)
                .where(
                    selected_rows.notna(),
                    None,
                )
                .to_dict(
                    orient="records"
                )
            )

        return AnomalyAnalysis(
            metric=resolved_metric,
            mean=mean,
            standard_deviation=standard_deviation,
            anomalous_rows=anomalous_rows,
        )

    @staticmethod
    def _requires_trend(
        plan: AnalyticalPlan,
    ) -> bool:
        """Determine whether trend analysis was requested."""

        objective = (
            plan.analytical_objective
            .lower()
        )

        return (
            "trend" in objective
            or "time" in objective
            or plan.time_range is not None
        )

    @staticmethod
    def _requires_anomaly(
        plan: AnalyticalPlan,
    ) -> bool:
        """Determine whether anomaly analysis was requested."""

        objective = (
            plan.analytical_objective
            .lower()
        )

        return "anomal" in objective

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

            if (
                converted.notna().mean()
                >= 0.8
            ):
                return str(column)

        return None