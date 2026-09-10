import re

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

        # Comparison is only meaningful for grouped results.
        if self._requires_comparison(plan):
            comparison = self._comparison(
                dataframe=dataframe,
                dimension=plan.dimensions[0],
                metric=plan.metrics[0],
            )

        # Trend analysis is performed only when the analytical
        # intent explicitly represents a trend/time-series request.
        #
        # A time range by itself does NOT mean trend analysis.
        #
        # Examples:
        #
        #   "What was revenue in 2025?"
        #       -> no trend
        #
        #   "What was total revenue by region?"
        #       -> no trend
        #
        #   "How has revenue changed over time?"
        #       -> trend
        #
        #   "Show the revenue trend by month."
        #       -> trend
        if self._requires_trend(plan) and plan.metrics:
            trend = self._trend(
                dataframe=dataframe,
                metric=plan.metrics[0],
            )

        # Anomaly analysis is independent of grouping and trend analysis.
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
        return pd.DataFrame(
            sql_result.rows,
            columns=sql_result.columns,
        )

    @staticmethod
    def _normalize_name(value: str) -> str:
        """
        Normalize natural-language names and SQL column names.

        Handles:
        - spaces: "total revenue" -> "total_revenue"
        - hyphens: "total-revenue" -> "total_revenue"
        - camelCase: "totalRevenue" -> "total_revenue"
        - PascalCase: "TotalRevenue" -> "total_revenue"
        - repeated separators
        """

        normalized = value.strip()

        # Split camelCase / PascalCase boundaries.
        normalized = re.sub(
            r"([a-z0-9])([A-Z])",
            r"\1_\2",
            normalized,
        )

        # Split acronym-to-word boundaries such as SQLQuery.
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
    def _find_matching_column(
        cls,
        requested_name: str,
        dataframe: pd.DataFrame,
    ) -> str | None:
        requested_normalized = cls._normalize_name(
            requested_name
        )

        # First try an exact normalized match.
        for column in dataframe.columns:
            if (
                cls._normalize_name(str(column))
                == requested_normalized
            ):
                return str(column)

        # Then allow aggregation words to differ.
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
        Determine whether deterministic comparison analysis
        is appropriate for the analytical plan.

        Comparison analysis requires:
        - at least one dimension
        - at least one metric

        Explicit comparisons, rankings, and distributions are
        supported. A generic "by" objective is intentionally
        supported because grouped analytical plans can benefit
        from deterministic comparison evidence.
        """

        if not plan.dimensions or not plan.metrics:
            return False

        objective = (
            plan.analytical_objective or ""
        ).lower()

        operation = (
            getattr(plan, "operation", None)
            or ""
        ).lower()

        comparison = (
            getattr(plan, "comparison", None)
            or ""
        ).lower()

        comparison_keywords = {
            "comparison",
            "compare",
            "comparison of",
            "ranking",
            "rank",
            "highest",
            "lowest",
            "top",
            "bottom",
            "distribution",
        }

        if comparison.strip():
            return True

        if any(
            keyword in objective
            for keyword in comparison_keywords
        ):
            return True

        if any(
            keyword in operation
            for keyword in comparison_keywords
        ):
            return True

        # Grouped analytical results are useful comparison evidence
        # even when Agent 1 describes the objective primarily as
        # an aggregation.
        #
        # Example:
        #
        #   "total revenue by region"
        #
        # should produce deterministic region-level comparison
        # evidence.
        if "by" in objective:
            return True

        return False

    @classmethod
    def _resolve_metric(
        cls,
        dataframe: pd.DataFrame,
        metric: str,
    ) -> str:
        resolved = cls._find_matching_column(
            metric,
            dataframe,
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
        resolved = cls._find_matching_column(
            dimension,
            dataframe,
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
        resolved_dimension = cls._resolve_dimension(
            dataframe,
            dimension,
        )

        resolved_metric = cls._resolve_metric(
            dataframe,
            metric,
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
        resolved_metric = cls._resolve_metric(
            dataframe,
            metric,
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
        resolved_metric = cls._resolve_metric(
            dataframe,
            metric,
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

        mean = float(
            valid_values.mean()
        )

        standard_deviation = float(
            valid_values.std(ddof=0)
        )

        if standard_deviation == 0:
            anomalous_rows = []
        else:
            mask = (
                (numeric_values - mean).abs()
                > 3 * standard_deviation
            )

            selected_rows = dataframe.loc[
                mask.fillna(False)
            ]

            anomalous_rows = (
                selected_rows.astype(object)
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
        """
        Return True only when the analytical plan explicitly
        represents a trend or time-series request.

        A time_range alone does not imply trend analysis.

        Examples:

            "What was the total revenue by region?"
            -> False

            "What was revenue in 2025?"
            -> False

            "How has revenue changed over time?"
            -> True

            "Show the revenue trend by month."
            -> True

        This prevents historical filtering and ordinary grouped
        aggregations from being incorrectly routed into trend
        analysis.
        """

        objective = str(
            getattr(
                plan,
                "analytical_objective",
                "",
            )
            or ""
        ).strip().lower()

        operation = str(
            getattr(
                plan,
                "operation",
                "",
            )
            or ""
        ).strip().lower()

        analytical_question = str(
            getattr(
                plan,
                "analytical_question",
                "",
            )
            or ""
        ).strip().lower()

        trend_terms = (
            "trend",
            "time series",
            "timeseries",
            "over time",
            "change over time",
            "changed over time",
            "growth over time",
            "historical trend",
        )

        # Explicit semantic classification from Agent 1.
        if objective in {
            "trend",
            "time series",
            "timeseries",
        }:
            return True

        if operation in {
            "trend",
            "time series",
            "timeseries",
        }:
            return True

        # Preserve explicit trend language from the analytical
        # question without treating every time-related question
        # as a trend.
        combined_text = " ".join(
            [
                objective,
                operation,
                analytical_question,
            ]
        )

        return any(
            term in combined_text
            for term in trend_terms
        )

    @staticmethod
    def _requires_anomaly(
        plan: AnalyticalPlan,
    ) -> bool:
        objective = (
            plan.analytical_objective or ""
        ).lower()

        operation = (
            getattr(plan, "operation", None)
            or ""
        ).lower()

        analytical_question = (
            getattr(plan, "analytical_question", None)
            or ""
        ).lower()

        combined_text = " ".join(
            [
                objective,
                operation,
                analytical_question,
            ]
        )

        return "anomal" in combined_text

    @staticmethod
    def _find_time_column(
        dataframe: pd.DataFrame,
    ) -> str | None:
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