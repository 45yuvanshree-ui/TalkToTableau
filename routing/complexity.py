
"""
Deterministic analytical complexity extraction for adaptive routing.

This module converts a natural-language business question into the
characteristics already expected by RoutingRequest.

It intentionally does not call an LLM. The existing AdaptiveRouter
remains responsible for converting these characteristics into a model tier.
"""

from __future__ import annotations

import re

from .models import RoutingRequest


class AnalyticalComplexityAnalyzer:
    """
    Estimate analytical characteristics from a business question.

    The analyzer is deliberately deterministic so that routing decisions
    remain reproducible, inexpensive, and easy to test.
    """

    _AGGREGATION_PATTERNS = (
        r"\btotal\b",
        r"\bsum\b",
        r"\baverage\b",
        r"\bavg\b",
        r"\bcount\b",
        r"\bnumber of\b",
        r"\bmaximum\b",
        r"\bminimum\b",
        r"\bmax\b",
        r"\bmin\b",
    )

    _GROUPING_PATTERNS = (
        r"\bby region\b",
        r"\bby product\b",
        r"\bby category\b",
        r"\bby customer\b",
        r"\bby customer type\b",
        r"\bby year\b",
        r"\bby month\b",
        r"\bby quarter\b",
        r"\bfor each\b",
        r"\beach region\b",
        r"\beach product\b",
        r"\beach category\b",
        r"\beach customer\b",
        r"\beach year\b",
    )

    _COMPARISON_PATTERNS = (
        r"\bcompare\b",
        r"\bcomparison\b",
        r"\bcompared with\b",
        r"\bcompared to\b",
        r"\bversus\b",
        r"\bvs\.?\b",
        r"\bhigher than\b",
        r"\blower than\b",
        r"\bgreater than\b",
        r"\bless than\b",
        r"\bmore than\b",
        r"\bmost\b",
        r"\bleast\b",
        r"\bhighest\b",
        r"\blowest\b",
        r"\btop\b",
        r"\bbottom\b",
        r"\bstrongest\b",
        r"\bweakest\b",
        r"\bbetter than\b",
        r"\bworse than\b",
        r"\bimprovement\b",
        r"\bchange\b",
        r"\bdifference\b",
        r"\bvariance\b",
        r"\brank(?:ing|ed)?\b",
    )

    _CONDITIONAL_PATTERNS = (
        r"\babove average\b",
        r"\bbelow average\b",
        r"\babove the average\b",
        r"\bbelow the average\b",
        r"\babove (?:the )?overall average\b",
        r"\bbelow (?:the )?overall average\b",
        r"\bexceeds\b",
        r"\bexceeding\b",
        r"\bgreater than the average\b",
        r"\bless than the average\b",
        r"\bmore than the average\b",
        r"\bdespite\b",
        r"\bwhile\b",
        r"\bwhere\b",
        r"\bthreshold\b",
    )

    _BENCHMARK_PATTERNS = (
        r"\babove average\b",
        r"\bbelow average\b",
        r"\babove the average\b",
        r"\bbelow the average\b",
        r"\babove (?:the )?overall average\b",
        r"\bbelow (?:the )?overall average\b",
        r"\bgreater than the average\b",
        r"\bless than the average\b",
        r"\bmore than the average\b",
        r"\bcompared with the overall average\b",
        r"\bcompared to the overall average\b",
    )

    _TEMPORAL_PATTERNS = (
        r"\byear\b",
        r"\byearly\b",
        r"\bmonthly\b",
        r"\bmonth\b",
        r"\bquarter\b",
        r"\bquarterly\b",
        r"\bover time\b",
        r"\bacross time\b",
        r"\bhistorical\b",
        r"\bhistory\b",
        r"\bfrom \d{4} to \d{4}\b",
        r"\bbetween \d{4} and \d{4}\b",
        r"\byear[- ]over[- ]year\b",
        r"\byoy\b",
        r"\b\d{4}\s+to\s+\d{4}\b",
    )

    # Derived metrics generally require expressions rather than a simple
    # aggregation, e.g. profit = revenue - cost or profit margin.
    _DERIVED_METRIC_PATTERNS = (
        r"\bprofit margin\b",
        r"\bprofitability\b",
        r"\bprofit\b",
        r"\bmargin\b",
        r"\bgrowth rate\b",
        r"\bgrowth\b",
        r"\brate\b",
        r"\bpercentage\b",
        r"\bpercent\b",
        r"\bchange in\b",
        r"\bdifference between\b",
    )

    # Multiple requested business measures increase both SQL and result
    # complexity. These are common analytical metrics in BI questions.
    _METRIC_PATTERNS = (
        r"\brevenue\b",
        r"\bcost\b",
        r"\bprofit\b",
        r"\bprofit margin\b",
        r"\bquantity\b",
        r"\bquantity sold\b",
        r"\bsales\b",
        r"\bmargin\b",
        r"\bgrowth\b",
    )

    _REASONING_PATTERNS = (
        r"\bexplain\b",
        r"\bwhy\b",
        r"\banalyze\b",
        r"\banalysis\b",
        r"\bdetermine\b",
        r"\bidentify\b",
        r"\binterpret\b",
        r"\binsight\b",
        r"\binsights\b",
        r"\bfactors\b",
        r"\bdrivers\b",
        r"\breason\b",
        r"\btrend\b",
        r"\btrends\b",
        r"\bimpact\b",
        r"\brelationship\b",
        r"\bcontributing\b",
        r"\bstrongest\b",
        r"\bweakest\b",
        r"\brank(?:ing|ed)?\b",
    )

    _OUTPUT_PATTERNS = (
        r"\bexplain\b",
        r"\banalyze\b",
        r"\banalysis\b",
        r"\binsight\b",
        r"\binsights\b",
        r"\breport\b",
        r"\bstory\b",
        r"\brecommend\b",
        r"\brecommendation\b",
        r"\brecommendations\b",
    )

    @classmethod
    def analyze(
        cls,
        user_request: str,
        schema_size: int = 1,
    ) -> RoutingRequest:
        """
        Convert a natural-language request into a RoutingRequest.
        """
        text = cls._normalize(user_request)

        aggregation = cls._count_matches(
            text,
            cls._AGGREGATION_PATTERNS,
        )

        grouping = cls._count_matches(
            text,
            cls._GROUPING_PATTERNS,
        )

        comparison = cls._count_matches(
            text,
            cls._COMPARISON_PATTERNS,
        )

        conditional = cls._count_matches(
            text,
            cls._CONDITIONAL_PATTERNS,
        )

        temporal = cls._count_matches(
            text,
            cls._TEMPORAL_PATTERNS,
        )

        benchmark_comparison = cls._count_matches(
            text,
            cls._BENCHMARK_PATTERNS,
        )

        derived_metric = cls._count_matches(
            text,
            cls._DERIVED_METRIC_PATTERNS,
        )

        metric_count = cls._count_matches(
            text,
            cls._METRIC_PATTERNS,
        )

        reasoning = cls._count_matches(
            text,
            cls._REASONING_PATTERNS,
        )

        output_signals = cls._count_matches(
            text,
            cls._OUTPUT_PATTERNS,
        )

        sql_complexity = cls._estimate_sql_complexity(
            aggregation=aggregation,
            grouping=grouping,
            comparison=comparison,
            conditional=conditional,
            temporal=temporal,
            derived_metric=derived_metric,
            metric_count=metric_count,
        )

        ambiguity = cls._estimate_ambiguity(
            text=text,
            comparison=comparison,
            conditional=conditional,
            temporal=temporal,
        )

        reasoning_required = cls._requires_reasoning(
            reasoning=reasoning,
            comparison=comparison,
            conditional=conditional,
            temporal=temporal,
            derived_metric=derived_metric,
            benchmark_comparison=benchmark_comparison,
        )

        output_complexity = cls._estimate_output_complexity(
            output_signals=output_signals,
            comparison=comparison,
            conditional=conditional,
            temporal=temporal,
            reasoning_required=reasoning_required,
            metric_count=metric_count,
            derived_metric=derived_metric,
        )

        return RoutingRequest(
            user_request=user_request,
            sql_complexity=sql_complexity,
            ambiguity=ambiguity,
            reasoning_required=reasoning_required,
            output_complexity=output_complexity,
            schema_size=max(schema_size, 1),
        )

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(
            r"\s+",
            " ",
            text.lower().strip(),
        )

    @staticmethod
    def _count_matches(
        text: str,
        patterns: tuple[str, ...],
    ) -> int:
        return sum(
            1
            for pattern in patterns
            if re.search(pattern, text)
        )

    @staticmethod
    def _estimate_sql_complexity(
        *,
        aggregation: int,
        grouping: int,
        comparison: int,
        conditional: int,
        temporal: int,
        derived_metric: int,
        metric_count: int,
    ) -> int:
        """
        Estimate SQL complexity on the existing 1-5 scale.

        Complexity reflects analytical operations rather than question length.
        """
        complexity = 1

        if aggregation:
            complexity += 1

        if grouping:
            complexity += 1

        if comparison:
            complexity += 1

        if conditional:
            complexity += 1

        if temporal:
            complexity += 1

        # Derived metrics and several requested measures can make a query
        # substantially more involved even when explicit SQL keywords
        # such as "compare" are absent.
        if derived_metric >= 2:
            complexity += 1
        elif derived_metric and metric_count >= 3:
            complexity += 1

        return min(complexity, 5)

    @staticmethod
    def _estimate_ambiguity(
        *,
        text: str,
        comparison: int,
        conditional: int,
        temporal: int,
    ) -> int:
        """
        Estimate ambiguity conservatively.

        Explicit analytical wording is treated as less ambiguous.
        Short, vague requests receive a higher ambiguity score.
        """
        if len(text.split()) <= 3:
            return 2

        score = 1

        if comparison and conditional:
            score += 1

        if temporal and comparison:
            score += 1

        return min(score, 5)

    @staticmethod
    def _requires_reasoning(
        *,
        reasoning: int,
        comparison: int,
        conditional: int,
        temporal: int,
        derived_metric: int,
        benchmark_comparison: int,
    ) -> bool:
        """
        Mark requests requiring interpretation or multi-step reasoning.

        Derived metrics compared against a benchmark are reasoning-intensive
        because they typically require calculating the metric, calculating
        the benchmark, comparing the results, and potentially filtering or
        ranking the resulting entities.
        """
        if reasoning:
            return True

        if conditional and comparison:
            return True

        if temporal and comparison:
            return True

        # A derived metric explicitly compared against an average/benchmark
        # is a multi-step analytical operation.
        if derived_metric and benchmark_comparison:
            return True

        # Preserve the broader derived-metric + conditional rule as a
        # fallback for equivalent benchmark-style wording.
        if derived_metric and conditional:
            return True

        # Multiple derived calculations plus comparison can also require
        # interpretation across several analytical steps.
        if derived_metric >= 2 and comparison:
            return True

        return False

    @staticmethod
    def _estimate_output_complexity(
        *,
        output_signals: int,
        comparison: int,
        conditional: int,
        temporal: int,
        reasoning_required: bool,
        metric_count: int,
        derived_metric: int,
    ) -> int:
        complexity = 1

        if comparison:
            complexity += 1

        if conditional:
            complexity += 1

        if temporal:
            complexity += 1

        if output_signals:
            complexity += 1

        if reasoning_required:
            complexity += 1

        # Multiple business metrics increase the amount of result data
        # that must be interpreted and communicated.
        if metric_count >= 3:
            complexity += 1

        if derived_metric >= 2:
            complexity += 1

        return min(complexity, 5)


def infer_routing_request(
    user_request: str,
    schema_size: int = 1,
) -> RoutingRequest:
    """
    Convenience wrapper used by orchestration.
    """
    return AnalyticalComplexityAnalyzer.analyze(
        user_request=user_request,
        schema_size=schema_size,
    )
