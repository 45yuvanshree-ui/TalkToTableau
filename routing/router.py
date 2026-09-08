from .models import (
    ModelTier,
    RoutingDecision,
    RoutingRequest,
)


class AdaptiveRouter:
    """
    Select an appropriate logical model tier based on analytical
    complexity rather than simple request length.
    """

    def route(
        self,
        request: RoutingRequest,
    ) -> RoutingDecision:
        """Return the model tier best suited to the request."""

        score = self._calculate_complexity_score(request)

        if score >= 70:
            return RoutingDecision(
                tier=ModelTier.REASONING,
                reason=(
                    "The request requires substantial reasoning "
                    "or has high analytical complexity."
                ),
                complexity_score=score,
            )

        if score >= 40:
            return RoutingDecision(
                tier=ModelTier.BALANCED,
                reason=(
                    "The request has moderate analytical "
                    "complexity."
                ),
                complexity_score=score,
            )

        return RoutingDecision(
            tier=ModelTier.FAST,
            reason=(
                "The request has relatively low analytical "
                "complexity."
            ),
            complexity_score=score,
        )

    @staticmethod
    def _calculate_complexity_score(
        request: RoutingRequest,
    ) -> int:
        """
        Calculate a bounded complexity score from multiple factors.

        The score considers analytical characteristics rather than
        relying on the number of words in the request.
        """

        score = 0

        score += request.sql_complexity * 6
        score += request.ambiguity * 5
        score += request.output_complexity * 5

        if request.reasoning_required:
            score += 30

        if request.schema_size >= 20:
            score += 10
        elif request.schema_size >= 10:
            score += 5

        return min(max(score, 1), 100)