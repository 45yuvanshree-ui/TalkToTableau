from routing import (
    AdaptiveRouter,
    ModelTier,
    RoutingRequest,
)


def test_simple_request_uses_fast_tier():
    router = AdaptiveRouter()

    decision = router.route(
        RoutingRequest(
            user_request="Show total revenue.",
            sql_complexity=1,
            ambiguity=1,
            reasoning_required=False,
            output_complexity=1,
            schema_size=5,
        )
    )

    assert decision.tier == ModelTier.FAST
    assert decision.complexity_score < 40


def test_moderate_request_uses_balanced_tier():
    router = AdaptiveRouter()

    decision = router.route(
        RoutingRequest(
            user_request=(
                "Compare revenue by region and customer type."
            ),
            sql_complexity=3,
            ambiguity=2,
            reasoning_required=False,
            output_complexity=3,
            schema_size=10,
        )
    )

    assert decision.tier == ModelTier.BALANCED
    assert 40 <= decision.complexity_score < 70


def test_complex_request_uses_reasoning_tier():
    router = AdaptiveRouter()

    decision = router.route(
        RoutingRequest(
            user_request=(
                "Identify unusual revenue trends and compare "
                "regional performance across time."
            ),
            sql_complexity=5,
            ambiguity=4,
            reasoning_required=True,
            output_complexity=5,
            schema_size=25,
        )
    )

    assert decision.tier == ModelTier.REASONING
    assert decision.complexity_score >= 70


def test_routing_does_not_depend_on_word_count():
    router = AdaptiveRouter()

    short_request = router.route(
        RoutingRequest(
            user_request="Revenue?",
            sql_complexity=5,
            ambiguity=4,
            reasoning_required=True,
            output_complexity=5,
            schema_size=25,
        )
    )

    long_request = router.route(
        RoutingRequest(
            user_request=(
                "Please provide a detailed analytical explanation "
                "of the requested business performance information "
                "using the available data."
            ),
            sql_complexity=1,
            ambiguity=1,
            reasoning_required=False,
            output_complexity=1,
            schema_size=5,
        )
    )

    assert short_request.tier == ModelTier.REASONING
    assert long_request.tier == ModelTier.FAST


def test_complexity_score_is_bounded():
    router = AdaptiveRouter()

    decision = router.route(
        RoutingRequest(
            user_request="Complex analysis",
            sql_complexity=5,
            ambiguity=5,
            reasoning_required=True,
            output_complexity=5,
            schema_size=100,
        )
    )

    assert 1 <= decision.complexity_score <= 100