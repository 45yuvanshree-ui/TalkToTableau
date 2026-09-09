from guardrails import validate_story_factual_claims
from models import BusinessStory, StorySection
from sql import SQLResult


def make_sql_result() -> SQLResult:
    return SQLResult(
        columns=[
            "region",
            "total_revenue",
        ],
        rows=[
            ["West", 33600.0],
            ["North", 60000.0],
            ["East", 31800.0],
            ["South", 55600.0],
        ],
        row_count=4,
    )


def make_story(
    summary: str,
    sections: list[StorySection] | None = None,
    visualization_explanation: str = (
        "The bar chart shows total revenue by region."
    ),
) -> BusinessStory:
    return BusinessStory(
        summary=summary,
        sections=sections or [
            StorySection(
                heading="Analysis",
                content=summary,
            )
        ],
        visualization_explanation=(
            visualization_explanation
        ),
        recommendations=[],
    )


def test_accepts_correct_numeric_values():
    story = make_story(
        "North generated $60,000 in total revenue."
    )

    is_valid, reason = validate_story_factual_claims(
        story,
        make_sql_result(),
    )

    assert is_valid is True
    assert reason is None


def test_rejects_hallucinated_numeric_value():
    story = make_story(
        "East generated $33,800 in total revenue."
    )

    is_valid, reason = validate_story_factual_claims(
        story,
        make_sql_result(),
    )

    assert is_valid is False
    assert "33800" in reason


def test_rejects_another_hallucinated_numeric_value():
    story = make_story(
        "West generated $34,600 in total revenue."
    )

    is_valid, reason = validate_story_factual_claims(
        story,
        make_sql_result(),
    )

    assert is_valid is False
    assert "34600" in reason


def test_accepts_correct_highest_claim():
    story = make_story(
        "North has the highest total revenue at $60,000."
    )

    is_valid, reason = validate_story_factual_claims(
        story,
        make_sql_result(),
    )

    assert is_valid is True
    assert reason is None


def test_rejects_incorrect_highest_claim():
    story = make_story(
        "West has the highest total revenue at $33,600."
    )

    is_valid, reason = validate_story_factual_claims(
        story,
        make_sql_result(),
    )

    assert is_valid is False
    assert "highest" in reason.lower()


def test_accepts_correct_lowest_claim():
    story = make_story(
        "East has the lowest total revenue at $31,800."
    )

    is_valid, reason = validate_story_factual_claims(
        story,
        make_sql_result(),
    )

    assert is_valid is True
    assert reason is None


def test_rejects_incorrect_lowest_claim():
    story = make_story(
        "West has the lowest total revenue at $33,600."
    )

    is_valid, reason = validate_story_factual_claims(
        story,
        make_sql_result(),
    )

    assert is_valid is False
    assert "lowest" in reason.lower()