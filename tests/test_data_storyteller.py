from agent_components import data_storyteller
from models import BusinessStory, StorySection


def test_data_storyteller_configuration():
    assert data_storyteller.name == "Data Storyteller"
    assert data_storyteller.output_type is BusinessStory


def test_story_section_model():
    section = StorySection(
        heading="Revenue comparison",
        content="North generated more revenue than South.",
    )

    assert section.heading == "Revenue comparison"
    assert "North" in section.content


def test_business_story_model():
    story = BusinessStory(
        summary="North generated the highest revenue.",
        sections=[
            StorySection(
                heading="Revenue comparison",
                content=(
                    "North generated 2200 in revenue, "
                    "compared with 1700 for South."
                ),
            )
        ],
        visualization_explanation=(
            "The bar chart compares revenue across regions."
        ),
        recommendations=[
            "Investigate the factors contributing to North's stronger revenue."
        ],
    )

    assert "North" in story.summary
    assert len(story.sections) == 1
    assert len(story.recommendations) == 1


def test_business_story_can_have_no_recommendations():
    story = BusinessStory(
        summary="The available evidence is insufficient.",
        sections=[],
        visualization_explanation=(
            "No visualization is available."
        ),
        recommendations=[],
    )

    assert story.recommendations == []