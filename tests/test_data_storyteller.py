from agents import Agent

from agent_components import (
    create_data_storyteller,
    data_storyteller,
)
from models import BusinessStory


def test_data_storyteller_configuration():
    assert data_storyteller.name == "Data Storyteller"
    assert data_storyteller.output_type is BusinessStory


def test_data_storyteller_can_accept_injected_model():
    agent = create_data_storyteller()

    assert isinstance(agent, Agent)
    assert agent.name == "Data Storyteller"


def test_business_story_structure():
    story = BusinessStory(
        summary="North generated the highest revenue.",
        sections=[],
        visualization_explanation=(
            "The visualization compares revenue by region."
        ),
        recommendations=[],
    )

    assert story.summary == (
        "North generated the highest revenue."
    )
    assert story.recommendations == []